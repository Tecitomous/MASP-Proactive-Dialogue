#!/usr/bin/env python3
"""Online-feedback DPO for P4G close timing.

This script collects preference pairs from train-split BDI-cache contexts by
sampling several assistant candidates, rolling each one through the train-side
user simulator for one user response, and scoring the resulting exchange with
the P4G success judge. It avoids held-out eval dumps and does not change BDI
generation: cached train BDIs are used only to condition the user simulator.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluate_masp import (
    _BAD_CANDIDATE_RE,
    _DONATION_ASK_RE,
    _GOODBYE_CANDIDATE_RE,
    _LOCAL_OBJECTION_RE,
    _MONEY_OBJECTION_RE,
    _LATER_OBJECTION_RE,
    _USER_QUESTION_RE,
    _recent_user_refusal_count,
)
from masp.data.bdi_dataset import BDILabelCache
from masp.eval.success_judge import build_success_judge
from masp.mind.bdi_schema import TASK_CONFIGS, BDI
from masp.models.policy import (
    LoRAPolicy,
    PolicyConfig,
    build_chat_prompt_for_policy,
    infer_lora_config_from_adapter,
)
from masp.rl.rewards import RationalityJudge
from masp.utils.io import dump_json, ensure_dir
from masp.utils.llm_client import LLMClient, LLMConfig
from masp.utils.seed import set_seed
from train_p4g_close_rule_dpo import _looks_like_user_already_committed


_USER_DONATION_COMMIT_RE = re.compile(
    r"("
    r"\b(?:yes|sure|absolutely|definitely|okay|ok)\b.{0,80}\b(?:donat|contribut|give)|"
    r"\bi\s*(?:am|will|would|can|'ll|’ll|m)?\s*(?:donat|contribut|give)\b|"
    r"\bi'?m\s+(?:willing|happy|going)\s+to\s+(?:donat|contribut|give)\b|"
    r"\b(?:donat|contribut|give)\b.{0,80}\b(?:\$?\d|cents?|dollars?)"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_SHORT_YES_RE = re.compile(r"^\s*(yes|sure|absolutely|definitely|ok|okay)[.!\\s]*$", re.I)
_SAVE_CHILDREN_TEXT_RE = re.compile(r"\bsave\s+the\s+children\b", re.I)
_UNSAFE_DPO_CHOSEN_RE = re.compile(
    r"("
    r"\$\s*(?:[1-9]|\d{2,})(?:\b|\s|[.,])|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+dollars?\b|"
    r"\b(?:month|monthly|per\s+month)\b|"
    r"\bmatch\s+your\s+(?:contribution|donation)\b|"
    r"\bmatch\s+whatever\b|\bi'?ll\s+match\b|"
    r"\bwe\s+need\s+some\s+money\b|"
    r"\bmust\s+be\s+exact\b|"
    r"\bhow\s+much\s+(?:money\s+)?(?:can|do)\s+you\s+(?:afford|have)\b|"
    r"\bhow\s+much\s+(?:do|would|will|can)\s+you\s+(?:like|want|wish)?\s*(?:to\s+)?(?:donate|contribute|give)\b|"
    r"\bhow\s+much\s+of\s+your\s+(?:task\s+payment|bonus|earnings|payment)\b|"
    r"\bwhat\s+(?:exact\s+)?amount\s+(?:are|would|will|can)\s+you\s+(?:willing|like|want|able)?\s*(?:to\s+)?(?:donate|contribute|give)?\b|"
    r"\b(?:any|whatever)\s+(?:amount|donation)\s+(?:you|would|will|can)\b|"
    r"\byou\s+can\s+choose\s+any\s+amount\b|"
    r"\bamount\s+of\s+your\s+choice\b|"
    r"\b(?:some|part|portion)\s+of\s+your\s+(?:task\s+)?(?:payment|bonus)\b|"
    r"\bor\s+more\b|"
    r"\b100\s*%\s+(?:goes?|go)\b|"
    r"\bnot\s+affiliated\s+with\s+any\s+government\b|"
    r"\byour\s+donation\s+is\s+essential\b|"
    r"\bare\s+you\s+sure\b|"
    r"\bpoor\s+needy\b|\bneedy\s+(?:children|kids)\b|"
    r"\bguilt(?:y)?\b|\bmoral\s+responsibility\b|"
    r"\bpaymen\b|\bchildrens\b|\bchilden\b|\bcildren\b|\bchidren\b|"
    r"\bdeadline\b|\bwaste\s+my\s+time\b|"
    r"\b(?:all|entire|full|whole)\s+(?:of\s+)?(?:your\s+)?(?:payment|task\s+payment|bonus|\$\d)\b|"
    r"\ball\s+the\s+way\s+(?:up\s+)?to\b|"
    r"\b(?:from|between|anything\s+from)\s+\$?0(?:\.00)?\s*(?:-|to|and|up\s+to)\b|"
    r"\b\$?0(?:\.00)?\s*(?:-|to|and)\s+\$?2\b|"
    r"\bchoose\s+\$?0(?:\.00)?\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_SAFE_CLOSE_CUE_RE = re.compile(
    r"("
    r"\$0?\.25|25\s+cents?|small\s+non-zero|non-zero\s+amount|"
    r"token\s+\$0?\.25|"
    r"any\s+other\s+non-zero"
    r")",
    re.IGNORECASE,
)


@dataclass
class OnlineContext:
    session_id: str
    turn_idx: int
    assistant_turn: int
    history_lines: List[str]
    bdi_text: str


@dataclass
class CandidateTrace:
    text: str
    user_reply: str
    reward: float
    success: bool
    label: str
    rationality: int
    score: float
    raw_judgments: List[str] = field(default_factory=list)


@dataclass
class OnlinePreferencePair:
    session_id: str
    turn_idx: int
    assistant_turn: int
    history_lines: List[str]
    bdi_text: str
    chosen: str
    rejected: str
    chosen_trace: CandidateTrace
    rejected_trace: CandidateTrace
    ref_chosen_logp: Optional[float] = None
    ref_rejected_logp: Optional[float] = None


def _assistant_turn(history_lines: Sequence[str]) -> int:
    return sum(1 for line in history_lines if str(line).startswith("Assistant:")) + 1


def _history_has_recent_user_commitment(history_lines: Sequence[str], max_lines: int = 8) -> bool:
    last_assistant = ""
    for line in history_lines[-max_lines:]:
        if str(line).startswith("Assistant:"):
            last_assistant = str(line)[10:].strip()
            continue
        if not str(line).startswith("User:"):
            continue
        user = str(line)[5:].strip()
        if _USER_DONATION_COMMIT_RE.search(user):
            return True
        if _SHORT_YES_RE.search(user) and _DONATION_ASK_RE.search(last_assistant):
            return True
    return False


def _history_mentions_save_children(history_lines: Sequence[str], max_lines: int = 12) -> bool:
    return any(_SAVE_CHILDREN_TEXT_RE.search(str(line)) for line in history_lines[-max_lines:])


def _load_contexts(args) -> List[OnlineContext]:
    cache = BDILabelCache.load(args.train_cache)
    contexts: List[OnlineContext] = []
    for entry in cache.entries:
        hist = list(entry.history_upto)
        if entry.next_speaker != "assistant":
            continue
        if not any(str(line).startswith("User:") for line in hist):
            continue
        aturn = _assistant_turn(hist)
        if aturn < int(args.min_assistant_turn) or aturn > int(args.max_assistant_turn):
            continue
        if bool(args.skip_recent_refusal) and _recent_user_refusal_count(hist) > 0:
            continue
        if bool(args.skip_user_committed) and _looks_like_user_already_committed(hist):
            continue
        if bool(args.skip_user_committed) and _history_has_recent_user_commitment(hist):
            continue
        if bool(args.require_save_children_context) and not _history_mentions_save_children(hist):
            continue
        contexts.append(OnlineContext(
            session_id=str(entry.session_id),
            turn_idx=int(entry.turn_idx),
            assistant_turn=int(aturn),
            history_lines=list(hist[-int(args.max_history_lines):]),
            bdi_text=entry.bdi.to_text() if isinstance(entry.bdi, BDI) else str(entry.bdi),
        ))
    rng = random.Random(args.seed)
    rng.shuffle(contexts)
    stride = max(int(args.context_stride), 1)
    offset = int(args.context_offset)
    if offset < 0 or offset >= stride:
        raise ValueError("--context_offset must be in [0, context_stride)")
    if stride > 1:
        contexts = contexts[offset::stride]
    if int(args.max_contexts) > 0:
        contexts = contexts[: int(args.max_contexts)]
    if not contexts:
        raise RuntimeError("no online-feedback contexts after filtering")
    return contexts


def _pair_from_dict(item: Dict) -> OnlinePreferencePair:
    chosen_trace = CandidateTrace(**dict(item["chosen_trace"]))
    rejected_trace = CandidateTrace(**dict(item["rejected_trace"]))
    return OnlinePreferencePair(
        session_id=str(item["session_id"]),
        turn_idx=int(item["turn_idx"]),
        assistant_turn=int(item["assistant_turn"]),
        history_lines=list(item["history_lines"]),
        bdi_text=str(item["bdi_text"]),
        chosen=str(item["chosen"]),
        rejected=str(item["rejected"]),
        chosen_trace=chosen_trace,
        rejected_trace=rejected_trace,
        ref_chosen_logp=item.get("ref_chosen_logp"),
        ref_rejected_logp=item.get("ref_rejected_logp"),
    )


def _load_pairs(paths: Sequence[str]) -> List[OnlinePreferencePair]:
    pairs: List[OnlinePreferencePair] = []
    seen = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a list of pairs")
        for item in data:
            pair = _pair_from_dict(item)
            key = (pair.session_id, pair.turn_idx, pair.chosen, pair.rejected)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(pair)
    if not pairs:
        raise RuntimeError("no DPO pairs loaded from --input_pairs")
    return pairs


def _make_llm(args, name: str) -> LLMClient:
    return LLMClient(LLMConfig(
        backend=args.judge_backend,
        model=args.judge_model,
        api_base=args.judge_api_base,
        api_key_env=args.judge_api_key_env,
        parallel_workers=args.judge_parallel_workers,
        verbose=bool(args.llm_verbose),
        name=name,
    ))


def _load_policy(
    model_path: str,
    adapter_path: str,
    device: str,
    args,
    *,
    max_new_tokens: int,
) -> LoRAPolicy:
    lora_kwargs = infer_lora_config_from_adapter(adapter_path)
    cfg = PolicyConfig(
        model_name_or_path=model_path,
        device=device,
        dtype=args.dtype,
        max_prompt_len=args.max_prompt_len,
        max_new_tokens=max_new_tokens,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        generation_use_cache=False,
        attn_implementation=args.attn_implementation,
        **lora_kwargs,
    )
    policy = LoRAPolicy(cfg)
    policy.load_adapter(adapter_path)
    return policy


def _system_messages(ctx: OnlineContext | OnlinePreferencePair, task: str):
    return build_chat_prompt_for_policy(
        role="assistant",
        history_lines=ctx.history_lines,
        task_name=task,
    )


def _user_messages(ctx: OnlineContext, candidate: str, task: str):
    return build_chat_prompt_for_policy(
        role="user",
        history_lines=list(ctx.history_lines) + [f"Assistant: {candidate}"],
        bdi_text=ctx.bdi_text,
        task_name=task,
    )


def _last_user_turn(history_lines: Sequence[str]) -> str:
    for line in reversed(history_lines):
        if str(line).startswith("User:"):
            return str(line)[5:].strip()
    return ""


def _template_candidates(ctx: OnlineContext, max_templates: int) -> List[str]:
    if int(max_templates) <= 0:
        return []
    last_user = _last_user_turn(ctx.history_lines)
    candidates: List[str] = []
    if _LOCAL_OBJECTION_RE.search(last_user):
        candidates.append(
            "I understand local charities matter too. Save the Children helps "
            "children in the U.S. and worldwide; would you be willing to donate "
            "a token $0.25 from your task payment now?"
        )
    elif _MONEY_OBJECTION_RE.search(last_user):
        candidates.append(
            "No pressure; even $0.25 is a valid small non-zero option. Would "
            "you be willing to donate $0.25 from your task payment to Save the "
            "Children now?"
        )
    elif _LATER_OBJECTION_RE.search(last_user):
        candidates.append(
            "I understand. Since this task can record a donation now, would you "
            "be willing to choose a token $0.25 donation to Save the Children "
            "from your task payment today?"
        )
    elif _USER_QUESTION_RE.search(last_user):
        candidates.append(
            "Save the Children supports children through health, education, "
            "protection, and emergency relief programs. Would you be willing "
            "to donate $0.25 or another small non-zero amount from your task "
            "payment now?"
        )

    candidates.extend([
        (
            "Save the Children supports children through health, education, "
            "protection, and emergency relief. Would you be willing to donate "
            "$0.25 from your task payment to Save the Children now?"
        ),
        (
            "Even a small non-zero amount such as $0.25 can help. What exact "
            "non-zero amount would you like to donate from your task payment "
            "to Save the Children now?"
        ),
        (
            "Would you be willing to donate $0.25 or another small non-zero "
            "amount from your task payment to Save the Children now?"
        ),
    ])
    deduped: List[str] = []
    seen = set()
    for text in candidates:
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
        if len(deduped) >= int(max_templates):
            break
    return deduped


def _is_strict_safe_chosen(text: str, history_lines: Sequence[str]) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if len(s.split()) > 70:
        return False
    if _UNSAFE_DPO_CHOSEN_RE.search(s):
        return False
    if _BAD_CANDIDATE_RE.search(s):
        return False
    if _GOODBYE_CANDIDATE_RE.search(s) and not _DONATION_ASK_RE.search(s):
        return False
    if not _DONATION_ASK_RE.search(s):
        return False
    if not _SAFE_CLOSE_CUE_RE.search(s):
        return False
    if not (_SAVE_CHILDREN_TEXT_RE.search(s) or _history_mentions_save_children(history_lines)):
        return False
    return True


def _candidate_penalty(text: str) -> float:
    penalty = 0.0
    if _BAD_CANDIDATE_RE.search(text or ""):
        penalty += 2.0
    if _UNSAFE_DPO_CHOSEN_RE.search(text or ""):
        penalty += 4.0
    if _GOODBYE_CANDIDATE_RE.search(text or "") and not _DONATION_ASK_RE.search(text or ""):
        penalty += 1.0
    if not _SAFE_CLOSE_CUE_RE.search(text or ""):
        penalty += 0.25
    return penalty


def _trace_score(trace: CandidateTrace, assistant_turn: int) -> float:
    ask_bonus = 0.10 if _DONATION_ASK_RE.search(trace.text or "") else 0.0
    early_bonus = max(0.0, (8.0 - float(assistant_turn) + 1.0) / 8.0)
    return (
        (5.0 if trace.success else 0.0)
        + float(trace.reward)
        + 0.25 * float(trace.rationality)
        + ask_bonus
        + (0.30 * early_bonus if trace.success else 0.0)
        - _candidate_penalty(trace.text)
    )


@torch.no_grad()
def collect_pairs(
    pi_S: LoRAPolicy,
    pi_U: LoRAPolicy,
    contexts: List[OnlineContext],
    args,
) -> List[OnlinePreferencePair]:
    pi_S.eval_mode()
    pi_U.eval_mode()
    judge_llm = _make_llm(args, "judge")
    p4g_judge = build_success_judge(
        task_name=args.task,
        llm=judge_llm,
        success_threshold=args.success_threshold,
        num_samples=args.judge_num_samples,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
    )
    rationality_judge = RationalityJudge(
        llm=judge_llm,
        task_description=TASK_CONFIGS[args.task].task_description,
    )

    pairs: List[OnlinePreferencePair] = []
    pbar = tqdm(contexts, desc="[online-dpo] collect")
    for ctx in pbar:
        sys_batch = [_system_messages(ctx, args.task) for _ in range(int(args.candidates_per_context))]
        cand_outs = pi_S.generate_batch(
            sys_batch,
            max_new_tokens=args.max_new_tokens_system,
            temperature=args.temperature_system,
            top_p=args.top_p,
            do_sample=True,
        )
        seen = set()
        candidates: List[str] = []
        for out in cand_outs:
            text = str(out.get("text", "")).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            candidates.append(text)
        for text in _template_candidates(ctx, int(args.template_candidates)):
            text = str(text).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            candidates.append(text)
        if len(candidates) < 2:
            continue

        traces: List[CandidateTrace] = []
        user_batch = [_user_messages(ctx, text, args.task) for text in candidates]
        user_outs = pi_U.generate_batch(
            user_batch,
            max_new_tokens=args.max_new_tokens_user,
            temperature=args.temperature_user,
            top_p=args.top_p,
            do_sample=True,
        )
        for text, uout in zip(candidates, user_outs):
            user_reply = str(uout.get("text", "")).strip()
            if not user_reply:
                continue
            sig = p4g_judge.score(ctx.history_lines, text, user_reply)
            rat = rationality_judge(
                assistant_turn=text,
                user_turn=user_reply,
                user_bdi_text=ctx.bdi_text,
            )
            trace = CandidateTrace(
                text=text,
                user_reply=user_reply,
                reward=float(sig.reward),
                success=bool(sig.success),
                label=str(sig.label),
                rationality=int(rat),
                score=0.0,
                raw_judgments=list(sig.raw_judgments),
            )
            trace.score = _trace_score(trace, ctx.assistant_turn)
            traces.append(trace)
        if len(traces) < 2:
            continue

        traces.sort(key=lambda x: x.score, reverse=True)
        chosen = traces[0]
        rejected = traces[-1]
        if chosen.text == rejected.text:
            continue
        if bool(args.require_chosen_ask) and not _DONATION_ASK_RE.search(chosen.text or ""):
            continue
        if bool(args.strict_safe_chosen) and not _is_strict_safe_chosen(
            chosen.text,
            ctx.history_lines,
        ):
            continue
        if bool(args.require_save_children_context) and not (
            _SAVE_CHILDREN_TEXT_RE.search(chosen.text or "")
            or _history_mentions_save_children(ctx.history_lines)
        ):
            continue
        if chosen.score - rejected.score < float(args.min_score_margin):
            continue
        if bool(args.require_success_contrast) and not (chosen.success and not rejected.success):
            continue
        if chosen.rationality < int(args.min_chosen_rationality):
            continue
        pairs.append(OnlinePreferencePair(
            session_id=ctx.session_id,
            turn_idx=ctx.turn_idx,
            assistant_turn=ctx.assistant_turn,
            history_lines=list(ctx.history_lines),
            bdi_text=ctx.bdi_text,
            chosen=chosen.text,
            rejected=rejected.text,
            chosen_trace=chosen,
            rejected_trace=rejected,
        ))
        pbar.set_postfix({"pairs": len(pairs)})
        if int(args.max_pairs) > 0 and len(pairs) >= int(args.max_pairs):
            break
    if not pairs:
        raise RuntimeError("no online-feedback DPO pairs after collection")
    return pairs


def _tokenize_target(tokenizer, text: str, max_target_len: int) -> torch.Tensor:
    ids = tokenizer(
        text or "",
        truncation=True,
        max_length=max(max_target_len - 1, 1),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    if tokenizer.eos_token_id is not None:
        eos = int(tokenizer.eos_token_id)
        if ids.numel() == 0 or int(ids[-1].item()) != eos:
            ids = torch.cat([ids, torch.tensor([eos], dtype=ids.dtype)], dim=0)
    return ids.long()


def _mean_logp(
    policy: LoRAPolicy,
    pair: OnlinePreferencePair,
    text: str,
    max_target_len: int,
    task: str,
) -> torch.Tensor:
    ids = _tokenize_target(policy.tokenizer, text, max_target_len=max_target_len)
    logp, mask = policy.log_probs_of_response(_system_messages(pair, task), ids)
    valid = mask.float()
    return (logp * valid).sum() / valid.sum().clamp(min=1.0)


@torch.no_grad()
def precompute_ref_logps(policy: LoRAPolicy, pairs: List[OnlinePreferencePair], args) -> None:
    policy.eval_mode()
    for pair in tqdm(pairs, desc="[online-dpo] ref logp"):
        pair.ref_chosen_logp = float(_mean_logp(
            policy, pair, pair.chosen, args.max_target_len, args.task,
        ).detach().item())
        pair.ref_rejected_logp = float(_mean_logp(
            policy, pair, pair.rejected, args.max_target_len, args.task,
        ).detach().item())


def train(args) -> Dict:
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    contexts: List[OnlineContext] = []
    pairs: List[OnlinePreferencePair]

    print(f"[online-dpo] loading pi_S={args.init_adapter}")
    pi_S = _load_policy(
        args.model_path, args.init_adapter, args.pi_S_device, args,
        max_new_tokens=args.max_new_tokens_system,
    )
    if args.input_pairs:
        pairs = _load_pairs(args.input_pairs)
        print(f"[online-dpo] loaded pairs={len(pairs)} from input files")
    else:
        contexts = _load_contexts(args)
        dump_json([asdict(c) for c in contexts], os.path.join(args.out_dir, "online_contexts.json"))
        print(
            f"[online-dpo] loading pi_U={args.pi_U_adapter} "
            f"contexts={len(contexts)} offset={args.context_offset}/{args.context_stride}"
        )
        pi_U = _load_policy(
            args.model_path, args.pi_U_adapter, args.pi_U_device, args,
            max_new_tokens=args.max_new_tokens_user,
        )
        pairs = collect_pairs(pi_S, pi_U, contexts, args)
        dump_json([asdict(p) for p in pairs], os.path.join(args.out_dir, "dpo_pairs_unscored.json"))
        print(f"[online-dpo] kept pairs={len(pairs)}")

    if bool(args.build_only):
        summary = {
            "task": args.task,
            "init_adapter": args.init_adapter,
            "pi_U_adapter": args.pi_U_adapter,
            "contexts": len(contexts),
            "pairs": len(pairs),
            "build_only": True,
            "args": vars(args),
        }
        dump_json(summary, os.path.join(args.out_dir, "online_dpo_log.json"))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    precompute_ref_logps(pi_S, pairs, args)
    dump_json([asdict(p) for p in pairs], os.path.join(args.out_dir, "dpo_pairs.json"))

    pi_S.train_mode()
    opt = torch.optim.AdamW(
        pi_S.trainable_parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
    )
    rng = random.Random(args.seed)
    beta = float(args.beta)
    batch_size = max(int(args.batch_size), 1)
    losses: List[float] = []
    accs: List[float] = []
    global_step = 0
    t0 = time.perf_counter()

    for epoch in range(int(args.num_epochs)):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        pbar = tqdm(
            range(0, len(order), batch_size),
            desc=f"[online-dpo] ep {epoch + 1}/{args.num_epochs}",
        )
        for start in pbar:
            idxs = order[start: start + batch_size]
            opt.zero_grad(set_to_none=True)
            batch_losses: List[torch.Tensor] = []
            batch_acc = 0.0
            for idx in idxs:
                pair = pairs[idx]
                pi_c = _mean_logp(pi_S, pair, pair.chosen, args.max_target_len, args.task)
                pi_r = _mean_logp(pi_S, pair, pair.rejected, args.max_target_len, args.task)
                ref_c = torch.tensor(float(pair.ref_chosen_logp), device=pi_c.device)
                ref_r = torch.tensor(float(pair.ref_rejected_logp), device=pi_c.device)
                logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))
                batch_losses.append(-F.logsigmoid(logits))
                batch_acc += float((pi_c > pi_r).detach().item())
            if not batch_losses:
                continue
            loss = torch.stack(batch_losses).mean()
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pi_S.trainable_parameters(), float(args.max_grad_norm))
            opt.step()
            global_step += 1
            losses.append(float(loss.detach().item()))
            accs.append(batch_acc / float(len(idxs)))
            if global_step % max(int(args.log_every), 1) == 0:
                pbar.set_postfix({
                    "loss": f"{losses[-1]:.4f}",
                    "acc": f"{sum(accs[-20:]) / min(len(accs), 20):.3f}",
                })

    save_dir = os.path.join(args.out_dir, "pi_S")
    ensure_dir(save_dir)
    pi_S.save_adapter(save_dir)
    summary = {
        "task": args.task,
        "init_adapter": args.init_adapter,
        "pi_U_adapter": args.pi_U_adapter,
        "save_dir": save_dir,
        "contexts": len(contexts),
        "pairs": len(pairs),
        "epochs": int(args.num_epochs),
        "global_steps": int(global_step),
        "final_loss": float(losses[-1]) if losses else None,
        "mean_loss": float(sum(losses) / len(losses)) if losses else None,
        "mean_pref_acc": float(sum(accs) / len(accs)) if accs else None,
        "elapsed_sec": float(time.perf_counter() - t0),
        "args": vars(args),
    }
    dump_json(summary, os.path.join(args.out_dir, "online_dpo_log.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_cache", default="data_cache/p4g_bdi_train.json")
    p.add_argument("--model_path", default="/path/to/base-model")
    p.add_argument("--init_adapter", default="checkpoints/phase2_closure/best/pi_S")
    p.add_argument("--pi_U_adapter", default="checkpoints/phase2_closure/best/pi_U")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--task", default="p4g", choices=list(TASK_CONFIGS.keys()))
    p.add_argument("--pi_S_device", default="cuda:0")
    p.add_argument("--pi_U_device", default="cuda:0")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--max_prompt_len", type=int, default=2048)
    p.add_argument("--max_target_len", type=int, default=128)
    p.add_argument("--max_new_tokens_system", type=int, default=96)
    p.add_argument("--max_new_tokens_user", type=int, default=96)
    p.add_argument("--min_assistant_turn", type=int, default=3)
    p.add_argument("--max_assistant_turn", type=int, default=5)
    p.add_argument("--max_history_lines", type=int, default=30)
    p.add_argument("--skip_recent_refusal", type=int, default=1)
    p.add_argument("--skip_user_committed", type=int, default=1)
    p.add_argument("--require_save_children_context", type=int, default=1)
    p.add_argument("--max_contexts", type=int, default=0)
    p.add_argument("--context_offset", type=int, default=0)
    p.add_argument("--context_stride", type=int, default=1)
    p.add_argument("--input_pairs", action="append", default=[],
                   help="Load pre-collected pair JSON files and train from them "
                        "instead of collecting online feedback in this run.")
    p.add_argument("--max_pairs", type=int, default=0)
    p.add_argument("--candidates_per_context", type=int, default=4)
    p.add_argument("--template_candidates", type=int, default=0,
                   help="Add this many bounded safe close templates to each "
                        "online-feedback context before rollout scoring.")
    p.add_argument("--min_score_margin", type=float, default=1.0)
    p.add_argument("--require_success_contrast", type=int, default=1)
    p.add_argument("--require_chosen_ask", type=int, default=1)
    p.add_argument("--strict_safe_chosen", type=int, default=0,
                   help="If 1, only keep chosen responses that pass strict "
                        "P4G safety/text-quality checks.")
    p.add_argument("--min_chosen_rationality", type=int, default=0)
    p.add_argument("--temperature_system", type=float, default=0.9)
    p.add_argument("--temperature_user", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--judge_backend", default="openai",
                   choices=["openai", "azure", "local", "heuristic"])
    p.add_argument("--judge_model", default="gpt-3.5-turbo")
    p.add_argument("--judge_api_base", default="")
    p.add_argument("--judge_api_key_env", default="")
    p.add_argument("--judge_num_samples", type=int, default=3)
    p.add_argument("--judge_temperature", type=float, default=1.0)
    p.add_argument("--judge_max_tokens", type=int, default=16)
    p.add_argument("--judge_parallel_workers", type=int, default=1)
    p.add_argument("--success_threshold", type=float, default=0.6)
    p.add_argument("--llm_verbose", type=int, default=0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--build_only", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
