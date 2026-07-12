#!/usr/bin/env python3
"""
Evaluate a trained system policy π_S against different user simulators.

We support TWO evaluation modes, each of which is a standard
DialogXpert-style SR / AT / AT_success scoring loop over a held-out split.
The only thing that varies between modes is *which* user model drives the
other side of the dialogue:

    --user_mode bc
        The Phase-1 behavior-cloned π_U (soft LLM simulator, no BDI anchor
        in the reward and no adversarial training). This is the "soft"
        environment that prior work like PPDPP / DialogXpert / UDP is
        implicitly evaluated in.

    --user_mode adversarial
        The Phase-2 adversarially trained π_U loaded from
        `out_dir/best/pi_U` of the self-play run. This is the "hard"
        environment — the committed BDI anchor plus the -α_shape reward
        make this user resist superficial persuasion attempts.

We report for each mode:
    SR          — success rate (DialogXpert judge > 0.6)
    AT          — average number of turns
    AT_success  — avg turns on successful dialogues only
    mean_reward — per-episode cumulative system reward
    avg_progress_final — mean of prog(z*_final), in BDI space
    avg_rationality    — mean rationality judge signal over all turns

The headline experiment (the "adversarial user robustness test") is run by
comparing SR under `--user_mode bc` vs `--user_mode adversarial`. Prior
work that trained only against a BC-frozen LLM simulator is predicted to
degrade substantially under the adversarial user, whereas a system policy
trained end-to-end with MASP should degrade much less.

Usage
-----
    python evaluate_masp.py \
        --test_path dataset/p4g/test.json \
        --test_cache data_cache/p4g_bdi_test.json \
        --model_path /path/to/base-model \
        --pi_S_adapter checkpoints/phase2/best/pi_S \
        --pi_U_adapter_bc checkpoints/phase1/pi_U \
        --pi_U_adapter_adv checkpoints/phase2/best/pi_U \
        --encoder_model /path/to/base-model \
        --mentalization_ckpt checkpoints/phase2/best/mentalization.pt \
        --user_mode adversarial \
        --out_path logs/eval_adversarial.json

In a typical experiment you run it twice (once per `--user_mode`) and
compare the two numbers in the final paper table.
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from masp.data.bdi_dataset import BDILabelCache
from masp.data.dataset_adapter import get_adapter
from masp.env.dialogue_env import POBGDialogueEnv
from masp.eval.metrics import DialogMetrics
from masp.eval.robustness_metrics import (
    RunMeta,
    compute_metrics_for_method,
    normalize_episodes,
)
from masp.eval.success_judge import build_success_judge
from masp.mind.bdi_extractor import BDIExtractor
from masp.mind.bdi_schema import TASK_CONFIGS, encode_bdi
from masp.mind.mind_prior import MindPrior, MindPriorEntry
from masp.models.mentalization import (
    MentalizationConfig,
    MentalizationModule,
    TeacherMentalizationModule,
)
from masp.models.policy import (
    LoRAPolicy,
    PolicyConfig,
    build_chat_prompt_for_policy,
    infer_lora_config_from_adapter,
)
from masp.models.sentence_encoder import SentenceEncoder, SentenceEncoderConfig
from masp.rl.rollout import _postprocess_system_turn
from masp.rl.rewards import RationalityJudge, RewardConfig, progress_score
from masp.utils.io import dump_json, ensure_dir
from masp.utils.llm_client import LLMClient, LLMConfig
from masp.utils.seed import set_seed


# ------------------------------------------------------------------ helpers

_TEXT_SELECTOR_MODEL = None
_TEXT_SELECTOR_METHOD = ""
_STOP_ROUTER_MODEL = None
_STOP_ROUTER_USER_TYPE = ""


def _load_text_selector_model(path: str):
    if not path:
        return None
    try:
        import joblib
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("system_text_selector_model requires joblib") from exc
    return joblib.load(path)


def _load_stop_router_model(path: str):
    if not path:
        return None
    try:
        import joblib
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("system_stop_router_model requires joblib") from exc
    return joblib.load(path)

def _build_mind_prior_from_cache(cache: BDILabelCache) -> MindPrior:
    entries: List[MindPriorEntry] = []
    for sid, bdi in cache.initial_bdi.items():
        entries.append(MindPriorEntry(
            session_id=sid,
            bdi=bdi,
            profile_text=cache.profile_text.get(sid, ""),
        ))
    return MindPrior(entries=entries)


def _make_llm(args, role: str) -> LLMClient:
    if role == "judge":
        backend = args.judge_backend
        model = args.judge_model
        api_base = args.judge_api_base
        api_key_env = args.judge_api_key_env
        workers = args.judge_parallel_workers
    elif role == "obu":
        backend = args.obu_backend
        model = args.obu_model
        api_base = args.obu_api_base
        api_key_env = args.obu_api_key_env
        workers = args.obu_parallel_workers
    else:
        raise ValueError(f"unknown llm role: {role}")
    return LLMClient(LLMConfig(
        backend=backend,
        model=model,
        api_base=api_base,
        api_key_env=api_key_env,
        azure_endpoint=args.azure_endpoint,
        azure_api_version=args.azure_api_version,
        azure_thinking_budget=args.azure_thinking_budget,
        parallel_workers=workers,
        verbose=bool(args.llm_verbose),
        name=role,
    ))


def _system_strategy_hint(mode: str) -> str:
    mode = (mode or "default").strip().lower()
    if mode == "default":
        return ""
    if mode == "early_close":
        return (
            "Fast P4G strategy: by your first or second turn, mention Save the "
            "Children and give exactly one concrete reason, then ask whether "
            "the user will donate any non-zero amount from their task payment. "
            "If the user asks a question, answer it in one short sentence and "
            "ask for a clear donation decision in the same turn. By assistant "
            "turn 3, ask for a specific amount or a yes/no donation decision."
        )
    if mode == "direct_ask":
        return (
            "Direct P4G strategy: on every turn until the user clearly accepts "
            "or refuses, include a concise donation ask. Prefer asking whether "
            "they will donate $1 or any other non-zero amount from the task "
            "payment today. Keep the reply to no more than two sentences."
        )
    if mode == "brief_lowpressure":
        return (
            "Brief low-pressure P4G strategy: keep each reply to one or two "
            "sentences. First answer the user's latest concern directly. If "
            "the user has not clearly refused twice, ask once for a small "
            "non-zero donation such as $0.25 from the task payment to Save "
            "the Children. Never ask for $1 or more, the full task payment, "
            "monthly donations, or matching donations. If the user clearly "
            "refuses twice, stop asking and thank them."
        )
    raise ValueError(f"unknown --system_strategy: {mode}")


_DONATION_ASK_RE = re.compile(
    r"("
    r"\b(?:would|will|could|can)\s+you\b.{0,120}\b(?:donat|contribut|help out)|"
    r"\byou'?d\s+like\b.{0,120}\b(?:donat|contribut|help out)|"
    r"\b(?:like|want)\s+to\b.{0,80}\b(?:donat|contribut)|"
    r"\b(?:willing|able)\b.{0,80}\b(?:donat|contribut)|"
    r"\bhow much\b.{0,80}\bdonat|"
    r"\bwhat amount\b.{0,80}\bdonat|"
    r"\bdonate\b.{0,80}\b(?:small|non-zero|any amount|\$|cent|task payment)"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_CONCRETE_SMALL_ASK_RE = re.compile(
    r"(\$0?\.(?:0[1-9]|[1-9]\d?)|\b(?:[1-9]|[1-9]\d)\s+cents?\b|"
    r"25 cents?|a few cents?|small non-zero|any non-zero|"
    r"small amount|even a little|from your task payment|part of your task payment)",
    re.IGNORECASE,
)
_SAVE_CHILDREN_RE = re.compile(r"\bsave the children\b", re.IGNORECASE)
_P4G_FACT_RE = re.compile(
    r"\b(children|education|health|hunger|food|shelter|emergency|"
    r"disaster|war|poverty|rights|relief|developing countries)\b",
    re.IGNORECASE,
)
_USER_QUESTION_RE = re.compile(
    r"\?|"
    r"\b(what|why|how|who|where|which|programs?|country|countries|"
    r"website|trust|cons?|provide|support|donated|donate today)\b",
    re.IGNORECASE,
)
_TRUST_QUESTION_RE = re.compile(
    r"\b("
    r"trust|legit(?:imate)?|scam|reputable|rated|rating|charity navigator|"
    r"website|proof|evidence|where (?:does|will) (?:the )?money go|"
    r"how (?:do|can) i know|how much (?:of|goes)|"
    r"administrative costs?|overhead|tax deductible"
    r")\b",
    re.IGNORECASE,
)
_BAD_CANDIDATE_RE = re.compile(
    r"\b("
    r"put you down|mark you down|sign you up|count you in|"
    r"take (?:your )?refusal as (?:a )?yes|"
    r"need to get this done|complete this task|need .*turns?|"
    r"don'?t you want to|poor needy|no con to donating|"
    r"all (?:of )?the money goes|guarantee|tax deductible"
    r")\b|"
    r"\bURL\b|https?://\S*URL\S*",
    re.IGNORECASE,
)
_GUARDED_BAD_CANDIDATE_RE = re.compile(
    r"\b("
    r"must make a donation|have to .*donat|"
    r"donation .*processed|won'?t get the full|"
    r"we have to talk|chat to register|submit this hit|"
    r"all (?:of )?(?:the )?(?:money|funds?) (?:goes?|go)|"
    r"100\s*%\s+(?:goes?|go)|"
    r"we need people like you|without support|"
    r"it is our duty|must use this hit wisely"
    r")\b",
    re.IGNORECASE,
)
_GENERIC_BANTER_RE = re.compile(
    r"\b(how are you|have you ever donated|may i tell you|"
    r"would you like to know|do you have any children|"
    r"do you ever get tired)\b",
    re.IGNORECASE,
)
_GOODBYE_CANDIDATE_RE = re.compile(
    r"\b(goodbye|bye|have a (?:great|nice|wonderful)|"
    r"thank you for your time|thanks for listening)\b",
    re.IGNORECASE,
)
_REFUSAL_RE = re.compile(
    r"\b(no thank|not donate|don'?t want to donate|do not want to donate|"
    r"would not donate|will not donate|wouldn'?t like to donate|"
    r"would not like to donate|not like to donate|do not wish to donate|"
    r"don'?t plan to donate|do not plan to donate|donate zero|"
    r"choose 0|i choose 0|\$0|0 cents?|0 dollars|nothing|"
    r"nope|nah|not today|not now|maybe later|another time|"
    r"in the future|at a later time|right now i don'?t|i'?m sure|im sure|"
    r"not ready|i am not ready|already donate|numerous donations|"
    r"prefer (?:a )?local|local charities|prefer cash|need the money|"
    r"really need the money|can'?t help|cannot help|"
    r"not at this time|can'?t donate|cannot donate|not interested in donating|"
    r"still not interested|not comfortable donating|don'?t feel comfortable donating|"
    r"do not feel comfortable donating|no i do not|no i would not)\b",
    re.IGNORECASE,
)
_SOFT_INTEREST_RE = re.compile(
    r"\b("
    r"sounds? (?:good|great|excellent|nice|worthwhile|important)|"
    r"great cause|good cause|excellent cause|worthwhile endeavor|"
    r"happy to (?:hear|donate)|willing to donate|i can donate|"
    r"i would donate|i'?ll donate|i will donate|interested|"
    r"look into it|what does|where does|how does|who does|"
    r"tell me more|more information"
    r")\b",
    re.IGNORECASE,
)
_WARMUP_POSITIVE_RE = re.compile(
    r"\b("
    r"sounds? (?:good|great|excellent|nice|worthwhile|important|interesting)|"
    r"that sounds (?:good|great|excellent|nice|worthwhile|important|interesting)|"
    r"(?:very|really|extremely)?\s*(?:good|great|awesome|admirable|impressive|important|interesting|nice|worthwhile)\s+"
    r"(?:cause|charity|organization|work|thing|goal|endeavor)|"
    r"awesome|admirable|impressive|important cause|"
    r"great cause|good cause|excellent cause|worthwhile endeavor|"
    r"good work|great work|worthy cause|worthwhile cause|"
    r"i(?:'| a)?m glad|glad to hear|happy to hear"
    r")\b",
    re.IGNORECASE,
)
_LOCAL_OBJECTION_RE = re.compile(r"\b(local|nearby|community|neighborhood)\b", re.IGNORECASE)
_MONEY_OBJECTION_RE = re.compile(
    r"\b(need the money|prefer cash|can'?t afford|cannot afford|"
    r"money too|short on money|tight on money|task payment)\b",
    re.IGNORECASE,
)
_LATER_OBJECTION_RE = re.compile(
    r"\b(later|future|another time|not today|not now|couple months|"
    r"look into it|will take a look)\b",
    re.IGNORECASE,
)
_REPAIR_BAD_CANDIDATE_RE = re.compile(
    r"\b(deadline|waste my time|supposed to close|close out of this chat|"
    r"end of the hit|how do i end the task|full \$2|entire \$2|"
    r"choose \$0|donate \$0|cildren|childen|chidren|"
    r"can we make it|\bi just need\b|not be charged)\b",
    re.IGNORECASE,
)
_UNSAFE_CLOSE_CANDIDATE_RE = re.compile(
    r"("
    r"\$\s*(?:[1-9]|\d{2,})(?:\b|\s|[.,])|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+dollars?\b|"
    r"\b(?:month|monthly|per\s+month)\b|"
    r"\bmatch\s+(?:your|whatever)|\bi'?ll\s+match\b|"
    r"\bpoor\s+needy\b|\bneedy\s+(?:children|kids)\b|"
    r"\bguilt(?:y)?\b|\bmoral\s+responsibility\b|"
    r"\bdeadline\b|\bwaste\s+my\s+time\b|"
    r"\bpaymen\b|\bchildens\b|\bchilden\b|\bcildren\b|\bchidren\b|"
    r"\b(?:from|between)\s+\$?0(?:\.00)?\s*(?:-|to|and|up\s+to)\b|"
    r"\b\$?0(?:\.00)?\s*(?:-|to|and)\s+\$?2\b|"
    r"\b(?:up\s+to|all\s+the\s+way\s+to)\s+(?:all\s+of\s+)?(?:your\s+)?(?:payment|task\s+payment|\$2)\b|"
    r"\ball\s+(?:of\s+)?(?:your\s+)?(?:payment|task\s+payment|bonus|it)\b|"
    r"\b(?:full|entire|whole)\s+(?:of\s+)?(?:your\s+)?(?:payment|task\s+payment|bonus|\$2)\b|"
    r"\bamount\s+equal\s+to\s+your\s+task\s+payment\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_OPEN_AMOUNT_CLOSE_RE = re.compile(
    r"("
    r"\bhow\s+much\s+(?:do|would|will|can)\s+you\s+(?:like|want|wish)?\s*(?:to\s+)?(?:donate|contribute|give)\b|"
    r"\bhow\s+much\s+of\s+your\s+(?:task\s+payment|bonus|earnings|payment)\b|"
    r"\bwhat\s+(?:exact\s+)?amount\s+(?:are|would|will|can)\s+you\s+(?:willing|like|want|able)?\s*(?:to\s+)?(?:donate|contribute|give)?\b|"
    r"\byou\s+can\s+choose\s+any\s+amount\b|"
    r"\bchoose\s+(?:any|whatever)\s+amount\b|"
    r"\b(?:any|whatever)\s+(?:amount|donation)\s+(?:you|would|will|can)\b|"
    r"\bamount\s+of\s+your\s+choice\b|"
    r"\b(?:from|between)\s+\$?0(?:\.00)?\s*(?:-|to|and|up\s+to)\b|"
    r"\b\$?0(?:\.00)?\s*(?:-|to|and)\s+\$?2\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_ZERO_AMOUNT_CLOSE_RE = re.compile(
    r"("
    r"\bdonat(?:e|ing)?\s+\$?0(?:\.00)?\b|"
    r"\b\$?0(?:\.00)?\s+(?:to|for)\s+save the children\b|"
    r"\bchoose\s+\$?0(?:\.00)?\b|"
    r"\bput you down for\s+\$?0(?:\.00)?\b"
    r")",
    re.IGNORECASE,
)
_PERSONAL_OFFTASK_RE = re.compile(
    r"\b("
    r"do you have children|do you have kids|how much money do you have|"
    r"money in your pocket|share my screen|have you had a good day|"
    r"what charities do you donate|have you ever donated|"
    r"are you sure you do(?:n't| not) have kids|can i ask why"
    r")\b",
    re.IGNORECASE,
)
_STRONG_DONATION_CUE_RE = re.compile(
    r"\b(i can donate|i could donate|i would donate|i will donate|"
    r"i'?ll donate|we could do|could do|can do|maybe give|"
    r"very small amount|small amount|portion of (?:the )?(?:bonus|money)|"
    r"what amount|how much|willing to donate|would like to donate)\b",
    re.IGNORECASE,
)
_USER_AMOUNT_COMMIT_RE = re.compile(
    r"\b("
    r"i (?:can|could|would|will|'ll|want to|would like to) (?:donate|give|do)|"
    r"i'?d (?:donate|give|do)|"
    r"we (?:can|could|would) do|"
    r"maybe (?:i )?(?:can|could|would )?(?:donate|give|do)|"
    r"i think maybe we could do|"
    r"what about (?:i|we) do|"
    r"i am willing to"
    r")\b",
    re.IGNORECASE,
)
_USER_AMOUNT_QUESTION_RE = re.compile(
    r"\?|"
    r"\b(could i|can i|would it|is it ok|is that|what about|portion|total)\b",
    re.IGNORECASE,
)
_INFO_NEED_RE = re.compile(
    r"\b("
    r"need (?:more )?(?:info|information|time|details)|"
    r"more (?:info|information|details)|"
    r"want (?:to )?(?:see|look|research|know)|"
    r"research|look into|not familiar|never heard|"
    r"why (?:did|do) you (?:pick|choose|suggest)|"
    r"why (?:this|that) amount|curious why"
    r")\b",
    re.IGNORECASE,
)
_SUBSTANTIVE_P4G_QUESTION_RE = re.compile(
    r"\b("
    r"need (?:more )?(?:info|information|details)|"
    r"more (?:info|information|details)|"
    r"do you work for|work for them|support them|"
    r"have you donated|how much did you donate|"
    r"where .*money|money .*go|donations? .*go|used for|"
    r"make a difference|really save|really help|impact|results?|"
    r"trust|legit(?:imate)?|scam|proof|evidence|website|"
    r"why (?:did|do) you (?:pick|choose|suggest)|why (?:this|that) amount|"
    r"curious why|what is save the children|who is save the children|"
    r"what does save the children|programs?"
    r")\b",
    re.IGNORECASE,
)
_NONZERO_AMOUNT_RE = re.compile(
    r"(\$\s*(?:0?\.\d*[1-9]\d*|[1-9]\d*(?:\.\d+)?)|"
    r"\b(?:0?\.\d*[1-9]\d*|[1-9]\d*(?:\.\d+)?)\s*(?:cents?|dollars?)\b|"
    r"\b(?:0?\.\d*[1-9]\d*)\b)",
    re.IGNORECASE,
)
_WORD_CENT_AMOUNT_RE = re.compile(
    r"\b("
    r"one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|fifteen|twenty|twenty five|twenty-five|"
    r"a penny|penny|a nickel|nickel|a dime|dime|a quarter|quarter"
    r")\s+cents?\b|\b(a penny|penny|a nickel|nickel|a dime|dime|a quarter|quarter)\b",
    re.IGNORECASE,
)


def _last_user_turn(history_lines: List[str]) -> str:
    for line in reversed(history_lines):
        if str(line).startswith("User:"):
            return str(line)[5:].strip()
    return ""


def _last_assistant_turn(history_lines: List[str]) -> str:
    for line in reversed(history_lines):
        if str(line).startswith("Assistant:"):
            return str(line)[10:].strip()
    return ""


def _is_user_refusal_text(text: str) -> bool:
    """Return true for current-session refusal, including terse variants."""
    s = str(text or "").strip()
    if not s:
        return False
    if _REFUSAL_RE.search(s):
        return True
    sl = s.lower()
    if re.search(r"^\s*(?:no|nope|nah)\s*(?:[,.!]+\s*)?(?:thank|thanks)\b", sl):
        return True
    if re.search(r"^\s*(?:no|nope|nah)\s*(?:[,.!]+)?\s*$", sl):
        return True
    if re.search(r"^\s*no\b", sl) and not re.search(r"\b(no problem|no worries)\b", sl):
        return True
    if re.search(r"\bi\s+(?:still\s+)?(?:don'?t|do not)\s+think\b", sl):
        return True
    if re.search(r"\bi'?m\s+done\b|\bi am done\b|\bstick to my original decision\b", sl):
        return True
    if re.search(r"\b(?:still|again)\s+(?:say\s+)?no\b", sl):
        return True
    return False


def _recent_user_refusal_count(history_lines: List[str], max_lines: int = 8) -> int:
    count = 0
    for line in reversed(history_lines[-max_lines:]):
        if str(line).startswith("User:") and _is_user_refusal_text(str(line)[5:]):
            count += 1
    return count


def _session_user_refusal_count(history_lines: List[str]) -> int:
    count = 0
    for line in history_lines:
        if str(line).startswith("User:") and _is_user_refusal_text(str(line)[5:]):
            count += 1
    return count


def _bdi_scalar(bdi, name: str, default: float) -> float:
    try:
        return float(getattr(bdi, name))
    except Exception:
        return float(default)


def _bdi_text(bdi) -> str:
    if bdi is None:
        return ""
    try:
        return str(bdi.to_text())
    except Exception:
        return " ".join(
            str(getattr(bdi, k, ""))
            for k in ("belief", "desire", "intention")
        )


def _bdi_positive_donation_state(bdi) -> bool:
    text = _bdi_text(bdi).lower()
    rho = _bdi_scalar(bdi, "receptivity", 0.5)
    val = _bdi_scalar(bdi, "valence", 0.0)
    positive_text = bool(re.search(
        r"\b(open|consider|possibly|maybe|willing|interested|likely|"
        r"intends? to donate|donate now|make a donation|help children)\b",
        text,
    ))
    negative_text = bool(re.search(
        r"\b(not donate|won'?t donate|decline|refus|zero|not interested|"
        r"does not want to donate|intends? not to)\b",
        text,
    ))
    return (rho >= 0.58 or val >= 0.25 or positive_text) and not negative_text


def _bdi_hard_negative_state(bdi) -> bool:
    text = _bdi_text(bdi).lower()
    rho = _bdi_scalar(bdi, "receptivity", 0.5)
    conf = _bdi_scalar(bdi, "confidence", 0.5)
    negative_text = bool(re.search(
        r"\b(not donate|won'?t donate|decline|refus|zero|not interested|"
        r"cannot donate|unable to donate|stick to .*decision)\b",
        text,
    ))
    return (rho <= 0.25 and conf >= 0.65) or (negative_text and conf >= 0.65)


def _user_positive_acceptance_after_ask(history_lines: List[str]) -> bool:
    """Detect a narrow yes/acceptance reply to the immediately previous ask."""
    last_user = _last_user_turn(history_lines)
    if not last_user or _is_user_refusal_text(last_user):
        return False
    if _USER_QUESTION_RE.search(last_user) or _INFO_NEED_RE.search(last_user):
        return False
    prev_assistant = _last_assistant_turn(history_lines[:-1])
    if not _DONATION_ASK_RE.search(prev_assistant):
        return False
    user = last_user.lower()
    positive = bool(re.search(
        r"\b("
        r"yes|sure|okay|ok|alright|fine|sounds good|that works|"
        r"i can do that|i could do that|i will do that|i'?ll do that|"
        r"i have no problem doing that|no problem doing that|"
        r"i'?m willing|i am willing|go ahead|let'?s do it|"
        r"happy to|i can donate|i will donate|i'?ll donate"
        r")\b",
        user,
    ))
    if not positive:
        return False
    if re.search(r"\b(but|however|not from|not donate|can't|cannot|won'?t|wouldn'?t|zero|\$0)\b", user):
        return False
    return True


def _history_has_polite_stop_without_ask(history_lines: List[str]) -> bool:
    """Whether a previous assistant turn already closed without a donation ask."""
    for line in history_lines:
        if not str(line).startswith("Assistant:"):
            continue
        text = str(line)[10:].strip()
        if _DONATION_ASK_RE.search(text):
            continue
        if (
            _GOODBYE_CANDIDATE_RE.search(text)
            or re.search(
                r"\b(thank you for considering|thank you for your consideration|"
                r"thanks for considering|we appreciate it|"
                r"i understand[.!]?\s+thank you)\b",
                text,
                re.IGNORECASE,
            )
        ):
            return True
    return False


def _stop_router_feature_text(
    pre_history: List[str],
    feature_version: str,
    reward_bundles: Optional[List[Dict]] = None,
    bdi_text: str = "",
    user_type: str = "",
    include_user_type: bool = False,
) -> str:
    history = [str(x) for x in pre_history]
    text = "\n".join(history[-8:])
    if feature_version not in {"state_v2", "state_reward_v1", "state_bdi_v1"}:
        if include_user_type:
            return f"[USER_TYPE] {user_type}\n{text}"
        return text
    last_user = _last_user_turn(history)
    assistant_turn = sum(1 for line in history if str(line).startswith("Assistant:")) + 1
    recent_refusals = _recent_user_refusal_count(history)
    session_refusals = _session_user_refusal_count(history)
    markers = [
        f"[ASSISTANT_TURN] {min(assistant_turn, 8)}",
        f"[RECENT_REFUSALS] {min(recent_refusals, 3)}",
        f"[SESSION_REFUSALS] {min(session_refusals, 4)}",
        f"[LAST_USER_QUESTION] {int(bool(_USER_QUESTION_RE.search(last_user)))}",
        f"[LAST_USER_INTEREST] {int(bool(_SOFT_INTEREST_RE.search(last_user) or _STRONG_DONATION_CUE_RE.search(last_user)))}",
        f"[LAST_USER_AMOUNT] {int(bool(_extract_nonzero_amount(last_user)))}",
        f"[HISTORY_HAS_ASK] {int(any(_DONATION_ASK_RE.search(str(line)) for line in history if str(line).startswith('Assistant:')))}",
    ]
    if feature_version == "state_reward_v1":
        bundles = list(reward_bundles or [])
        labels = [str(b.get("task_label", "")).lower() for b in bundles if isinstance(b, dict)]
        last = bundles[-1] if bundles and isinstance(bundles[-1], dict) else {}
        try:
            last_progress = float(last.get("progress_delta", 0.0))
        except Exception:
            last_progress = 0.0
        try:
            last_close_quality = float(last.get("close_quality", 0.0))
        except Exception:
            last_close_quality = 0.0
        progress_bin = (
            "pos" if last_progress > 0.05 else
            "neg" if last_progress < -0.02 else
            "flat"
        )
        close_bin = (
            "high" if last_close_quality >= 0.7 else
            "mid" if last_close_quality > 0.0 else
            "none"
        )
        markers.extend([
            f"[LAST_TASK_LABEL] {str(last.get('task_label', 'none')).lower()}",
            f"[LAST_SUCCESS] {int(bool(last.get('success', False)))}",
            f"[LAST_RATIONALITY] {str(last.get('rationality', 0))}",
            f"[LAST_PROGRESS_BIN] {progress_bin}",
            f"[LAST_CLOSE_QUALITY] {close_bin}",
            f"[LABEL_POSITIVE_COUNT] {labels.count('positive')}",
            f"[LABEL_REFUSED_COUNT] {labels.count('refused')}",
            f"[LABEL_AGREE_COUNT] {labels.count('agree')}",
            f"[LABEL_NEUTRAL_COUNT] {labels.count('neutral')}",
        ])
    if feature_version == "state_bdi_v1":
        markers.append("[BDI_TEXT]")
        markers.append(str(bdi_text or ""))
    if include_user_type:
        markers.insert(0, f"[USER_TYPE] {user_type}")
    return "\n".join(markers + [text])


def _route_stop_after_from_text(
    pre_history: List[str],
    default: int,
    reward_bundles: Optional[List[Dict]] = None,
    current_bdi=None,
) -> int:
    model = _STOP_ROUTER_MODEL
    if model is None:
        return int(default)
    vectorizer = model.get("vectorizer")
    clf = model.get("clf")
    if vectorizer is None or clf is None:
        return int(default)
    sample = _stop_router_feature_text(
        pre_history,
        str(model.get("feature_version", "text_v1")),
        reward_bundles=reward_bundles,
        bdi_text=_bdi_text(current_bdi),
        user_type=_STOP_ROUTER_USER_TYPE,
        include_user_type=bool(model.get("include_user_type", False)),
    )
    try:
        x = vectorizer.transform([sample])
        probs = clf.predict_proba(x)[0]
        classes = list(clf.classes_)
        best_i = max(range(len(probs)), key=lambda i: float(probs[i]))
        confidence = float(probs[best_i])
        min_conf = float(model.get("min_confidence", 0.0))
        if confidence < min_conf:
            return int(default)
        label = str(classes[best_i])
    except Exception:
        return int(default)
    if label in {"stop2", "2"}:
        return 2
    if label in {"stop3", "3"}:
        return 3
    if label in {"stop4", "4"}:
        return 4
    return int(default)


def _consecutive_user_refusal_count(history_lines: List[str], max_lines: int = 8) -> int:
    count = 0
    for line in reversed(history_lines[-max_lines:]):
        if not str(line).startswith("User:"):
            continue
        if _is_user_refusal_text(str(line)[5:]):
            count += 1
            continue
        break
    return count


def _count_questions(text: str) -> int:
    return text.count("?") + len(re.findall(r"\b(would|will|could|can|what|why|how)\b", text, re.I))


def _extract_nonzero_amount(text: str) -> str:
    s = str(text or "")
    for match in _NONZERO_AMOUNT_RE.finditer(s):
        amount = match.group(1).strip()
        local = s[max(0, match.start() - 25): match.end() + 25].lower()
        if re.search(r"\b(?:zero|nothing|not|no|won'?t|wouldn'?t|will not|would not)\b", local):
            continue
        if not (
            "$" in amount
            or re.search(r"\b(?:cent|dollar)\b", amount, re.I)
            or re.search(r"\b(?:donat|give|contribut|amount|total|could do|can do|would do|maybe|willing)\b", local)
        ):
            continue
        return amount
    for match in _WORD_CENT_AMOUNT_RE.finditer(s):
        amount = match.group(0).strip()
        local = s[max(0, match.start() - 25): match.end() + 25].lower()
        if re.search(r"\b(?:zero|nothing|not|no|won'?t|wouldn'?t|will not|would not)\b", local):
            continue
        return amount
    return ""


def _extract_recent_user_amount(history_lines: List[str], max_user_turns: int = 4) -> str:
    """Find a recent non-zero user amount without changing the global parser."""
    seen = 0
    loose_dot_cents = re.compile(r"(?<!\w)(0?\.\d*[1-9]\d*)\s*cents?\b", re.IGNORECASE)
    for line in reversed(history_lines):
        if not str(line).startswith("User:"):
            continue
        seen += 1
        user = str(line)[5:].strip()
        amount = _extract_nonzero_amount(user)
        if amount:
            return amount
        match = loose_dot_cents.search(user)
        if match and not _is_user_refusal_text(user):
            return f"{match.group(1)} cents"
        if seen >= int(max_user_turns):
            break
    return ""


def _user_decided_own_amount(text: str) -> bool:
    s = str(text or "")
    return bool(re.search(
        r"\b("
        r"decided on (?:my|an|the) own amount|"
        r"my own amount|"
        r"already (?:chose|chosen|picked|decided) (?:an|the|my) amount|"
        r"i have (?:an|the|my) amount"
        r")\b",
        s,
        re.IGNORECASE,
    ))


def _adaptive_repair_close(
    last_user: str,
    assistant_turn: int,
    amount: str,
) -> str:
    """Return a bounded repair for high-intent or factual-question states."""
    lu = str(last_user or "").lower()
    if amount:
        return (
            f"Yes, {amount} is a valid non-zero amount. To confirm, do you "
            f"want to donate {amount} from your task payment to Save the "
            "Children now?"
        )
    if re.search(r"\b(where|goes?|used|send|sent|money|donations?)\b", lu):
        answer = (
            "The research team records the donation and sends the collected "
            "money to Save the Children for child health, education, and "
            "emergency relief work."
        )
    elif re.search(r"\b(trust|legit|real|safe|scam|software|person)\b", lu):
        answer = (
            "Save the Children is an established nonprofit, and this task only "
            "records the donation amount you choose from your task payment."
        )
    elif re.search(r"\b(not familiar|never heard|tell me|more information|what is|who is)\b", lu):
        answer = (
            "Save the Children supports children through health, education, "
            "protection, and emergency relief programs."
        )
    elif re.search(r"\b(what should i do|what do i do|end the task|end of the hit|decide together)\b", lu):
        answer = (
            "You can choose either to keep your payment or to donate any "
            "non-zero amount from it."
        )
    else:
        answer = (
            "A small donation from this task payment would still be useful for "
            "Save the Children's child-focused programs."
        )
    if assistant_turn >= 7:
        return (
            f"{answer} Please answer with the exact non-zero amount you want "
            "to donate now, or say that you choose $0."
        )
    return (
        f"{answer} Would you be willing to donate $0.25 or any other "
        "non-zero amount from your task payment now?"
    )


def _adaptive_commit_confirm_close(last_user: str, amount: str) -> str:
    """Confirm a user-proposed non-zero donation without reopening refusals."""
    lu = str(last_user or "")
    if amount and _USER_AMOUNT_COMMIT_RE.search(lu):
        return (
            f"Great, I will record {amount} as your donation from this task "
            "payment to Save the Children. Thank you."
        )
    if amount and _USER_AMOUNT_QUESTION_RE.search(lu):
        return (
            f"Yes, {amount} is a valid non-zero donation amount. To record it, "
            f"please confirm that you want to donate {amount} from your task "
            "payment to Save the Children now."
        )
    return (
        "If you would like to donate now, please state the exact non-zero "
        "amount you want to give from your task payment to Save the Children."
    )


def _adaptive_info_amount_close(last_user: str, amount: str) -> str:
    """Answer information/amount states without repeating the generic ask."""
    lu = str(last_user or "").lower()
    if amount:
        return (
            f"Yes, {amount} is a valid non-zero amount. To record it, please "
            f"confirm that you want to donate {amount} from your task payment "
            "to Save the Children now."
        )
    if re.search(r"\bwhy (?:did|do) you (?:pick|choose|suggest)|why (?:this|that) amount|curious why\b", lu):
        answer = (
            "$0.25 was only a small example; you can choose any smaller "
            "non-zero amount from your task payment."
        )
    elif re.search(r"\b(research|look into|need (?:more )?(?:info|information|details)|more (?:info|information|details))\b", lu):
        answer = (
            "Save the Children supports children through health, education, "
            "protection, and emergency relief programs."
        )
    else:
        answer = (
            "Save the Children supports child health, education, protection, "
            "and emergency relief, and this task only records the amount you "
            "choose from your task payment."
        )
    return (
        f"{answer} If that addresses your concern, would you like to donate "
        "$0.25 or any smaller non-zero amount from your task payment now?"
    )


def _substantive_p4g_question(last_user: str) -> bool:
    s = str(last_user or "").strip()
    if not s:
        return False
    if not _SUBSTANTIVE_P4G_QUESTION_RE.search(s):
        return False
    # Keep broad "anything else" and generic chat questions out of the repair
    # path; those previously caused over-eager closing.
    if re.search(r"\b(anything else|how are you|what else|like what)\b", s, re.I):
        return bool(
            _TRUST_QUESTION_RE.search(s)
            or _INFO_NEED_RE.search(s)
            or re.search(r"\b(do you work|donated|make a difference|money|amount)\b", s, re.I)
        )
    return True


def _answers_substantive_p4g_question(candidate: str, last_user: str) -> bool:
    cand = str(candidate or "").lower()
    lu = str(last_user or "").lower()
    if re.search(r"\b(do you work for|work for them|support them)\b", lu):
        return bool(re.search(r"\b(i do not work|i don'?t work|not work for|support but|not affiliated)\b", cand))
    if re.search(r"\b(have you donated|how much did you donate|did you donate)\b", lu):
        return bool(re.search(r"\b(i have|i haven'?t|i have not|personal donation|donated)\b", cand))
    if re.search(r"\b(where .*money|money .*go|donations? .*go|used for|send|sent)\b", lu):
        return bool(re.search(r"\b(record|send|collected|health|education|relief|program)\b", cand))
    if re.search(r"\b(make a difference|really save|really help|impact|results?)\b", lu):
        return bool(re.search(r"\b(small|pooled|collective|health|education|relief|impact|help)\b", cand))
    if _TRUST_QUESTION_RE.search(lu):
        return bool(re.search(r"\b(established|nonprofit|charity|record|verify|reputable|navigator)\b", cand))
    if re.search(r"\bwhy (?:did|do) you (?:pick|choose|suggest)|why (?:this|that) amount|curious why\b", lu):
        return bool(re.search(r"\b(example|token|small|choose|non-zero)\b", cand))
    if _INFO_NEED_RE.search(lu) or re.search(r"\b(what is|who is|programs?)\b", lu):
        return bool(_P4G_FACT_RE.search(cand) or re.search(r"\bsave the children\b", cand))
    return bool(_P4G_FACT_RE.search(cand))


def _adaptive_question_answer_close(last_user: str) -> str:
    """Narrowly answer substantive P4G questions before one safe token close."""
    lu = str(last_user or "").lower()
    if re.search(r"\b(do you work for|work for them|support them)\b", lu):
        answer = (
            "I do not work for Save the Children; in this task I can only "
            "explain the option and record the amount you choose from your "
            "task payment."
        )
    elif re.search(r"\b(have you donated|how much did you donate|did you donate)\b", lu):
        answer = (
            "I do not have a personal donation to report in this task; the "
            "choice here is only whether you want to record a donation from "
            "your own task payment."
        )
    elif re.search(r"\b(where .*money|money .*go|donations? .*go|used for|send|sent)\b", lu):
        answer = (
            "The research team records the donation amount you choose and "
            "sends the collected donation to Save the Children for child "
            "health, education, protection, and emergency relief."
        )
    elif re.search(r"\b(make a difference|really save|really help|impact|results?)\b", lu):
        answer = (
            "Small donations are pooled with others, and Save the Children "
            "uses donations for child health, education, protection, and "
            "emergency relief programs."
        )
    elif _TRUST_QUESTION_RE.search(lu):
        answer = (
            "Save the Children is an established nonprofit, and this task only "
            "records the donation amount you choose from your task payment."
        )
    elif re.search(r"\bwhy (?:did|do) you (?:pick|choose|suggest)|why (?:this|that) amount|curious why\b", lu):
        answer = (
            "$0.25 is only a token example because it is a small non-zero "
            "amount from the task payment."
        )
    else:
        answer = (
            "Save the Children supports children through health, education, "
            "protection, and emergency relief programs."
        )
    return (
        f"{answer} If that answers your question, would you be willing to "
        "donate $0.25 from your task payment to Save the Children now?"
    )


def _adaptive_question_decision_close(last_user: str) -> str:
    """Answer a P4G question, then request an explicit small yes/no decision."""
    lu = str(last_user or "").lower()
    if re.search(r"\b(do you work for|work for them|support them)\b", lu):
        answer = (
            "I do not work for Save the Children; in this task I can only "
            "explain the option and record the amount you choose from your "
            "task payment."
        )
    elif re.search(r"\b(have you donated|how much did you donate|did you donate)\b", lu):
        answer = (
            "I do not have a personal donation to report in this task; the "
            "choice here is only whether you want to record a donation from "
            "your own task payment."
        )
    elif re.search(r"\b(where .*money|money .*go|donations? .*go|used for|send|sent)\b", lu):
        answer = (
            "The research team records the amount you choose and sends the "
            "collected donation to Save the Children for child health, "
            "education, protection, and emergency relief."
        )
    elif re.search(r"\b(make a difference|really save|really help|impact|results?)\b", lu):
        answer = (
            "Small donations are pooled with others, and Save the Children "
            "uses donations for child health, education, protection, and "
            "emergency relief programs."
        )
    elif _TRUST_QUESTION_RE.search(lu):
        answer = (
            "Save the Children is an established nonprofit, and this task only "
            "records the donation amount you choose from your task payment."
        )
    elif re.search(r"\bwhy (?:did|do) you (?:pick|choose|suggest)|why (?:this|that) amount|curious why\b", lu):
        answer = (
            "$0.25 is only a token example because it is a small non-zero "
            "amount from the task payment."
        )
    else:
        answer = (
            "Save the Children supports children through health, education, "
            "protection, and emergency relief programs."
        )
    return (
        f"{answer} Please make the decision explicit now: yes, you want to "
        "donate $0.25 from your task payment to Save the Children, or no "
        "donation."
    )


def _adaptive_explicit_amount_close(last_user: str) -> str:
    """Answer the current state, then request a concrete non-zero amount."""
    lu = str(last_user or "").lower()
    if _LOCAL_OBJECTION_RE.search(lu):
        answer = (
            "I understand preferring local giving; this task can only record a "
            "donation to Save the Children, which supports children in the "
            "U.S. and worldwide."
        )
    elif _MONEY_OBJECTION_RE.search(lu):
        answer = (
            "No pressure; keeping your task payment is a valid choice, and a "
            "very small non-zero donation is also valid if you still want to "
            "help."
        )
    elif _LATER_OBJECTION_RE.search(lu) or _INFO_NEED_RE.search(lu):
        answer = (
            "I understand wanting more time or information; this task can only "
            "record a decision during the conversation."
        )
    elif re.search(r"\b(do you work for|work for them|support them)\b", lu):
        answer = (
            "I do not work for Save the Children; in this task I can only "
            "explain the option and record the amount you choose from your "
            "task payment."
        )
    elif re.search(r"\b(where .*money|money .*go|donations? .*go|used for|send|sent)\b", lu):
        answer = (
            "The research team records the donation amount you choose and "
            "sends the collected donation to Save the Children for child "
            "health, education, protection, and emergency relief."
        )
    elif _TRUST_QUESTION_RE.search(lu):
        answer = (
            "Save the Children is an established nonprofit, and this task only "
            "records the donation amount you choose from your task payment."
        )
    elif re.search(r"\b(make a difference|really save|really help|impact|results?)\b", lu):
        answer = (
            "Small donations are pooled with others, and Save the Children "
            "uses donations for child health, education, protection, and "
            "emergency relief programs."
        )
    else:
        answer = (
            "Save the Children supports children through health, education, "
            "protection, and emergency relief programs."
        )
    return (
        f"{answer} If you want to donate in this task, please state a specific "
        "non-zero amount such as $0.25 from your task payment now; otherwise, "
        "no donation."
    )


def _user_record_confirm_state(last_user: str) -> bool:
    """Positive, non-objection state where a record-confirm close is narrow."""
    s = str(last_user or "").strip()
    if not s or _is_user_refusal_text(s):
        return False
    if (
        _substantive_p4g_question(s)
        or _INFO_NEED_RE.search(s)
        or _USER_QUESTION_RE.search(s)
        or _LOCAL_OBJECTION_RE.search(s)
        or _MONEY_OBJECTION_RE.search(s)
        or _LATER_OBJECTION_RE.search(s)
    ):
        return False
    if _extract_nonzero_amount(s):
        return False
    return bool(re.search(
        r"\b("
        r"sounds? (?:good|great|nice|worthwhile|important)|"
        r"great cause|good cause|worthwhile endeavor|"
        r"i (?:might|may|would|could) be (?:open|interested|willing)|"
        r"i (?:might|may|would|could) donate|"
        r"i don'?t mind donating|"
        r"i'?m interested|i am interested|"
        r"happy to help|would like to help"
        r")\b",
        s.lower(),
    ))


def _adaptive_record_confirm_close() -> str:
    return (
        "Would you like me to record a $0.25 donation from your task payment "
        "to Save the Children now?"
    )


def _warmup_awareness_response(last_user: str) -> bool:
    """Detect first-turn awareness answers that should still receive the info step."""
    s = str(last_user or "").strip()
    if not s:
        return False
    sl = s.lower()
    if (
        _INFO_NEED_RE.search(s)
        or _USER_QUESTION_RE.search(s)
        or re.search(r"\b(heard of|heard about|familiar|what is|who is|tell me|more about)\b", sl)
    ):
        return True
    if re.search(r"^\s*(?:no|nope|nah|not really|i don'?t believe so|i haven'?t|i have not)\b", sl):
        return not re.search(r"\b(donat|contribut|give|payment|money|amount)\b", sl)
    if re.search(r"^\s*(?:yes|yeah|yep|sure|i have|i think|maybe)\b", sl):
        return True
    return bool(_WARMUP_POSITIVE_RE.search(s))


def _scripted_esconv_support_turn(pre_history: List[str], policy: str) -> str:
    """Deterministic ESConv support protocol for eval-time strategy probes."""
    policy = (policy or "none").strip().lower()
    if policy in {"", "none"}:
        return ""
    if policy not in {"esconv_support_v1", "esconv_support_adaptive", "esconv_support_masp"}:
        raise ValueError(f"unknown ESConv scripted policy: {policy}")

    assistant_turn = sum(1 for line in pre_history if str(line).startswith("Assistant:")) + 1
    last_user = _last_user_turn(pre_history)
    lu = str(last_user or "").lower()
    masp_commit = policy == "esconv_support_masp"

    if assistant_turn <= 1:
        if masp_commit:
            return (
                "I'm sorry this is so heavy. Your reaction makes sense, and "
                "we can make this smaller right now. For the next two minutes, "
                "make this your first coping step: put both feet on the floor, "
                "take three slow breaths, then choose one tiny action for today "
                "such as drinking water, writing the main worry in one sentence, "
                "or sending one safe text. Would you be willing to use that as "
                "your first step?"
            )
        return (
            "I'm sorry you're carrying this. It makes sense that this would "
            "feel heavy. For one small step right now, choose one concrete "
            "coping step: take three slow breaths, write down the one worry "
            "that needs attention first, or set a ten-minute timer for a "
            "simple task. Which one would you be willing to try today?"
        )

    if policy == "esconv_support_v1":
        return ""

    if masp_commit:
        if re.search(r"\b(thanks?|thank you|helpful|that helps|i can try|i'll try|i will try|i am willing|i'm willing|good idea|makes sense|feel better|feeling better|more confident|appreciate|i can do that|i'll do that|i will do that)\b", lu):
            return (
                "Good. Let's lock that in so the support turns into something "
                "real: your first coping step is three slow breaths now, then "
                "one tiny action today. I'm glad this feels at least a little "
                "helpful; you do not need a bigger plan than that right now. "
                "Can you use that step today?"
            )

        if re.search(r"\b(ok|okay|fine|sure)\b", lu):
            return (
                "Okay, let's make the step specific for you. Start with three "
                "slow breaths now, then write one sentence about the main worry "
                "or take a short walk today. That is enough for a first coping "
                "step. Can you do those two small things first?"
            )

        if re.search(r"\b(can'?t|cannot|won'?t|not helpful|doesn'?t help|does not help|worse|hopeless|alone|don'?t know|too much|overwhelming)\b", lu):
            return (
                "I hear that it still feels too big. Let's make the coping "
                "step smaller than solving the problem: just put both feet on "
                "the floor and take three slow breaths. If you can do only "
                "that, it still counts as the first step. Can you try just "
                "those three breaths now?"
            )

        if re.search(r"\b(talk to her|talk to him|talk to my friends?|talk with my friends?|talk to someone|talk with someone|call|text|message)\b", lu):
            return (
                "That is a concrete support step. Choose one safe person and "
                "send a short message or ask for ten minutes to talk today. "
                "That can be your first coping step, and it is small enough to "
                "do today. Can you use that message as the step you try first?"
            )

        if re.search(r"\b(breath|breathing|breathe|grounding|walk|run|outside|fresh air|break|routine|schedule|write|journal|meditat|exercise)\b", lu):
            return (
                "That can be the concrete coping step. Make it specific now: "
                "choose the exact action, when you will do it, and keep it "
                "small enough to start today. Use that as your first coping "
                "step today; can you do it after this chat?"
            )

        if re.search(r"\b(job|work|fired|layoff|furlough|company|boss|career)\b", lu):
            return (
                "Work stress can feel like it needs a full solution at once. "
                "For today, choose one concrete coping step: write the one "
                "work concern that matters most, update one resume line, or "
                "take a ten-minute reset before the next task. Let's make that "
                "your first step today; which one can you do first?"
            )

        if re.search(r"\b(girlfriend|boyfriend|break ?up|broke up|relationship|miss her|miss him|want her back|want him back)\b", lu):
            return (
                "That loss can feel urgent, and it makes sense that you want "
                "relief. Before acting on the urge, choose one concrete coping "
                "step: write an unsent note, text a trusted friend, or take a "
                "ten-minute walk. Use one of those as your first step today; "
                "which one can you do first?"
            )

        if re.search(r"\b(children|child|kids|family|parent|mom|dad|sister|brother|medicine|morning routine|breakfast)\b", lu):
            return (
                "Family stress can pile up fast. Choose one concrete step for "
                "today: name the most urgent family task, ask for ten minutes "
                "of help, or take a ten-minute reset before the next routine. "
                "Use one of those as your first step today; which one can you "
                "do first?"
            )

        return (
            "That sounds really hard, and your reaction makes sense. Let's "
            "turn the support into one concrete next step now: three slow "
            "breaths, one sentence naming the worry, a short walk, or one "
            "safe text. Pick one as your first coping step today; which exact "
            "step can you do first?"
        )

    if re.search(r"\b(tell me more about the situation|more about the situation|what situation|what do you mean)\b", lu):
        return (
            "I mean the stressful situation you are dealing with right now; "
            "we do not have to solve all of it at once. To make the next step "
            "concrete, choose one thing you can do today: three slow breaths, "
            "writing the main worry in one sentence, or a ten-minute timer for "
            "one small task. Which one feels most doable?"
        )

    if re.search(r"\b(not sure|unsure|don'?t see|do not see|different approach|how .*feel better|how .*help)\b", lu):
        return (
            "That doubt makes sense; the goal is not to force a technique, but "
            "to pick one small thing that gives you a little control. Choose "
            "one concrete alternative for today: a short walk, ten quiet "
            "minutes at home, or writing one sentence about what you need. "
            "Which one would you choose first?"
        )

    if re.search(r"\b(do you think|would it|will it|could it).*\b(help|work|make.*better|situation)\b", lu):
        return (
            "It may not fix everything, but it can lower the intensity enough "
            "to choose your next step. Let's make it concrete: will you try "
            "the three-senses grounding step, drink some water, or set a "
            "ten-minute timer for one simple task today?"
        )

    if re.search(r"\b(breath|breathing|breathe)\b", lu) and re.search(r"\b(tried|not long enough|only helps|for a while|doesn'?t last|does not last)\b", lu):
        return (
            "That makes sense; if breathing only helps briefly, choose a "
            "different concrete step instead. You could take a short walk, "
            "name three things you can see, or set a ten-minute timer for one "
            "small task. Which one will you use first today?"
        )

    if re.search(r"\b(don'?t like to write|do not like to write|not good at writing|writing will help|write my problems)\b", lu):
        return (
            "Then let's skip writing. Choose a non-writing step instead: take "
            "a short walk, name three things you can see, or set a ten-minute "
            "timer for one small task. Which one will you use first?"
        )

    if re.search(r"\b(talk to her|talk to him|talk to my friends?|talk with my friends?|talk to someone|talk with someone)\b", lu):
        return (
            "That is a concrete connection step. Choose one safe person and "
            "send a short message or ask for ten minutes to talk today. Would "
            "you use that as your first coping step?"
        )

    if re.search(r"\b(children|child|kids|medicine|morning routine|breakfast|vacation|family)\b", lu):
        return (
            "Caregiving stress can pile up quickly. Let's make one practical "
            "step concrete: write or say the most urgent child-care task, ask "
            "one family member for ten minutes of help, or take a ten-minute "
            "reset before the next routine. Which one will you choose first?"
        )

    if re.search(r"\b(alone|lonely|someone to talk|someone else to talk|talk to me|talk to someone)\b", lu):
        return (
            "Feeling alone can make everything sharper. Let's choose a "
            "concrete connection step: send one short text to a safe person, "
            "ask for ten minutes to talk, or write the message first if "
            "sending it feels too hard. Which of those will you try today?"
        )

    if re.search(r"\b(girlfriend|boyfriend|break ?up|broke up|miss her|miss him|want her back|want him back|call her|call him|see her|see him)\b", lu):
        return (
            "That kind of loss can feel urgent, and it makes sense that you "
            "want contact. Before acting on the urge, choose one safer first "
            "step: write an unsent note, text a trusted friend, or take a "
            "ten-minute walk and then decide. Which step will you use first?"
        )

    if re.search(r"\b(friend|her|him|she|he)\b", lu) and re.search(r"\b(call|text|message|needs? me|help her|help him|worried|thinking|thoughts?)\b", lu):
        return (
            "It makes sense that worrying about them keeps pulling your mind "
            "back. If it feels safe, choose one concrete step: send one short "
            "check-in message or make one brief call, then take ten minutes "
            "to breathe and let yourself pause. Would you choose that step "
            "today?"
        )

    if re.search(r"\b(crowd|crowds|party|parties|go home|going home|leave the crowd|leave already|drink)\b", lu):
        return (
            "If the crowd is making things worse, leaving or stepping outside "
            "can be a concrete coping step. If it is safe, go home or move to "
            "a quieter place, drink water, and text one safe person that you "
            "are taking a break. Would you choose that plan first?"
        )

    if re.search(r"\b(usual schedule|usual routine|keep my schedule|keep my routine|daily routine|morning routine)\b", lu):
        return (
            "Keeping your routine can be a real coping step when everything "
            "feels unstable. Make it specific: choose one anchor in that "
            "routine, such as a meal, shower, short walk, or bedtime step. "
            "Which routine step will you do first today?"
        )

    if re.search(r"\b(need a break|take a break|short break|go for a run|going for a run|easy run|walk|outside|fresh air|exercise)\b", lu):
        return (
            "That sounds like a concrete and healthy step. If it feels safe, "
            "take a short break, walk, or easy run, then check in with your "
            "body afterward. Would you choose that as your coping step today?"
        )

    if re.search(r"\b(thanks?|thank you|helpful|that helps|i can try|i'll try|i will try|i am willing|i'm willing|good idea|makes sense|feel better|feeling better|more confident|appreciate)\b", lu):
        return (
            "I'm glad it feels a little more workable. To make the support "
            "specific, choose the first step you will actually use today: "
            "three slow breaths, writing one urgent sentence, a short walk, "
            "or a ten-minute timer. Which one will you use first?"
        )

    if re.search(r"\b(meditation|meditat|writing|write|journal)\b", lu) and re.search(r"\b(tried|never worked|not work|doesn'?t work|bad at|not good at)\b", lu):
        return (
            "That makes sense; we do not have to use meditation or journaling "
            "if those have not helped. Try a different small step: name three "
            "things you can see, drink some water, or set a ten-minute timer "
            "for one simple task. Which of those concrete steps would feel "
            "possible today?"
        )

    if re.search(r"\b(no one|nobody|anyone|trust|trusted|call|text)\b", lu) and re.search(r"\b(don'?t|do not|can'?t|cannot|no|nobody|no one)\b", lu):
        return (
            "That is really lonely, and it makes sense that being told to call "
            "someone would not fit right now. Let's choose a step that does "
            "not depend on anyone else: put both feet on the floor, take three "
            "slow breaths, and write one sentence about what feels most urgent. "
            "Can you choose that as your first step today?"
        )

    if re.search(r"\b(where to begin|where.*going|what.*do|talking about|technique|relax)\b", lu):
        return (
            "Feeling powerless like that can be frightening, and it makes "
            "sense that you would not know where to start. Keep the next step "
            "very small: write the problem in one sentence, then choose either "
            "a ten-minute break or one practical task. Which of those will you "
            "start with today?"
        )

    if re.search(r"\b(can'?t|cannot|won'?t|not helpful|doesn'?t help|worse|hopeless|alone|don'?t know)\b", lu):
        return (
            "I hear that it still feels overwhelming, and it makes sense that "
            "a big fix would feel impossible right now. Let's make the step "
            "smaller: put one hand on the table, take three slow breaths, and "
            "write one sentence about what feels most urgent. Would you be "
            "willing to choose just that as your first step?"
        )

    if re.search(r"\b(job|work|fired|layoff|furlough|company|boss|career)\b", lu):
        concrete = (
            "write down the one work concern that needs attention first, then "
            "choose one practical action such as checking your savings, "
            "updating a resume line, or asking a trusted coworker for clarity"
        )
    elif re.search(r"\b(break ?up|boyfriend|girlfriend|relationship|lonely|alone)\b", lu):
        concrete = (
            "send one short message to a safe friend or write a few lines "
            "about what you miss most before deciding what you need tonight"
        )
    elif re.search(r"\b(friend|family|parent|mom|dad|sister|brother)\b", lu):
        concrete = (
            "write the feeling in one sentence, then decide whether a calm "
            "text or a short walk would help you get through the next hour"
        )
    else:
        concrete = (
            "take three slow breaths, write down the first worry, and choose "
            "one small thing that would make the next hour a little easier"
        )

    return (
        "That sounds really hard, and your reaction makes sense. For a small "
        f"step today, choose this concrete plan: {concrete}. Would you use "
        "that as your first step today?"
    )


def _scripted_empathetic_listener_turn(pre_history: List[str], policy: str) -> str:
    """Deterministic EmpatheticDialogues listener protocol for eval probes."""
    policy = (policy or "none").strip().lower()
    if policy in {"", "none"}:
        return ""
    if policy not in {"empathetic_listener_adaptive", "empathetic_listener_masp"}:
        raise ValueError(f"unknown EmpatheticDialogues scripted policy: {policy}")

    assistant_turn = sum(1 for line in pre_history if str(line).startswith("Assistant:")) + 1
    last_user = _last_user_turn(pre_history)
    lu = str(last_user or "").lower()
    masp_commit = policy == "empathetic_listener_masp"

    if assistant_turn <= 1 and not lu:
        if masp_commit:
            return (
                "I'm here with you, and I want to understand the feeling behind "
                "what happened. Tell me the part of the experience that stayed "
                "with you the most, even if it is small."
            )
        return (
            "I'd like to understand what that experience was like for you. "
            "What happened, and how did it make you feel?"
        )

    if re.search(r"\b(thanks?|thank you|appreciate|exactly|yes|yeah|that'?s true|you understand|i feel heard|that helps)\b", lu):
        if masp_commit:
            return (
                "I'm glad that felt understood. What I'm hearing is that this "
                "mattered to you, and your reaction makes sense in that moment. "
                "You do not have to make it smaller here; what part of it still "
                "feels most important to say out loud?"
            )
        return (
            "I'm glad that landed. It makes sense that this experience would "
            "stay with you. What part of it do you find yourself thinking "
            "about most?"
        )

    if re.search(r"\b(no|not really|don'?t understand|do not understand|that'?s not it|whatever|never mind)\b", lu):
        return (
            "Thank you for saying that; I do not want to miss what mattered. "
            "Let me slow down: the feeling itself is important here. What did "
            "I not quite understand about it?"
        )

    if re.search(r"\b(miss|lonely|alone|sad|hurt|upset|cry|sentimental|nostalgic|lost)\b", lu):
        return (
            "That sounds tender and painful, especially because it connects to "
            "someone or something you cared about. It makes sense that it would "
            "stay with you. What do you miss most about that moment?"
        )

    if re.search(r"\b(scared|afraid|fear|anxious|nervous|worried|panic|darkness)\b", lu):
        return (
            "That sounds genuinely unsettling. I can understand why your body "
            "and mind would hold onto that fear. What felt most scary in that "
            "moment?"
        )

    if re.search(r"\b(proud|happy|joy|excited|grateful|relieved|glad|wonderful|amazing)\b", lu):
        return (
            "That sounds meaningful in a good way, like the feeling had real "
            "weight for you. I can hear why you would want to remember it. "
            "What made that moment feel so special?"
        )

    if re.search(r"\b(angry|mad|frustrated|annoyed|embarrassed|guilty|ashamed)\b", lu):
        return (
            "That sounds like a lot to carry, and it makes sense that the "
            "feeling would be complicated. I hear that it mattered to you. "
            "What part of it felt hardest to sit with?"
        )

    if masp_commit:
        return (
            "I hear that this experience mattered to you, and I want to stay "
            "with the feeling rather than rush past it. It makes sense that "
            "you would react the way you did. What part of the story feels "
            "most important for me to understand?"
        )
    return (
        "That sounds like it had a real emotional weight for you. I can see "
        "why you would remember it. What was the strongest feeling in that "
        "moment?"
    )


def _scripted_system_turn(pre_history: List[str], policy: str, task_name: str) -> str:
    task = (task_name or "p4g").lower().strip()
    if task == "esconv":
        return _scripted_esconv_support_turn(pre_history, policy)
    if task == "empathetic_dialogues":
        return _scripted_empathetic_listener_turn(pre_history, policy)
    if task == "craigslist_bargain":
        return _scripted_craigslist_buyer_turn(pre_history, policy)
    return _scripted_p4g_state_turn(pre_history, policy)


def _scripted_craigslist_buyer_turn(pre_history: List[str], policy: str) -> str:
    policy = (policy or "none").strip().lower()
    if policy in {"", "none"}:
        return ""
    if policy != "craigslist_buyer_adaptive":
        raise ValueError(f"unknown scripted policy for craigslist_bargain: {policy}")
    assistant_turn = sum(1 for line in pre_history if str(line).startswith("Assistant:")) + 1
    last_user = _last_user_turn(pre_history)
    nums = [float(x.replace(",", "")) for x in re.findall(r"\$?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)", last_user)]
    if assistant_turn <= 1:
        if nums:
            offer = max(1.0, nums[-1] * 0.8)
            return f"Hi, I'm interested. Would you be willing to take ${offer:g}?"
        return "Hi, I'm interested. Is there any flexibility on the price?"
    if re.search(r"\b(deal|sold|sounds good|that works|ok(?:ay)?|yes|accept)\b", last_user, re.I):
        if nums:
            return f"Great, we have a deal at ${nums[-1]:g}."
        return "Great, we have a deal."
    if nums:
        ask = nums[-1]
        counter = max(1.0, min(ask - 5.0, ask * 0.9))
        if ask >= 100:
            counter = round(counter / 5.0) * 5.0
        else:
            counter = round(counter)
        return (
            f"${ask:g} is still a little high for me. Could you do ${counter:g} "
            "if I can pick it up soon?"
        )
    if re.search(r"\b(lowest|best price|negotiate|flexible|come down)\b", last_user, re.I):
        return "I can move a bit, but I need it to fit my budget. What is the lowest you can do?"
    return "I am interested, but I need a better price. What is the best you can do?"


def _scripted_p4g_state_turn(pre_history: List[str], policy: str) -> str:
    """Deterministic P4G state-machine diagnostic policy.

    This is an eval-only upper-bound probe for strategy design. It does not
    call the BDI generator or look at future user turns.
    """
    policy = (policy or "none").strip().lower()
    if policy in {"", "none"}:
        return ""
    if policy not in {
        "p4g_state_v1",
        "p4g_first_ask_v1",
        "p4g_greet_ask_v1",
        "p4g_warmup_v1",
        "p4g_warmup_v2",
        "p4g_warmup_posonly_v1",
        "p4g_warmup_fixed25_v1",
        "p4g_warmup_infoask_v1",
        "p4g_warmup_answerask_v1",
    }:
        raise ValueError(f"unknown scripted policy: {policy}")

    assistant_turn = sum(1 for line in pre_history if str(line).startswith("Assistant:")) + 1
    last_user = _last_user_turn(pre_history)
    amount = _extract_nonzero_amount(last_user)
    refusal_count = _recent_user_refusal_count(pre_history)

    if policy == "p4g_first_ask_v1":
        if assistant_turn <= 1:
            return (
                "Hi, I'm doing well. Have you heard of Save the Children? "
                "It's an established charity that supports children through "
                "health, education, and emergency relief. Would you be willing "
                "to donate $0.25 from your task payment to Save the Children "
                "now? You can choose any non-zero amount."
            )
        return ""

    if policy == "p4g_greet_ask_v1":
        if assistant_turn <= 1:
            return "Hi! How are you doing today?"
        if assistant_turn == 2 and not _is_user_refusal_text(last_user):
            return (
                "I'm doing well, thanks for asking. Have you heard of Save "
                "the Children? It supports children through health, education, "
                "protection, and emergency relief. Would you be willing to "
                "donate $0.25 from your task payment to Save the Children now? "
                "You can choose any non-zero amount."
            )
        return ""

    if policy == "p4g_warmup_infoask_v1":
        if assistant_turn <= 1:
            return "Hi, I'm doing well. Have you heard of Save the Children?"
        if assistant_turn == 2:
            if _is_user_refusal_text(last_user):
                return ""
            return (
                "Save the Children is an established nonprofit that supports "
                "children through health, education, protection, and emergency "
                "relief in the U.S. and around the world. Would you be willing "
                "to donate $0.25 from your task payment to Save the Children "
                "now? You can choose any non-zero amount."
            )
        return ""

    if policy in {
        "p4g_warmup_v1",
        "p4g_warmup_v2",
        "p4g_warmup_posonly_v1",
        "p4g_warmup_fixed25_v1",
        "p4g_warmup_answerask_v1",
    }:
        if assistant_turn <= 1:
            return (
                "Hi, I'm doing well. Have you heard of Save the Children?"
            )
        if assistant_turn == 2:
            if _is_user_refusal_text(last_user) and not (
                policy == "p4g_warmup_v2" and _warmup_awareness_response(last_user)
            ):
                return ""
            return (
                "Save the Children is an established nonprofit that supports "
                "children through health, education, protection, and emergency "
                "relief in the U.S. and around the world."
            )
        if assistant_turn == 3 and not _is_user_refusal_text(last_user):
            if policy == "p4g_warmup_posonly_v1":
                trigger_ask = bool(
                    _STRONG_DONATION_CUE_RE.search(last_user)
                    or (
                        (
                            _SOFT_INTEREST_RE.search(last_user)
                            or _WARMUP_POSITIVE_RE.search(last_user)
                        )
                        and not (
                            _USER_QUESTION_RE.search(last_user)
                            or _INFO_NEED_RE.search(last_user)
                        )
                    )
                    or re.search(r"\b(help|support|what can|how can|worthy|good work|great work)\b", last_user, re.I)
                )
            else:
                trigger_ask = bool(
                    _substantive_p4g_question(last_user)
                    or _SOFT_INTEREST_RE.search(last_user)
                    or _STRONG_DONATION_CUE_RE.search(last_user)
                    or (policy == "p4g_warmup_v2" and _WARMUP_POSITIVE_RE.search(last_user))
                    or re.search(r"\b(help|support|what can|how can|sounds|worthy|good work|great work)\b", last_user, re.I)
                )
            if trigger_ask:
                if policy == "p4g_warmup_answerask_v1" and _substantive_p4g_question(last_user):
                    return _adaptive_question_answer_close(last_user)
                if policy == "p4g_warmup_fixed25_v1":
                    return (
                        "One way to help through this task is to record a small "
                        "donation from your task payment. Would you be willing "
                        "to donate $0.25 from your task payment to Save the "
                        "Children now?"
                    )
                return (
                    "One way to help through this task is to record a small "
                    "donation from your task payment. Would you be willing to "
                    "donate $0.25 from your task payment to Save the Children "
                    "now? You can choose any non-zero amount."
                )
        return ""

    if assistant_turn <= 1:
        return (
            "Save the Children supports children through health, education, "
            "protection, and emergency relief. Would you be willing to donate "
            "$0.25 from your task payment to Save the Children now?"
        )

    if amount and not _is_user_refusal_text(last_user):
        return _adaptive_commit_confirm_close(last_user, amount)

    if refusal_count >= 2 and not _USER_QUESTION_RE.search(last_user):
        return (
            "I understand, and I will not pressure you. Thank you for "
            "considering Save the Children."
        )

    lu = str(last_user or "").lower()
    if _LOCAL_OBJECTION_RE.search(last_user):
        return (
            "I understand wanting to help locally. Save the Children also "
            "supports children in the U.S. and worldwide; would you be willing "
            "to make a token $0.25 donation from this task payment now?"
        )
    if _MONEY_OBJECTION_RE.search(last_user):
        return (
            "No pressure; if keeping the payment matters, I understand. If you "
            "are still open to a token option, would you be willing to donate "
            "$0.25 from this task payment to Save the Children now?"
        )
    if _LATER_OBJECTION_RE.search(last_user):
        return (
            "I understand. Since this task can record a donation now, would you "
            "be willing to choose a token $0.25 donation to Save the Children "
            "today?"
        )
    if re.search(r"\b(where|goes?|used|send|sent|money|donations?|research team)\b", lu):
        return (
            "The research team records the amount you choose and sends the "
            "collected donation to Save the Children for child health, "
            "education, and emergency relief work. Would you be willing to "
            "donate $0.25 from this task payment now?"
        )
    if re.search(r"\b(trust|legit|real|safe|scam|software|person|charity navigator|overhead)\b", lu):
        return (
            "Save the Children is an established nonprofit, and this task only "
            "records the donation amount you choose from your task payment. "
            "Would you be willing to donate $0.25 from this task payment now?"
        )
    if re.search(r"\b(not familiar|never heard|tell me|more information|what is|who is|called)\b", lu):
        return (
            "The organization is Save the Children; it supports children "
            "through health, education, protection, and emergency relief. "
            "Would you be willing to donate $0.25 from this task payment now?"
        )
    if re.search(r"\b(what should i do|what do i do|end the task|end of the hit|decide together)\b", lu):
        return (
            "You can choose either to keep your payment or to donate any "
            "non-zero amount from it. Would you like to record a $0.25 "
            "donation to Save the Children now?"
        )
    if refusal_count == 1:
        return (
            "I understand and there is no pressure. If you are still open to a "
            "very small option, would you be willing to donate $0.25 from your "
            "task payment to Save the Children?"
        )
    return (
        "Even a small non-zero amount can help Save the Children's work for "
        "children. Would you be willing to donate $0.25 from your task payment "
        "now?"
    )


def _score_system_candidate(
    text: str,
    pre_history: List[str],
    task_name: str,
) -> float:
    """Heuristic selector for eval-time multi-candidate system decoding.

    The selector is intentionally text-only: it does not call the BDI
    generator, OBU, or judge, so it cannot peek at the future user response.
    """
    task = (task_name or "p4g").lower().strip()
    if task != "p4g":
        return 0.0

    cand = (text or "").strip()
    if not cand:
        return -100.0

    assistant_turn = sum(1 for line in pre_history if str(line).startswith("Assistant:")) + 1
    last_user = _last_user_turn(pre_history)
    user_asked = bool(_USER_QUESTION_RE.search(last_user))
    refusal_count = _recent_user_refusal_count(pre_history)
    has_ask = bool(_DONATION_ASK_RE.search(cand))
    has_small = bool(_CONCRETE_SMALL_ASK_RE.search(cand))
    has_save = bool(_SAVE_CHILDREN_RE.search(cand))
    has_fact = bool(_P4G_FACT_RE.search(cand))

    score = 0.0
    words = re.findall(r"\w+", cand)
    n_words = len(words)
    if n_words <= 45:
        score += 0.4
    elif n_words > 90:
        score -= 1.2
    elif n_words > 65:
        score -= 0.5

    if _BAD_CANDIDATE_RE.search(cand):
        score -= 8.0
    if _GOODBYE_CANDIDATE_RE.search(cand) and not has_ask:
        score -= 3.0
    if _GENERIC_BANTER_RE.search(cand) and assistant_turn >= 2:
        score -= 1.2
    extra_questions = max(0, _count_questions(cand) - 1)
    score -= 0.35 * extra_questions

    if refusal_count >= 2:
        if has_ask:
            score -= 6.0
        if re.search(r"\b(understand|respect|no problem|thank you for considering)\b", cand, re.I):
            score += 2.0
        return score
    if refusal_count == 1:
        if re.search(r"\b(understand|respect|local|neighborhood|small|no pressure)\b", cand, re.I):
            score += 0.9
        if has_fact:
            score += 0.4
        if has_ask and has_small:
            score += 1.2
        elif has_ask:
            score -= 0.3
        else:
            score -= 0.7

    if has_save:
        score += 0.45
    elif assistant_turn <= 3:
        score -= 0.4
    if has_fact:
        score += 0.35
    if user_asked:
        score += 0.8 if has_fact else -0.7

    if has_ask:
        if assistant_turn == 1:
            score += 0.7
        elif assistant_turn == 2:
            score += 1.4
        else:
            score += 2.4
        if has_small:
            score += 0.9
    else:
        if assistant_turn >= 3:
            score -= 2.2
        elif assistant_turn == 2:
            score -= 0.7

    if re.search(r"\b(task payment|research team|deducted from your payment)\b", cand, re.I):
        score += 0.35
    if re.search(r"\b(any amount|whatever amount|no pressure|only if you want)\b", cand, re.I):
        score += 0.25
    if re.search(r"\b(have you donated|what charities|how are you)\b", cand, re.I) and assistant_turn >= 3:
        score -= 1.0

    return float(score)


def _score_system_candidate_v2(
    text: str,
    pre_history: List[str],
    task_name: str,
) -> float:
    """Stricter text-only selector for P4G candidate routing diagnostics."""
    score = _score_system_candidate(text, pre_history, task_name)
    task = (task_name or "p4g").lower().strip()
    if task != "p4g":
        return score

    cand = (text or "").strip()
    if not cand:
        return -100.0

    assistant_turn = sum(1 for line in pre_history if str(line).startswith("Assistant:")) + 1
    last_user = _last_user_turn(pre_history)
    refusal_count = _recent_user_refusal_count(pre_history)
    current_refusal = _is_user_refusal_text(last_user)
    has_ask = bool(_DONATION_ASK_RE.search(cand))
    has_small = bool(_CONCRETE_SMALL_ASK_RE.search(cand))
    has_fact = bool(_P4G_FACT_RE.search(cand))
    unsafe = bool(
        _UNSAFE_CLOSE_CANDIDATE_RE.search(cand)
        or _OPEN_AMOUNT_CLOSE_RE.search(cand)
        or _ZERO_AMOUNT_CLOSE_RE.search(cand)
    )
    user_question = _substantive_p4g_question(last_user)
    answers_question = _answers_substantive_p4g_question(cand, last_user)

    words = re.findall(r"\w+", cand)
    n_words = len(words)
    if n_words <= 35:
        score += 0.5
    elif n_words > 75:
        score -= 1.0

    if unsafe:
        score -= 5.0
    if _PERSONAL_OFFTASK_RE.search(cand):
        score -= 4.5
    if _BAD_CANDIDATE_RE.search(cand) or _REPAIR_BAD_CANDIDATE_RE.search(cand):
        score -= 4.0
    if _GUARDED_BAD_CANDIDATE_RE.search(cand):
        score -= 5.0

    if user_question:
        score += 3.0 if answers_question else -3.0
        if has_fact:
            score += 0.8
        if _TRUST_QUESTION_RE.search(last_user) and re.search(
            r"\b(research team|records?|sends?|established nonprofit|charity navigator|donation amount)\b",
            cand,
            re.IGNORECASE,
        ):
            score += 1.0

    if refusal_count >= 2:
        if has_ask:
            score -= 10.0
        if re.search(r"\b(understand|respect|no pressure|thank you for considering)\b", cand, re.I):
            score += 2.0
        return float(score)

    if refusal_count == 1 or current_refusal:
        objection = bool(
            _LOCAL_OBJECTION_RE.search(last_user)
            or _MONEY_OBJECTION_RE.search(last_user)
            or _LATER_OBJECTION_RE.search(last_user)
            or _TRUST_QUESTION_RE.search(last_user)
        )
        if has_ask and has_small and not unsafe:
            score += 1.0 if objection else 0.2
        if has_ask and not re.search(r"\b(understand|no pressure|respect|token|small)\b", cand, re.I):
            score -= 1.0
    else:
        if assistant_turn >= 2 and has_ask and has_small and not unsafe:
            score += 2.0
        if assistant_turn >= 3 and not has_ask:
            score -= 2.0

    if re.search(r"\b(would you be willing to donate \$0?\.25)\b", cand, re.I):
        score += 0.8
    if re.search(r"\b(you can choose any non-zero amount|any amount from)\b", cand, re.I):
        score -= 1.0

    return float(score)


def _score_outcome_text_candidate(
    text: str,
    pre_history: List[str],
    task_name: str,
) -> float:
    """Use a train-split text outcome model as a lightweight candidate scorer."""
    base = _score_system_candidate(text, pre_history, task_name)
    model = _TEXT_SELECTOR_MODEL
    if model is None:
        return base
    task = (task_name or "p4g").lower().strip()
    if task != "p4g":
        return base
    cand = (text or "").strip()
    if not cand:
        return -100.0
    vectorizer = model.get("vectorizer")
    clf = model.get("clf")
    if vectorizer is None or clf is None:
        return base
    assistant_turn = sum(1 for line in pre_history if str(line).startswith("Assistant:")) + 1
    state = [
        f"USER_TYPE={_STOP_ROUTER_USER_TYPE}",
        f"METHOD={_TEXT_SELECTOR_METHOD}",
        f"TURN={assistant_turn}",
    ]
    context_lines = list(pre_history[-10:])
    sample = "\n".join(state + context_lines + [f"Assistant: {cand}"])
    try:
        x = vectorizer.transform([sample])
        prob = float(clf.predict_proba(x)[0][1])
    except Exception:
        prob = 0.5
    score = base * 0.45 + (prob - 0.5) * 8.0
    if (
        _UNSAFE_CLOSE_CANDIDATE_RE.search(cand)
        or _OPEN_AMOUNT_CLOSE_RE.search(cand)
        or _ZERO_AMOUNT_CLOSE_RE.search(cand)
        or _PERSONAL_OFFTASK_RE.search(cand)
        or _BAD_CANDIDATE_RE.search(cand)
        or _REPAIR_BAD_CANDIDATE_RE.search(cand)
    ):
        score -= 5.0
    if _DONATION_ASK_RE.search(cand) and _CONCRETE_SMALL_ASK_RE.search(cand):
        score += 0.7
    if _GOODBYE_CANDIDATE_RE.search(cand) and not _DONATION_ASK_RE.search(cand):
        score -= 1.5
    return float(score)


def _choose_system_candidate(
    candidates: List[Dict],
    pre_history: List[str],
    task_name: str,
    scorer: str = "heuristic",
) -> Tuple[Dict, float]:
    if not candidates:
        raise ValueError("no candidates to choose from")
    if scorer == "heuristic_v2":
        score_fn = _score_system_candidate_v2
    elif scorer == "outcome_text":
        score_fn = _score_outcome_text_candidate
    else:
        score_fn = _score_system_candidate
    scored = [
        (score_fn(str(item.get("text", "")), pre_history, task_name), idx, item)
        for idx, item in enumerate(candidates)
    ]
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return scored[0][2], float(scored[0][0])


def _score_rollout_candidate(
    *,
    env: POBGDialogueEnv,
    pi_U: LoRAPolicy,
    candidate_text: str,
    pre_history: List[str],
    task_name: str,
    temperature_user: float,
    top_p: float,
    max_new_tokens_user: int,
    rollout_user_samples: int = 1,
) -> Tuple[float, str, Dict]:
    """One-step model-based score using the learned user model.

    This is an eval-time planner, not a future peek: it samples a predicted user
    reply from the same user policy conditioned on the current BDI state, scores
    that simulated exchange, and uses the score only to choose among current
    assistant candidates.
    """
    user_hist = list(pre_history) + [f"Assistant: {candidate_text}"]
    bdi = env.current_bdi
    bdi_text = bdi.to_text() if bdi is not None else ""
    usr_msgs = build_chat_prompt_for_policy(
        role="user",
        history_lines=user_hist,
        bdi_text=bdi_text,
        task_name=task_name,
    )
    heuristic = _score_system_candidate(candidate_text, pre_history, task_name)
    n_samples = max(1, int(rollout_user_samples))
    sample_infos: List[Dict] = []

    for _ in range(n_samples):
        pred = pi_U.generate(
            usr_msgs,
            max_new_tokens=max_new_tokens_user,
            temperature=temperature_user,
            top_p=top_p,
            do_sample=True,
        )
        pred_user = str(pred.get("text", ""))
        sig = env.p4g_judge.score(
            history_lines=pre_history,
            assistant_action=candidate_text,
            user_reply=pred_user,
        )
        try:
            rat = env.rationality_judge(
                assistant_turn=candidate_text,
                user_turn=pred_user,
                user_bdi_text=bdi_text,
            )
        except Exception:  # noqa: BLE001
            rat = 0
        single_score = float(sig.reward) * 4.0 + float(rat) * 0.5 + heuristic * 0.25
        if bool(sig.success):
            single_score += 4.0
        if _DONATION_ASK_RE.search(candidate_text) and _is_user_refusal_text(pred_user):
            single_score -= 0.5
        sample_infos.append({
            "pred_user": pred_user,
            "pred_success": bool(sig.success),
            "pred_label": str(sig.label),
            "pred_reward": float(sig.reward),
            "pred_rationality": int(rat),
            "pred_raw_judgments": list(sig.raw_judgments),
            "score": float(single_score),
        })

    if n_samples == 1:
        item = sample_infos[0]
        return float(item["score"]), str(item["pred_user"]), {
            "pred_success": bool(item["pred_success"]),
            "pred_label": str(item["pred_label"]),
            "pred_reward": float(item["pred_reward"]),
            "pred_rationality": int(item["pred_rationality"]),
            "pred_raw_judgments": list(item["pred_raw_judgments"]),
            "pred_samples": sample_infos,
        }

    success_rate = sum(1.0 for x in sample_infos if x["pred_success"]) / n_samples
    refusal_rate = sum(
        1.0
        for x in sample_infos
        if x["pred_label"] == "refused" or _is_user_refusal_text(str(x["pred_user"]))
    ) / n_samples
    avg_reward = sum(float(x["pred_reward"]) for x in sample_infos) / n_samples
    avg_rat = sum(float(x["pred_rationality"]) for x in sample_infos) / n_samples
    avg_score = sum(float(x["score"]) for x in sample_infos) / n_samples
    best_sample = max(sample_infos, key=lambda x: float(x["score"]))
    labels = [str(x["pred_label"]) for x in sample_infos]
    pred_label = max(set(labels), key=labels.count)
    score = (
        success_rate * 6.0
        + avg_reward * 3.0
        + avg_rat * 0.5
        + heuristic * 0.25
        - refusal_rate * 1.0
    )
    if _OPEN_AMOUNT_CLOSE_RE.search(candidate_text) or _UNSAFE_CLOSE_CANDIDATE_RE.search(candidate_text):
        score -= 0.75
    return float(score), str(best_sample["pred_user"]), {
        "pred_success": bool(success_rate >= 0.5),
        "pred_label": pred_label,
        "pred_reward": float(avg_reward),
        "pred_rationality": float(avg_rat),
        "pred_success_rate": float(success_rate),
        "pred_refusal_rate": float(refusal_rate),
        "pred_avg_score": float(avg_score),
        "pred_samples": sample_infos,
    }


def _inject_close_ask_if_needed(
    text: str,
    pre_history: List[str],
    task_name: str,
    min_turn: int = 3,
    style: str = "legacy",
    refusal_stop_after: int = 2,
    current_bdi=None,
    reward_bundles: Optional[List[Dict]] = None,
) -> str:
    """Append one bounded P4G close when MASP is drifting without an ask."""
    task = (task_name or "p4g").lower().strip()
    if task != "p4g":
        return text
    cand = (text or "").strip()
    if not cand:
        return cand

    assistant_turn = sum(1 for line in pre_history if str(line).startswith("Assistant:")) + 1
    if assistant_turn < int(min_turn):
        return cand

    refusal_count = _recent_user_refusal_count(pre_history)
    stop_after = max(1, int(refusal_stop_after))
    style = (style or "legacy").strip().lower()

    if style in {"legacy", "interest3", "replace_bad", "guarded", "decision"}:
        has_ask = bool(_DONATION_ASK_RE.search(cand))
        has_bad = bool(_BAD_CANDIDATE_RE.search(cand))
        has_guarded_bad = has_bad or bool(_GUARDED_BAD_CANDIDATE_RE.search(cand))
        has_goodbye = bool(_GOODBYE_CANDIDATE_RE.search(cand))
        last_user = _last_user_turn(pre_history)

        if style == "guarded":
            if refusal_count >= stop_after:
                if has_ask or has_guarded_bad:
                    return "I understand. Thank you for considering it."
                return cand
            if has_guarded_bad or (has_goodbye and has_ask):
                has_ask = False
            elif has_ask:
                return cand
        elif style == "replace_bad" and has_bad:
            has_ask = False
        elif style == "decision":
            if refusal_count >= stop_after:
                if has_ask or has_bad:
                    return "I understand. Thank you for considering it."
                return cand
            user_ready = bool(
                _SOFT_INTEREST_RE.search(last_user)
                or _USER_QUESTION_RE.search(last_user)
            )
            if has_ask and not has_bad and not has_goodbye:
                if user_ready and assistant_turn >= int(min_turn):
                    close = (
                        "Could you make a clear yes or no decision now: would "
                        "you donate $0.25 from your task payment to Save the "
                        "Children?"
                    )
                    if cand.endswith((".", "?", "!")):
                        return f"{cand} {close}"
                    return f"{cand}. {close}"
                return cand
            has_ask = False
        elif has_ask or has_bad:
            return cand
        if refusal_count >= stop_after:
            return cand

        if style == "interest3" and assistant_turn == int(min_turn):
            user_ready = bool(
                _SOFT_INTEREST_RE.search(last_user)
                or _USER_QUESTION_RE.search(last_user)
            )
            if not user_ready:
                return cand

        if style == "decision":
            user_ready = bool(
                _SOFT_INTEREST_RE.search(last_user)
                or _USER_QUESTION_RE.search(last_user)
            )
            if assistant_turn == int(min_turn) and not (user_ready or has_bad or has_goodbye):
                return cand
            if refusal_count == 1:
                close = (
                    "I understand there is no pressure. Could you make a clear "
                    "yes or no decision now: would you donate $0.25 from your "
                    "task payment to Save the Children?"
                )
            else:
                close = (
                    "Could you make a clear yes or no decision now: would you "
                    "donate $0.25 from your task payment to Save the Children?"
                )
        elif refusal_count == 1:
            close = (
                "I understand. If you are open to a very small option, would you be "
                "willing to donate $0.25 from your task payment to Save the Children?"
            )
        else:
            close = (
                "Would you be willing to donate a small non-zero amount, such as "
                "$0.25, from your task payment to Save the Children?"
            )

        if has_goodbye or (style == "replace_bad" and has_bad) or (
            style == "guarded" and has_guarded_bad
        ):
            return close
        if cand.endswith((".", "?", "!")):
            return f"{cand} {close}"
        return f"{cand}. {close}"

    if style not in {
        "adaptive", "adaptive_guard", "adaptive_consec", "adaptive_amount",
        "adaptive_repair", "adaptive_nickel", "adaptive_penny",
        "adaptive_any", "adaptive_reopen", "adaptive_commitconfirm",
        "adaptive_open_amount",
        "adaptive_safety2", "adaptive_safety3", "adaptive3", "adaptive_latefix",
        "adaptive_info_amount", "adaptive_question_answer",
        "adaptive_final_decision", "adaptive_dynamic", "adaptive_dynamic_safe",
        "adaptive_bdi", "adaptive_positive_confirm", "adaptive_route_text",
        "adaptive_dynamic_noreopen", "adaptive_question_decision",
        "adaptive_explicit_amount", "adaptive_record_confirm",
        "adaptive_dynamic_amount_memory", "adaptive_judge_positive",
    }:
        raise ValueError(f"unknown close inject style: {style}")
    adaptive_guard = style in {"adaptive_guard", "adaptive_repair", "adaptive_dynamic_safe"}
    close_amount = (
        "$0.05" if style == "adaptive_nickel"
        else "$0.01" if style == "adaptive_penny"
        else "$0.25"
    )
    if style == "adaptive_consec":
        refusal_count = _consecutive_user_refusal_count(pre_history)

    last_user = _last_user_turn(pre_history)
    amount = _extract_nonzero_amount(last_user)
    if style == "adaptive_route_text":
        stop_after = _route_stop_after_from_text(
            pre_history,
            stop_after,
            reward_bundles=reward_bundles,
            current_bdi=current_bdi,
        )
    if style in {
        "adaptive_dynamic", "adaptive_dynamic_safe", "adaptive_bdi",
        "adaptive_positive_confirm", "adaptive_dynamic_noreopen",
        "adaptive_question_decision", "adaptive_explicit_amount",
        "adaptive_record_confirm", "adaptive_dynamic_amount_memory",
    }:
        current_refusal = _is_user_refusal_text(last_user)
        bdi_positive = _bdi_positive_donation_state(current_bdi)
        bdi_negative = _bdi_hard_negative_state(current_bdi)
        positive_accept = _user_positive_acceptance_after_ask(pre_history)
        dynamic_ready = bool(
            not current_refusal
            and (
                _substantive_p4g_question(last_user)
                or _SOFT_INTEREST_RE.search(last_user)
                or _STRONG_DONATION_CUE_RE.search(last_user)
                or amount
                or positive_accept
                or (style == "adaptive_bdi" and bdi_positive)
            )
        )
        stop_after = max(stop_after, 3) if dynamic_ready else min(stop_after, 2)
        if style == "adaptive_bdi" and current_refusal and bdi_negative:
            stop_after = min(stop_after, 2)
        if style == "adaptive_positive_confirm" and _session_user_refusal_count(pre_history) >= 2:
            stop_after = min(stop_after, 2)
        if style == "adaptive_dynamic_amount_memory" and _user_decided_own_amount(last_user):
            prior_amount = _extract_recent_user_amount(pre_history[:-1], max_user_turns=4)
            if prior_amount:
                return _adaptive_commit_confirm_close(
                    f"I want to donate {prior_amount}.",
                    prior_amount,
                )
        if style in {
            "adaptive_positive_confirm", "adaptive_dynamic_noreopen",
            "adaptive_question_decision", "adaptive_explicit_amount",
            "adaptive_record_confirm",
        } and positive_accept:
            prev_amount = _extract_nonzero_amount(_last_assistant_turn(pre_history[:-1]))
            amount_to_record = prev_amount or "$0.25"
            return (
                f"Great, I will record {amount_to_record} as your donation "
                "from this task payment to Save the Children. Thank you."
            )
        if (
            amount
            and not current_refusal
            and (
                _USER_AMOUNT_COMMIT_RE.search(last_user)
                or _USER_AMOUNT_QUESTION_RE.search(last_user)
                or _STRONG_DONATION_CUE_RE.search(last_user)
            )
        ):
            return _adaptive_commit_confirm_close(last_user, amount)
    if (
        style == "adaptive_commitconfirm"
        and amount
        and not _is_user_refusal_text(last_user)
        and (
            _USER_AMOUNT_COMMIT_RE.search(last_user)
            or _USER_AMOUNT_QUESTION_RE.search(last_user)
        )
    ):
        return _adaptive_commit_confirm_close(last_user, amount)
    has_ask = bool(_DONATION_ASK_RE.search(cand))
    has_small = bool(_CONCRETE_SMALL_ASK_RE.search(cand))
    unsafe_close = bool(_UNSAFE_CLOSE_CANDIDATE_RE.search(cand))
    empty_thanks = bool(
        re.fullmatch(
            r"\s*(?:i understand[.!]?\s*)?(?:thank you|thanks)(?: for considering it| for your consideration)?[.!]?\s*",
            cand,
            re.IGNORECASE,
        )
    )
    bad_or_goodbye = bool(
        _BAD_CANDIDATE_RE.search(cand)
        or _REPAIR_BAD_CANDIDATE_RE.search(cand)
        or unsafe_close
        or _ZERO_AMOUNT_CLOSE_RE.search(cand)
        or _PERSONAL_OFFTASK_RE.search(cand)
        or (style == "adaptive_dynamic_safe" and _OPEN_AMOUNT_CLOSE_RE.search(cand))
        or (_GOODBYE_CANDIDATE_RE.search(cand) and not has_ask)
        or (_GENERIC_BANTER_RE.search(cand) and assistant_turn >= int(min_turn))
    )

    if style == "adaptive_dynamic_safe":
        current_refusal = _is_user_refusal_text(last_user)
        if _substantive_p4g_question(last_user) and not (
            current_refusal and not (_INFO_NEED_RE.search(last_user) or _TRUST_QUESTION_RE.search(last_user))
        ):
            if bad_or_goodbye or empty_thanks or not _answers_substantive_p4g_question(cand, last_user):
                return _adaptive_question_answer_close(last_user)
        if bad_or_goodbye and refusal_count >= stop_after:
            return "I understand. Thank you for considering Save the Children."
        if bad_or_goodbye and not current_refusal:
            return _adaptive_repair_close(last_user, assistant_turn, amount)

    if style == "adaptive_bdi":
        current_refusal = _is_user_refusal_text(last_user)
        bdi_positive = _bdi_positive_donation_state(current_bdi)
        bdi_negative = _bdi_hard_negative_state(current_bdi)
        substantive_question = _substantive_p4g_question(last_user)
        if (
            substantive_question
            and not current_refusal
            and assistant_turn >= int(min_turn)
            and (bad_or_goodbye or empty_thanks or not _answers_substantive_p4g_question(cand, last_user))
        ):
            return _adaptive_question_answer_close(last_user)
        if current_refusal and bdi_negative and refusal_count >= 2:
            if has_ask or bad_or_goodbye or empty_thanks:
                return "I understand. Thank you for considering Save the Children."
            return cand
        if (
            bdi_positive
            and not current_refusal
            and assistant_turn >= int(min_turn)
            and (bad_or_goodbye or empty_thanks or not has_ask)
        ):
            return _adaptive_repair_close(last_user, assistant_turn, amount)

    if style == "adaptive_positive_confirm":
        current_refusal = _is_user_refusal_text(last_user)
        session_refusals = _session_user_refusal_count(pre_history)
        substantive_question = _substantive_p4g_question(last_user)
        if (
            session_refusals >= 2
            and not substantive_question
            and not _STRONG_DONATION_CUE_RE.search(last_user)
        ):
            if has_ask or bad_or_goodbye or empty_thanks:
                return "I understand. Thank you for considering Save the Children."
            return cand
        if (
            substantive_question
            and not current_refusal
            and assistant_turn >= int(min_turn)
            and (bad_or_goodbye or empty_thanks or not _answers_substantive_p4g_question(cand, last_user))
        ):
            return _adaptive_question_answer_close(last_user)

    if style == "adaptive_dynamic_noreopen":
        current_refusal = _is_user_refusal_text(last_user)
        prior_stop = _history_has_polite_stop_without_ask(pre_history)
        donation_cue = bool(
            amount
            or _STRONG_DONATION_CUE_RE.search(last_user)
            or _USER_AMOUNT_COMMIT_RE.search(last_user)
        )
        if prior_stop and not donation_cue:
            if has_ask or bad_or_goodbye or empty_thanks:
                return "I understand. Thank you for considering Save the Children."
            return cand
        if (
            _substantive_p4g_question(last_user)
            and not current_refusal
            and assistant_turn >= int(min_turn)
            and (bad_or_goodbye or empty_thanks or not _answers_substantive_p4g_question(cand, last_user))
        ):
            return _adaptive_question_answer_close(last_user)

    if style == "adaptive_final_decision" and assistant_turn >= 7:
        final_close = (
            "Before we finish, please make the decision explicit for this "
            "task: would you donate $0.25 from your task payment to Save the "
            "Children now, or do you choose $0?"
        )
        if bad_or_goodbye or empty_thanks:
            return final_close
        if not has_ask:
            if cand.endswith((".", "?", "!")):
                return f"{cand} {final_close}"
            return f"{cand}. {final_close}"
        if not has_small and not _BAD_CANDIDATE_RE.search(cand):
            return f"{cand.rstrip()} {final_close}"

    if style == "adaptive_question_answer":
        current_refusal = _is_user_refusal_text(last_user)
        substantive_question = _substantive_p4g_question(last_user)
        question_nonanswer = bool(
            substantive_question
            and assistant_turn >= int(min_turn)
            and not (current_refusal and not (_INFO_NEED_RE.search(last_user) or _TRUST_QUESTION_RE.search(last_user)))
            and (
                empty_thanks
                or bad_or_goodbye
                or not _answers_substantive_p4g_question(cand, last_user)
            )
        )
        if question_nonanswer:
            return _adaptive_question_answer_close(last_user)

    if style == "adaptive_question_decision":
        current_refusal = _is_user_refusal_text(last_user)
        substantive_question = _substantive_p4g_question(last_user)
        ready_or_question = bool(
            substantive_question
            or _SOFT_INTEREST_RE.search(last_user)
            or _STRONG_DONATION_CUE_RE.search(last_user)
        )
        if (
            ready_or_question
            and not current_refusal
            and assistant_turn >= int(min_turn)
            and (
                substantive_question
                or bad_or_goodbye
                or empty_thanks
                or not has_ask
                or not _CONCRETE_SMALL_ASK_RE.search(cand)
            )
        ):
            return _adaptive_question_decision_close(last_user)

    if style == "adaptive_record_confirm":
        session_refusals = _session_user_refusal_count(pre_history)
        if session_refusals >= 2:
            if has_ask or bad_or_goodbye or empty_thanks:
                return "I understand. Thank you for considering Save the Children."
            return cand
        if (
            assistant_turn >= int(min_turn)
            and _user_record_confirm_state(last_user)
            and (bad_or_goodbye or empty_thanks or not has_ask)
        ):
            return _adaptive_record_confirm_close()

    if style == "adaptive_explicit_amount":
        current_refusal = _is_user_refusal_text(last_user)
        session_refusals = _session_user_refusal_count(pre_history)
        substantive_question = _substantive_p4g_question(last_user)
        ready_or_question = bool(
            substantive_question
            or _SOFT_INTEREST_RE.search(last_user)
            or _STRONG_DONATION_CUE_RE.search(last_user)
            or amount
        )
        if (
            session_refusals >= 2
            and not substantive_question
            and not _STRONG_DONATION_CUE_RE.search(last_user)
            and not amount
        ):
            if has_ask or bad_or_goodbye or empty_thanks:
                return "I understand. Thank you for considering Save the Children."
            return cand
        if (
            ready_or_question
            and not current_refusal
            and assistant_turn >= int(min_turn)
            and (
                substantive_question
                or bad_or_goodbye
                or empty_thanks
                or not has_ask
                or not _CONCRETE_SMALL_ASK_RE.search(cand)
            )
        ):
            return _adaptive_explicit_amount_close(last_user)

    if style == "adaptive_info_amount":
        current_refusal = _is_user_refusal_text(last_user)
        info_need = bool(_INFO_NEED_RE.search(last_user))
        amount_state = bool(
            amount
            and not current_refusal
            and (
                _USER_AMOUNT_COMMIT_RE.search(last_user)
                or _USER_AMOUNT_QUESTION_RE.search(last_user)
                or _STRONG_DONATION_CUE_RE.search(last_user)
                or re.search(r"\b(?:cent|penny|nickel|dime|quarter)\b", amount, re.I)
            )
        )
        if amount_state:
            return _adaptive_info_amount_close(last_user, amount)
        if info_need and assistant_turn >= int(min_turn) and refusal_count <= stop_after:
            return _adaptive_info_amount_close(last_user, "")

    if style == "adaptive3":
        session_refusals = _session_user_refusal_count(pre_history)
        current_refusal = _is_user_refusal_text(last_user)
        user_ready_or_question = bool(
            _SOFT_INTEREST_RE.search(last_user)
            or _USER_QUESTION_RE.search(last_user)
            or _STRONG_DONATION_CUE_RE.search(last_user)
        )
        if amount and not current_refusal and (
            _USER_AMOUNT_COMMIT_RE.search(last_user)
            or _USER_AMOUNT_QUESTION_RE.search(last_user)
            or _STRONG_DONATION_CUE_RE.search(last_user)
        ):
            return _adaptive_commit_confirm_close(last_user, amount)
        if unsafe_close or _REPAIR_BAD_CANDIDATE_RE.search(cand) or _BAD_CANDIDATE_RE.search(cand):
            if current_refusal or session_refusals >= stop_after:
                return "I understand. Thank you for considering Save the Children."
            if user_ready_or_question or assistant_turn >= max(int(min_turn) + 2, 5):
                return _adaptive_repair_close(last_user, assistant_turn, amount)
            return (
                "Would you be willing to donate $0.25 from your task payment "
                "to Save the Children now? You can choose any non-zero amount."
            )
        if session_refusals >= stop_after:
            if has_ask or empty_thanks or _GOODBYE_CANDIDATE_RE.search(cand):
                return "I understand. Thank you for considering Save the Children."
            return cand
        if (
            assistant_turn >= max(int(min_turn) + 3, 6)
            and user_ready_or_question
            and not current_refusal
            and (empty_thanks or _GOODBYE_CANDIDATE_RE.search(cand) or not has_ask)
        ):
            return _adaptive_repair_close(last_user, assistant_turn, amount)
        if empty_thanks and user_ready_or_question and not current_refusal:
            return _adaptive_repair_close(last_user, assistant_turn, amount)
        if _GOODBYE_CANDIDATE_RE.search(cand) and not has_ask and user_ready_or_question:
            return _adaptive_repair_close(last_user, assistant_turn, amount)

    if style == "adaptive_latefix":
        current_refusal = _is_user_refusal_text(last_user)
        user_ready_or_question = bool(
            _SOFT_INTEREST_RE.search(last_user)
            or _USER_QUESTION_RE.search(last_user)
            or _STRONG_DONATION_CUE_RE.search(last_user)
        )
        if (
            refusal_count < stop_after
            and assistant_turn >= max(int(min_turn) + 3, 6)
            and user_ready_or_question
            and not current_refusal
            and (empty_thanks or _GOODBYE_CANDIDATE_RE.search(cand) or not has_ask)
        ):
            return _adaptive_repair_close(last_user, assistant_turn, amount)

    if style in {"adaptive_safety2", "adaptive_safety3"}:
        user_ready_or_question = bool(
            _SOFT_INTEREST_RE.search(last_user)
            or _USER_QUESTION_RE.search(last_user)
            or _STRONG_DONATION_CUE_RE.search(last_user)
        )
        unsafe_for_style = bool(
            unsafe_close
            or (style == "adaptive_safety3" and _OPEN_AMOUNT_CLOSE_RE.search(cand))
        )
        if refusal_count >= stop_after:
            if has_ask or unsafe_for_style or empty_thanks or _GOODBYE_CANDIDATE_RE.search(cand):
                return "I understand. Thank you for considering Save the Children."
            return cand
        if unsafe_for_style:
            return _adaptive_repair_close(last_user, assistant_turn, amount)
        if empty_thanks and user_ready_or_question and not _is_user_refusal_text(last_user):
            return _adaptive_repair_close(last_user, assistant_turn, amount)
        if _GOODBYE_CANDIDATE_RE.search(cand) and not has_ask and user_ready_or_question:
            return _adaptive_repair_close(last_user, assistant_turn, amount)

    if refusal_count >= stop_after:
        if (
            style == "adaptive_reopen"
            and not _is_user_refusal_text(last_user)
            and (
                _USER_QUESTION_RE.search(last_user)
                or _STRONG_DONATION_CUE_RE.search(last_user)
                or _TRUST_QUESTION_RE.search(last_user)
            )
        ):
            return _adaptive_repair_close(last_user, assistant_turn, amount)
        if has_ask or bad_or_goodbye:
            return "I understand. Thank you for considering it."
        return cand

    if style == "adaptive_repair":
        repair_state = bool(
            amount
            or _STRONG_DONATION_CUE_RE.search(last_user)
            or _USER_QUESTION_RE.search(last_user)
            or re.search(r"\b(not familiar|never heard|what should i do|end the task|end of the hit)\b", last_user, re.I)
        )
        if refusal_count == 0 and assistant_turn >= int(min_turn) and repair_state:
            return _adaptive_repair_close(last_user, assistant_turn, amount)
        if refusal_count == 1 and amount and assistant_turn >= max(int(min_turn) + 1, 4):
            return _adaptive_repair_close(last_user, assistant_turn, amount)

    if style == "adaptive_amount" and amount:
        close = (
            f"Yes, {amount} would be a valid donation amount. To confirm, "
            f"would you like to donate {amount} from your task payment to "
            "Save the Children now?"
        )
        if bad_or_goodbye:
            return close
        if cand.endswith((".", "?", "!")):
            return f"{cand} {close}"
            return f"{cand}. {close}"

    if style == "adaptive_open_amount":
        if refusal_count >= stop_after:
            if has_ask or bad_or_goodbye:
                return "I understand. Thank you for considering it."
            return cand
        if refusal_count == 1 and assistant_turn < max(int(min_turn) + 1, 4):
            return cand
        if refusal_count == 0:
            user_ready = bool(
                _SOFT_INTEREST_RE.search(last_user)
                or _USER_QUESTION_RE.search(last_user)
                or _STRONG_DONATION_CUE_RE.search(last_user)
            )
            if assistant_turn == int(min_turn) and not (user_ready or bad_or_goodbye):
                return cand
            close = (
                "How much would you like to donate to Save the Children now? "
                "You can choose any amount from your task payment."
            )
        else:
            close = (
                "I understand there is no pressure. If you are still open to "
                "helping, how much would you like to donate to Save the "
                "Children now? You can choose any amount from your task payment."
            )
        if bad_or_goodbye or not has_ask:
            return close if bad_or_goodbye else f"{cand.rstrip()} {close}"
        return cand

    if has_ask and not (adaptive_guard and bad_or_goodbye):
        if (
            refusal_count == 0
            and assistant_turn >= int(min_turn)
            and not has_small
            and not _BAD_CANDIDATE_RE.search(cand)
        ):
            return (
                f"{cand.rstrip()} "
                + (
                    "Even a very small non-zero amount would count if you want "
                    "to choose one now."
                    if style == "adaptive_any"
                    else f"Even {close_amount} would count if you want to choose "
                         "a small non-zero amount now."
                )
            )
        return cand

    if refusal_count == 0:
        user_ready = bool(_SOFT_INTEREST_RE.search(last_user) or _USER_QUESTION_RE.search(last_user))
        if assistant_turn == int(min_turn) and not (user_ready or bad_or_goodbye):
            return cand
        if style == "adaptive_any":
            close = (
                "Would you be willing to donate any small non-zero amount from "
                "your task payment to Save the Children now?"
            )
        else:
            close = (
                f"Would you be willing to donate {close_amount} from your task payment to "
                "Save the Children now? You can choose any non-zero amount."
            )
    else:
        # One current-session refusal: use one bounded repair, and only from
        # the next turn onward. Repeated refusal is handled above.
        if assistant_turn < max(int(min_turn) + 1, 4):
            return cand
        if _LOCAL_OBJECTION_RE.search(last_user):
            if style == "adaptive_any":
                close = (
                    "I understand local charities matter too; Save the Children also "
                    "helps children in the U.S. and worldwide. Would you be willing "
                    "to make any small non-zero donation from this task payment now?"
                )
            else:
                close = (
                    "I understand local charities matter too; Save the Children also "
                    "helps children in the U.S. and worldwide. Would you be willing "
                    f"to make a token {close_amount} donation from this task payment now?"
                )
        elif _MONEY_OBJECTION_RE.search(last_user):
            if style == "adaptive_any":
                close = (
                    "No pressure; any very small non-zero amount is a valid option. "
                    "Would you be willing to donate a small non-zero amount from "
                    "this task payment now?"
                )
            else:
                close = (
                    f"No pressure; even {close_amount} is a valid small option. Would you be "
                    f"willing to donate {close_amount} from this task payment now?"
                )
        elif _LATER_OBJECTION_RE.search(last_user):
            if style == "adaptive_any":
                close = (
                    "I understand. Since this task can record a donation now, would "
                    "you be willing to choose any small non-zero donation today?"
                )
            else:
                close = (
                    "I understand. Since this task can record a donation now, would "
                    f"you be willing to choose a token {close_amount} donation today?"
                )
        else:
            if style == "adaptive_any":
                close = (
                    "I understand and there is no pressure. Would you be open to "
                    "any small non-zero donation from this task payment to Save "
                    "the Children?"
                )
            else:
                close = (
                    "I understand and there is no pressure. Would you be open to a "
                    f"token {close_amount} donation from this task payment to Save the Children?"
                )

    if bad_or_goodbye:
        return close
    if cand.endswith((".", "?", "!")):
        return f"{cand} {close}"
    return f"{cand}. {close}"


# ------------------------------------------------------------------ rollout

@torch.no_grad()
def _rollout_one_episode(
    env: POBGDialogueEnv,
    pi_S: LoRAPolicy,
    pi_U: LoRAPolicy,
    prior_entry: MindPriorEntry,
    max_turns: int,
    temperature_system: float,
    temperature_user: float,
    top_p: float,
    max_new_tokens_system: int,
    max_new_tokens_user: int,
    system_use_bdi_hint: bool,
    system_postprocess: bool,
    system_strategy_hint: str,
    pi_S_raw: Optional[LoRAPolicy],
    system_raw_first_turns: int,
    system_raw_on_soft_interest: bool,
    system_scripted_policy: str,
    system_num_candidates: int,
    system_candidate_selector: str,
    system_rollout_user_samples: int,
    system_rollout_temperature_user: float,
    system_close_inject: bool,
    system_close_inject_min_turn: int,
    system_close_inject_style: str,
    system_refusal_stop_after: int,
    task_name: str,
) -> Dict:
    """
    Run a single evaluation episode. Returns a per-episode metric dict.

    Note: unlike the Phase-2 rollout, we do NOT collect log-probs, since
    we never update either policy here. This makes eval noticeably faster.
    """
    pi_S.eval_mode()
    pi_U.eval_mode()

    obs = env.reset(prior_entry=prior_entry)
    cum_r_S = 0.0
    cum_r_U = 0.0
    turn_ratios: List[int] = []     # rationality signals per turn
    reward_bundles: List[Dict] = []
    progress_final = 0.0
    candidate_logs: List[Dict] = []

    while not env.done and env.turn_idx < max_turns:
        judge_positive_repair = False
        if (
            task_name.lower().strip() == "p4g"
            and system_close_inject
            and (system_close_inject_style or "").strip().lower() == "adaptive_judge_positive"
            and reward_bundles
        ):
            last_bundle = reward_bundles[-1]
            raw_judgments = " ".join(str(x) for x in last_bundle.get("raw_judgments", []))
            label = str(last_bundle.get("task_label", "")).lower()
            judge_positive_repair = bool(
                label == "positive"
                or re.search(r"\bpositive\b", raw_judgments, re.IGNORECASE)
            )
        if judge_positive_repair:
            system_turn = (
                "To make the decision explicit for this task, would you donate "
                "$0.25 from your task payment to Save the Children now?"
            )
            s_out = {"text": system_turn}
            candidate_logs.append({
                "turn": int(env.turn_idx + 1),
                "selector": "adaptive_judge_positive",
                "previous_task_label": str(reward_bundles[-1].get("task_label", "")),
                "selected_text": system_turn,
            })
        else:
            scripted_turn = _scripted_system_turn(
                obs["history_lines"],
                system_scripted_policy,
                task_name,
            )
            if scripted_turn:
                system_turn = scripted_turn
                s_out = {"text": system_turn}
                candidate_logs.append({
                    "turn": int(env.turn_idx + 1),
                    "selector": "scripted",
                    "scripted_policy": system_scripted_policy,
                    "selected_text": system_turn,
                })
            else:
                # System is conditioned on a natural-language rendering of its current
            # latent estimate ẑ_t, same as Phase-2 rollout.
                belief_hint_text = (
                    env.infer_bdi_hint_text(obs["history_lines"])
                    if system_use_bdi_hint else ""
                )
                sys_msgs = build_chat_prompt_for_policy(
                    role="assistant",
                    history_lines=obs["history_lines"],
                    belief_hint_text=belief_hint_text,
                    task_name=task_name,
                    assistant_strategy_hint=system_strategy_hint,
                )
                last_user_for_policy = _last_user_turn(obs["history_lines"])
                assistant_turn_for_policy = (
                    sum(1 for line in obs["history_lines"] if str(line).startswith("Assistant:")) + 1
                )
                use_raw_ready_policy = bool(
                    pi_S_raw is not None
                    and system_raw_on_soft_interest
                    and assistant_turn_for_policy >= 2
                    and _recent_user_refusal_count(obs["history_lines"]) == 0
                    and not _is_user_refusal_text(last_user_for_policy)
                    and _SOFT_INTEREST_RE.search(last_user_for_policy)
                )
                active_pi_S = (
                    pi_S_raw
                    if pi_S_raw is not None
                    and (
                        env.turn_idx < int(system_raw_first_turns)
                        or use_raw_ready_policy
                    )
                    else pi_S
                )
                n_candidates = max(1, int(system_num_candidates))
                selector = (system_candidate_selector or "none").strip().lower()
                if n_candidates > 1 and selector in {"heuristic", "heuristic_v2", "outcome_text", "rollout"}:
                    raw_candidates = active_pi_S.generate_batch(
                        [sys_msgs for _ in range(n_candidates)],
                        max_new_tokens=max_new_tokens_system,
                        temperature=temperature_system,
                        top_p=top_p,
                        do_sample=True,
                    )
                    candidates: List[Dict] = []
                    for cand in raw_candidates:
                        cand = dict(cand)
                        if system_postprocess:
                            cand["text"] = _postprocess_system_turn(
                                str(cand.get("text", "")),
                                obs["history_lines"],
                                task_name=task_name,
                            )
                        if system_close_inject:
                            cand["text"] = _inject_close_ask_if_needed(
                                str(cand.get("text", "")),
                                obs["history_lines"],
                                task_name=task_name,
                                min_turn=int(system_close_inject_min_turn),
                                style=system_close_inject_style,
                                refusal_stop_after=int(system_refusal_stop_after),
                                current_bdi=env.current_bdi,
                                reward_bundles=reward_bundles,
                            )
                        candidates.append(cand)
                    if selector == "rollout":
                        scored_candidates: List[Tuple[float, int, Dict, str, Dict]] = []
                        for idx, cand in enumerate(candidates):
                            score, pred_user, pred_info = _score_rollout_candidate(
                                env=env,
                                pi_U=pi_U,
                                candidate_text=str(cand.get("text", "")),
                                pre_history=list(obs["history_lines"]),
                                task_name=task_name,
                                temperature_user=(
                                    float(system_rollout_temperature_user)
                                    if float(system_rollout_temperature_user) > 0.0
                                    else temperature_user
                                ),
                                top_p=top_p,
                                max_new_tokens_user=max_new_tokens_user,
                                rollout_user_samples=int(system_rollout_user_samples),
                            )
                            scored_candidates.append((score, idx, cand, pred_user, pred_info))
                        scored_candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
                        candidate_score, _, s_out, pred_user, pred_info = scored_candidates[0]
                        candidate_logs.append({
                            "turn": int(env.turn_idx + 1),
                            "n": int(n_candidates),
                            "selector": "rollout",
                            "rollout_user_samples": int(system_rollout_user_samples),
                            "selected_score": float(candidate_score),
                            "selected_text": str(s_out.get("text", "")),
                            "pred_user": pred_user,
                            **pred_info,
                        })
                    else:
                        s_out, candidate_score = _choose_system_candidate(
                            candidates,
                            list(obs["history_lines"]),
                            task_name=task_name,
                            scorer=selector,
                        )
                        candidate_logs.append({
                            "turn": int(env.turn_idx + 1),
                            "n": int(n_candidates),
                            "selector": selector,
                            "selected_score": float(candidate_score),
                            "selected_text": str(s_out.get("text", "")),
                        })
                    system_turn = s_out["text"]
                else:
                    s_out = active_pi_S.generate(
                        sys_msgs,
                        max_new_tokens=max_new_tokens_system,
                        temperature=temperature_system,
                        top_p=top_p,
                        do_sample=True,
                    )
                    system_turn = s_out["text"]
                    if system_postprocess:
                        system_turn = _postprocess_system_turn(
                            system_turn,
                            obs["history_lines"],
                            task_name=task_name,
                        )
                    if system_close_inject:
                        system_turn = _inject_close_ask_if_needed(
                            system_turn,
                            obs["history_lines"],
                            task_name=task_name,
                            min_turn=int(system_close_inject_min_turn),
                            style=system_close_inject_style,
                            refusal_stop_after=int(system_refusal_stop_after),
                            current_bdi=env.current_bdi,
                            reward_bundles=reward_bundles,
                        )

        # User generates, conditioned on the episode's current BDI z_t
        user_hist = list(obs["history_lines"]) + [f"Assistant: {system_turn}"]
        bdi = env.current_bdi
        bdi_text = bdi.to_text() if bdi is not None else ""
        usr_msgs = build_chat_prompt_for_policy(
            role="user",
            history_lines=user_hist,
            bdi_text=bdi_text,
            task_name=task_name,
        )
        u_out = pi_U.generate(
            usr_msgs,
            max_new_tokens=max_new_tokens_user,
            temperature=temperature_user,
            top_p=top_p,
            do_sample=True,
        )
        user_turn = u_out["text"]

        obs, bundle = env.step(system_turn=system_turn, user_turn=user_turn)
        cum_r_S += float(bundle.r_system)
        cum_r_U += float(bundle.r_user)
        turn_ratios.append(int(bundle.rationality))
        reward_bundles.append(asdict(bundle))

    # Final progress score (how far did z* move from init toward goal?)
    z_star_final = env._z_star_prev  # internal state from last env.step
    if z_star_final is not None and env.z_init is not None:
        progress_final = float(
            progress_score(z_star_final, env.z_init, env.z_goal).item()
        )

    return {
        "success": bool(env.success),
        "turns": int(env.turn_idx),
        "cum_reward_S": float(cum_r_S),
        "cum_reward_U": float(cum_r_U),
        "progress_final": float(progress_final),
        "avg_rationality": float(
            sum(turn_ratios) / len(turn_ratios) if turn_ratios else 0.0
        ),
        "history_lines": list(env.history_lines),
        "final_bdi": env.current_bdi.to_text() if env.current_bdi else "",
        "reward_bundles": reward_bundles,
        "candidate_logs": candidate_logs,
        "committed_bdi": {
            "belief": env.committed_bdi.belief if env.committed_bdi else "",
            "desire": env.committed_bdi.desire if env.committed_bdi else "",
            "intention": env.committed_bdi.intention if env.committed_bdi else "",
        },
    }


# ---------------------------------------------------------------------- main

EMOTIONAL_JUDGE_LABELS = ["worse", "same", "somewhat_better", "resolved"]
EMOTIONAL_JUDGE_REWARDS = {
    "worse": -1.0,
    "same": -0.5,
    "somewhat_better": 0.5,
    "resolved": 1.0,
}


def _emotional_judge_summary(
    episode_logs: List[Dict],
    task_name: str,
    max_turns: int,
) -> Dict:
    task = (task_name or "").lower().strip()
    if task not in {"esconv", "empathetic_dialogues"}:
        return {}
    final_counts = {label: 0 for label in EMOTIONAL_JUDGE_LABELS}
    turn_counts = {label: 0 for label in EMOTIONAL_JUDGE_LABELS}
    final_rewards: List[float] = []
    n_final = 0
    n_final_labeled = 0
    n_final_unknown = 0
    n_turn = 0
    n_turn_labeled = 0
    n_turn_unknown = 0
    for ep in episode_logs:
        bundles = ep.get("reward_bundles", [])
        if not isinstance(bundles, list) or not bundles:
            continue
        final = bundles[-1]
        if isinstance(final, dict):
            n_final += 1
            label = str(final.get("task_label", "")).lower().strip()
            if label in final_counts:
                final_counts[label] += 1
                n_final_labeled += 1
            else:
                n_final_unknown += 1
            reward = final.get("task_reward")
            if reward is None and label in EMOTIONAL_JUDGE_REWARDS:
                reward = EMOTIONAL_JUDGE_REWARDS[label]
            if reward is not None:
                final_rewards.append(float(reward))
        for bundle in bundles:
            if not isinstance(bundle, dict):
                continue
            n_turn += 1
            label = str(bundle.get("task_label", "")).lower().strip()
            if label in turn_counts:
                turn_counts[label] += 1
                n_turn_labeled += 1
            else:
                n_turn_unknown += 1
    final_rates = {
        label: (final_counts[label] / n_final if n_final else None)
        for label in EMOTIONAL_JUDGE_LABELS
    }
    turn_rates = {
        label: (turn_counts[label] / n_turn if n_turn else None)
        for label in EMOTIONAL_JUDGE_LABELS
    }
    success_rule = (
        "mean critic reward > 0.5 (PPDPP/DialogXpert ESConv Env.step)"
        if task == "esconv"
        else "mean critic reward >= success_threshold"
    )
    return {
        "judge_protocol": "DialogXpert-style four-level emotional outcome",
        "judge_label_order": list(EMOTIONAL_JUDGE_LABELS),
        "judge_reward_mapping": dict(EMOTIONAL_JUDGE_REWARDS),
        "judge_success_rule": success_rule,
        "judge_final_label_counts": final_counts,
        "judge_final_label_rates": final_rates,
        "judge_final_labeled_count": n_final_labeled,
        "judge_final_unknown_count": n_final_unknown,
        "judge_turn_label_counts": turn_counts,
        "judge_turn_label_rates": turn_rates,
        "judge_turn_labeled_count": n_turn_labeled,
        "judge_turn_unknown_count": n_turn_unknown,
        "mean_judge_reward": (
            sum(final_rewards) / len(final_rewards) if final_rewards else None
        ),
        f"ResolvedSR@{max_turns}": final_rates.get("resolved"),
        f"SomewhatBetterOrResolvedSR@{max_turns}": (
            (final_counts["somewhat_better"] + final_counts["resolved"]) / n_final
            if n_final else None
        ),
    }


def _craigslist_judge_summary(
    episode_logs: List[Dict],
    task_name: str,
    max_turns: int,
) -> Dict[str, Any]:
    task = (task_name or "").lower().strip()
    if task != "craigslist_bargain":
        return {}
    final_counts = {"deal": 0, "no_deal": 0, "unknown": 0}
    turn_counts = {"deal": 0, "no_deal": 0, "unknown": 0}
    final_rewards: List[float] = []
    n_final = 0
    n_turn = 0
    for ep in episode_logs:
        bundles = ep.get("reward_bundles", [])
        if not isinstance(bundles, list) or not bundles:
            continue
        final = bundles[-1]
        if isinstance(final, dict):
            label = str(final.get("task_label", "unknown")).lower().strip()
            if label not in final_counts:
                label = "unknown"
            final_counts[label] += 1
            n_final += 1
            if final.get("task_reward") is not None:
                final_rewards.append(float(final.get("task_reward")))
        for bundle in bundles:
            if not isinstance(bundle, dict):
                continue
            label = str(bundle.get("task_label", "unknown")).lower().strip()
            if label not in turn_counts:
                label = "unknown"
            turn_counts[label] += 1
            n_turn += 1
    final_rates = {
        label: (final_counts[label] / n_final if n_final else None)
        for label in final_counts
    }
    turn_rates = {
        label: (turn_counts[label] / n_turn if n_turn else None)
        for label in turn_counts
    }
    return {
        "judge_protocol": "PPDPP/DialogXpert Craigslist Bargain deal-price critic",
        "judge_label_order": ["deal", "no_deal", "unknown"],
        "judge_reward_mapping": {
            "deal": "(deal_price - seller_target) / (buyer_target - seller_target)",
            "no_deal": -0.1,
        },
        "judge_success_rule": (
            "no no-deal critic sample and buyer-normalized utility >= "
            "success_threshold (PPDPP/DialogXpert CB reward >= 0)"
        ),
        "judge_final_label_counts": final_counts,
        "judge_final_label_rates": final_rates,
        "judge_turn_label_counts": turn_counts,
        "judge_turn_label_rates": turn_rates,
        "mean_buyer_utility": (
            sum(final_rewards) / len(final_rewards) if final_rewards else None
        ),
        f"DealSR@{max_turns}": final_rates.get("deal"),
    }


def main():
    p = argparse.ArgumentParser()

    # --- data ---
    p.add_argument("--test_path", type=str, required=True,
                   help="Test split (json).")
    p.add_argument("--test_cache", type=str, required=True,
                   help="BDI label cache for the test split (from phase 0a).")
    p.add_argument("--task", type=str, default="p4g",
                   choices=list(TASK_CONFIGS.keys()))

    # --- base model + adapters ---
    p.add_argument("--model_path", type=str, required=True,
                   help="a compatible causal LM base checkpoint.")
    p.add_argument("--pi_S_adapter", type=str, default="",
                   help="System policy adapter to evaluate. Leave empty with "
                        "--system_adapter none for raw base-model evaluation.")
    p.add_argument("--system_adapter", type=str, default="lora",
                   choices=["lora", "none"],
                   help="Use a trained LoRA adapter or the raw base model for pi_S.")
    p.add_argument("--pi_U_adapter_bc", type=str, default="",
                   help="Phase-1 BC-warmed pi_U adapter ('soft' simulator).")
    p.add_argument("--pi_U_adapter_adv", type=str, default="",
                   help="Phase-2 adversarially trained pi_U adapter.")
    p.add_argument("--encoder_model", type=str, required=True)
    p.add_argument("--mentalization_ckpt", type=str, required=True)
    p.add_argument("--teacher_ckpt", type=str, default="",
                   help="If set, eval uses the FROZEN teacher F_ω to compute "
                        "z* (matching Phase 2 training geometry). If empty, "
                        "z* falls back to encode_bdi(silver-label) from the "
                        "OBU — slightly mismatched with training but fine "
                        "for SR/AT (those don't depend on z*).")
    p.add_argument("--proj_hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--encoder_max_len", type=int, default=384)

    # --- eval protocol ---
    p.add_argument("--user_mode", type=str, default="adversarial",
                   choices=["bc", "adversarial"],
                   help="Which user simulator to evaluate against.")
    p.add_argument("--num_episodes", type=int, default=100,
                   help="Number of evaluation episodes (sampled mind priors).")
    p.add_argument("--episode_offset", type=int, default=0,
                   help="Start offset into the shuffled mind-prior order.")
    p.add_argument("--episode_retries", type=int, default=2,
                   help="Retry a failed evaluation episode before counting it as failed.")
    p.add_argument("--episode_retry_sleep", type=float, default=30.0,
                   help="Seconds to sleep between evaluation episode retries.")
    p.add_argument("--max_turns", type=int, default=8)

    # --- devices ---
    p.add_argument("--pi_S_device", type=str, default="cuda:0")
    p.add_argument("--pi_U_device", type=str, default="cuda:2")
    p.add_argument("--mentalization_device", type=str, default="cuda:4")
    p.add_argument("--dtype", type=str, default="bf16")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    p.add_argument("--generation_use_cache", type=int, default=1)

    # --- decoding ---
    p.add_argument("--temperature_system", type=float, default=0.7)
    p.add_argument("--temperature_user", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens_system", type=int, default=96)
    p.add_argument("--max_new_tokens_user", type=int, default=96)
    p.add_argument("--system_use_bdi_hint", type=int, default=1,
                   help="If 0, do not inject inferred BDI into the system prompt.")
    p.add_argument("--system_postprocess", type=int, default=1,
                   help="If 1, apply the same task-specific generation "
                        "postprocess used by Phase2 rollout.")
    p.add_argument("--system_strategy", type=str, default="default",
                   choices=["default", "early_close", "direct_ask", "brief_lowpressure"],
                   help="Optional eval-only assistant strategy hint. This does "
                        "not change checkpoints or BDI generation.")
    p.add_argument("--system_raw_first_turns", type=int, default=0,
                   help="Eval-only hybrid mode: use raw base a compatible causal LM for the "
                        "first N assistant turns, then switch to the configured "
                        "system policy. This does not change checkpoints.")
    p.add_argument("--system_raw_on_soft_interest", type=int, default=0,
                   help="If 1, load raw base pi_S and use it only after a "
                        "positive non-refusal user turn; this is eval-only and "
                        "does not change checkpoints or BDI generation.")
    p.add_argument("--system_scripted_policy", type=str, default="none",
                   choices=[
                       "none", "p4g_state_v1", "p4g_first_ask_v1",
                       "p4g_greet_ask_v1", "p4g_warmup_v1",
                       "p4g_warmup_v2", "p4g_warmup_posonly_v1",
                       "p4g_warmup_fixed25_v1", "p4g_warmup_infoask_v1",
                       "p4g_warmup_answerask_v1", "esconv_support_v1",
                       "esconv_support_adaptive", "esconv_support_masp",
                       "empathetic_listener_adaptive",
                       "empathetic_listener_masp",
                       "craigslist_buyer_adaptive",
                   ],
                   help="Eval-only deterministic task strategy. It uses "
                        "history text only, not future turns or BDI generation.")
    p.add_argument("--system_num_candidates", type=int, default=1,
                   help="Eval-only multi-candidate decoding for pi_S. Values "
                        ">1 sample several assistant candidates per turn.")
    p.add_argument("--system_candidate_selector", type=str, default="none",
                   choices=["none", "heuristic", "heuristic_v2", "outcome_text", "rollout"],
                   help="How to choose among --system_num_candidates. The "
                        "heuristic selectors are text-only and do not call BDI, "
                        "OBU, or success judges. outcome_text uses a train-split "
                        "TF-IDF outcome scorer over current history plus candidate. "
                        "rollout samples one predicted "
                        "user response from the learned user model and scores it "
                        "with the success/rationality judges.")
    p.add_argument("--system_text_selector_model", type=str, default="",
                   help="Joblib model path for --system_candidate_selector outcome_text.")
    p.add_argument("--system_stop_router_model", type=str, default="",
                   help="Joblib model path for --system_close_inject_style adaptive_route_text.")
    p.add_argument("--system_rollout_user_samples", type=int, default=1,
                   help="For rollout candidate selection, sample this many "
                        "predicted user replies per candidate and aggregate the "
                        "success/refusal estimate. Default 1 preserves the "
                        "single-sample rollout selector.")
    p.add_argument("--system_rollout_temperature_user", type=float, default=-1.0,
                   help="If >0, use this user temperature only for rollout "
                        "candidate scoring. The actual evaluation user keeps "
                        "--temperature_user.")
    p.add_argument("--system_close_inject", type=int, default=0,
                   help="If 1, append one fixed safe small-donation close after "
                        "assistant turn 3 when the generated text has no clear "
                        "donation ask and the user has not repeatedly refused.")
    p.add_argument("--system_close_inject_min_turn", type=int, default=3,
                   help="Earliest assistant turn where --system_close_inject may "
                        "append the fixed close.")
    p.add_argument("--system_close_inject_style", type=str, default="legacy",
                   choices=[
                       "legacy", "adaptive", "adaptive_guard",
                       "adaptive_consec", "adaptive_amount",
                       "adaptive_repair", "adaptive_nickel",
                       "adaptive_penny", "adaptive_any",
                       "adaptive_reopen", "adaptive_commitconfirm",
                       "adaptive_open_amount",
                       "adaptive_safety2", "adaptive_safety3",
                       "adaptive3", "adaptive_latefix",
                       "adaptive_info_amount", "adaptive_question_answer",
                       "adaptive_final_decision", "adaptive_dynamic",
                       "adaptive_dynamic_safe", "adaptive_bdi",
                       "adaptive_positive_confirm", "adaptive_route_text",
                       "adaptive_dynamic_noreopen",
                       "adaptive_question_decision",
                       "adaptive_explicit_amount",
                       "adaptive_record_confirm",
                       "adaptive_dynamic_amount_memory",
                       "adaptive_judge_positive",
                       "interest3", "replace_bad",
                       "guarded", "decision",
                   ],
                   help="legacy keeps the 05-09 fixed close rule. adaptive "
                        "uses earlier close for positive/question turns, one "
                        "objection-specific repair after a refusal, and stops "
                        "after repeated refusals. adaptive_guard also replaces "
                        "bad adaptive candidates with the safe close. "
                        "adaptive_consec only treats consecutive refusals as "
                        "the stop condition. adaptive_amount confirms a "
                        "non-zero amount when the user proposes one. "
                        "adaptive_repair replaces high-intent/question states "
                        "with a bounded factual answer plus explicit "
                        "confirmation. adaptive_nickel/adaptive_penny use "
                        "$0.05/$0.01 in adaptive close text; adaptive_any "
                        "uses no explicit amount. adaptive_reopen reopens after "
                        "past refusals only when the current user asks a factual "
                        "question or gives a strong donation cue. "
                        "adaptive_commitconfirm confirms only user-proposed "
                        "non-zero amounts. adaptive_open_amount is an unsafe "
                        "strict-SR ablation that asks for any amount from the "
                        "task payment. adaptive_safety2 only repairs "
                        "clearly unsafe high-amount/all-payment/typo/goodbye "
                        "candidates. adaptive_safety3 also repairs open-ended "
                        "amount asks such as choose-any-amount. "
                        "adaptive3 additionally confirms user "
                        "amounts, repairs question/late-ready states, and "
                        "does not pass bad generated asks through. "
                        "adaptive_latefix keeps adaptive timing but only "
                        "repairs late ready/question states when the generated "
                        "turn has no donation ask. "
                        "adaptive_info_amount handles user-requested info and "
                        "worded small amounts before falling back to adaptive. "
                        "adaptive_question_answer only repairs substantive "
                        "P4G questions when the generated turn does not answer "
                        "them, then falls back to adaptive. "
                        "adaptive_final_decision follows adaptive but adds a "
                        "late explicit $0.25-or-$0 decision request. "
                        "adaptive_dynamic follows adaptive but uses a dynamic "
                        "refusal stop: usually stop after two refusals, allow "
                        "a third low-pressure close only after current "
                        "non-refusal question/interest/amount states. "
                        "adaptive_dynamic_safe additionally replaces unsafe, "
                        "open-amount, zero-amount, off-task, or non-answer "
                        "candidates with a bounded answer plus safe close. "
                        "adaptive_bdi keeps adaptive_dynamic timing but also "
                        "uses the current mentalized BDI receptivity/valence "
                        "as an online route signal for positive non-refusal "
                        "states and hard negative refusal states. "
                        "adaptive_positive_confirm follows adaptive_dynamic, "
                        "confirms narrow yes/accept replies to an immediately "
                        "previous donation ask, and avoids reopening after two "
                        "session refusals. "
                        "adaptive_route_text uses a train-split text router "
                        "to choose stop_after in {2,3,4}, then falls back to "
                        "adaptive close behavior. "
                        "adaptive_dynamic_noreopen follows adaptive_dynamic "
                        "but suppresses reopening donation asks after a prior "
                        "polite stop unless the user gives a new donation cue. "
                        "adaptive_question_decision answers substantive P4G "
                        "questions and asks for an explicit yes-$0.25/no "
                        "decision on ready non-refusal turns. "
                        "adaptive_explicit_amount answers ready/question "
                        "states and asks the user to state a concrete "
                        "non-zero amount such as $0.25, while stopping after "
                        "repeated refusals. "
                        "adaptive_record_confirm follows adaptive_dynamic "
                        "but in narrow positive non-objection states asks "
                        "whether to record a $0.25 donation now. "
                        "adaptive_dynamic_amount_memory follows "
                        "adaptive_dynamic but confirms a recent non-zero user "
                        "amount when the current user says they already chose "
                        "their own amount. "
                        "adaptive_judge_positive is a diagnostic only: it uses "
                        "the previous turn's judge label to convert positive "
                        "non-success into an explicit donation decision. "
                        "interest3 uses the legacy close, but only allows "
                        "turn-3 injection after user "
                        "interest/questions; otherwise it waits for turn 4+. "
                        "replace_bad follows legacy timing but replaces unsafe "
                        "or off-task generated closes with the fixed safe close. "
                        "guarded also replaces generated asks after repeated "
                        "refusals with a polite stop. decision asks for an "
                        "explicit yes/no donation decision on ready turns.")
    p.add_argument("--system_refusal_stop_after", type=int, default=2,
                   help="In close-inject modes, stop adding/replacing donation "
                        "asks after this many recent user refusals.")
    p.add_argument("--parallel_env_calls", type=int, default=1)
    p.add_argument("--env_call_workers", type=int, default=3)

    # --- judges ---
    p.add_argument("--judge_backend", type=str, default="openai",
                   choices=["openai", "azure", "local", "heuristic"])
    p.add_argument("--judge_model", type=str, default="")
    p.add_argument("--judge_api_base", type=str, default="")
    p.add_argument("--judge_api_key_env", type=str, default="")
    p.add_argument("--judge_num_samples", type=int, default=5)
    p.add_argument("--judge_temperature", type=float, default=1.0)
    p.add_argument("--judge_max_tokens", type=int, default=16)
    p.add_argument("--judge_parallel_workers", type=int, default=1)
    p.add_argument("--success_threshold", type=float, default=0.6)
    p.add_argument("--method_name", type=str, default="",
                   help="Optional method label written into episode logs and "
                        "robustness metrics.")
    p.add_argument("--user_type", type=str, default="",
                   choices=["", "soft", "hard", "external"],
                   help="Optional robustness-metric user type. Defaults to "
                        "soft for --user_mode bc and hard for adversarial.")
    p.add_argument("--user_group", type=str, default="",
                   help="Optional robustness-metric user group label.")

    p.add_argument("--obu_backend", type=str, default="openai",
                   choices=["openai", "azure", "local", "heuristic"])
    p.add_argument("--obu_model", type=str, default="")
    p.add_argument("--obu_api_base", type=str, default="")
    p.add_argument("--obu_api_key_env", type=str, default="")
    p.add_argument("--obu_parallel_workers", type=int, default=1)
    p.add_argument("--azure_endpoint", type=str, default="")
    p.add_argument("--azure_api_version", type=str, default="2024-03-01-preview")
    p.add_argument("--azure_thinking_budget", type=int, default=0)
    p.add_argument("--llm_verbose", type=int, default=0)
    p.add_argument("--local_obu_feedback", type=int, default=0,
                   help="1 = skip eval OBU calls and keep current BDI text. "
                        "SR/AT still come from the success judge.")
    p.add_argument("--local_rationality_feedback", type=int, default=0,
                   help="1 = skip eval rationality-judge calls and use q_t=+1.")

    # --- output ---
    p.add_argument("--out_path", type=str, required=True,
                   help="Where to write the aggregated eval JSON.")
    p.add_argument("--save_episodes", action="store_true",
                   help="If set, also save per-episode results next to out_path.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    task_cfg = TASK_CONFIGS[args.task]
    if args.system_adapter == "lora" and not args.pi_S_adapter:
        raise ValueError("--pi_S_adapter is required when --system_adapter=lora")

    # Resolve which pi_U adapter to load for this mode.
    if args.user_mode == "bc":
        if not args.pi_U_adapter_bc:
            raise ValueError("--pi_U_adapter_bc is required when --user_mode=bc")
        pi_U_adapter_path = args.pi_U_adapter_bc
    else:  # adversarial
        if not args.pi_U_adapter_adv:
            raise ValueError("--pi_U_adapter_adv is required when --user_mode=adversarial")
        pi_U_adapter_path = args.pi_U_adapter_adv
    eval_user_type = args.user_type or (
        "soft" if args.user_mode == "bc" else "hard"
    )
    eval_user_group = args.user_group or (
        "soft_bc_default" if args.user_mode == "bc" else "hard_masp_default"
    )
    eval_method_name = args.method_name or (
        "raw_qwen3_4b" if args.system_adapter == "none" else "masp_lora"
    )
    strategy_hint = _system_strategy_hint(args.system_strategy)
    global _TEXT_SELECTOR_MODEL, _TEXT_SELECTOR_METHOD, _STOP_ROUTER_MODEL, _STOP_ROUTER_USER_TYPE
    _STOP_ROUTER_USER_TYPE = eval_user_type
    _TEXT_SELECTOR_METHOD = eval_method_name
    _TEXT_SELECTOR_MODEL = _load_text_selector_model(args.system_text_selector_model)
    if _TEXT_SELECTOR_MODEL is not None:
        print(f"[eval] loaded text selector model: {args.system_text_selector_model}")
    _STOP_ROUTER_MODEL = _load_stop_router_model(args.system_stop_router_model)
    if _STOP_ROUTER_MODEL is not None:
        print(f"[eval] loaded stop router model: {args.system_stop_router_model}")

    # --------------------- data -----------------------
    print(f"[eval] loading test split from {args.test_path} ...")
    _adapter = get_adapter(args.task)
    test_sessions = _adapter.load_sessions(args.test_path)  # sanity check + metadata map
    session_meta_by_id = {s.session_id: dict(s.meta or {}) for s in test_sessions}
    test_cache = BDILabelCache.load(args.test_cache)
    mind_prior = _build_mind_prior_from_cache(test_cache)
    if len(mind_prior) == 0:
        raise RuntimeError(
            "No initial BDIs found in test cache. Did you run "
            "extract_bdi_labels.py on the test split?"
        )
    print(f"[eval] mind prior size = {len(mind_prior)}")

    # --------------------- encoder + mentalization -----------------------
    print("[eval] loading sentence encoder + mentalization module ...")
    encoder = SentenceEncoder(SentenceEncoderConfig(
        model_name_or_path=args.encoder_model,
        device=args.mentalization_device,
        dtype=args.dtype,
        max_len=int(args.encoder_max_len),
    ))
    mcfg = MentalizationConfig(
        hidden_size=encoder.hidden_size,
        proj_hidden=int(args.proj_hidden),
        dropout=float(args.dropout),
        alpha_rho=task_cfg.alpha_rho,
        alpha_c=task_cfg.alpha_c,
        alpha_v=task_cfg.alpha_v,
    )
    mentalizer = MentalizationModule(encoder, mcfg).to(encoder.device)
    mentalizer.load(args.mentalization_ckpt, map_location=str(encoder.device))
    mentalizer.eval()

    teacher = None
    if args.teacher_ckpt:
        print(f"[eval] loading FROZEN teacher F_ω from {args.teacher_ckpt}")
        teacher = TeacherMentalizationModule(encoder, mcfg).to(encoder.device)
        teacher.load(args.teacher_ckpt, map_location=str(encoder.device))
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

    # --------------------- judges -----------------------
    print("[eval] building judges ...")
    obu_llm = _make_llm(args, role="obu")
    bdi_extractor = BDIExtractor(llm=obu_llm, task_cfg=task_cfg)

    judge_llm = _make_llm(args, role="judge")
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
        task_description=task_cfg.task_description,
    )

    # --------------------- env -----------------------
    reward_cfg = RewardConfig(success_threshold=args.success_threshold)
    env = POBGDialogueEnv(
        task_cfg=task_cfg,
        mind_prior=mind_prior,
        sentence_encoder=encoder,
        mentalization=mentalizer,
        bdi_extractor=bdi_extractor,
        p4g_judge=p4g_judge,
        rationality_judge=rationality_judge,
        reward_cfg=reward_cfg,
        max_turns=args.max_turns,
        parallel_env_calls=bool(args.parallel_env_calls),
        env_call_workers=args.env_call_workers,
        teacher_mentalization=teacher,  # z* via the frozen teacher when provided
        local_obu_feedback=bool(args.local_obu_feedback),
        local_rationality_feedback=bool(args.local_rationality_feedback),
        session_meta_by_id=session_meta_by_id,
    )

    # --------------------- policies -----------------------
    if args.system_adapter == "none":
        print("[eval] loading raw base pi_S (no LoRA adapter) ...")
        pi_S = LoRAPolicy(PolicyConfig(
            model_name_or_path=args.model_path,
            device=args.pi_S_device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens_system,
            attn_implementation=args.attn_implementation,
            generation_use_cache=bool(args.generation_use_cache),
            gradient_checkpointing=False,
        ), adapter_init=False)
        pi_S_adapter_path = "__raw_qwen3_4b__"
    else:
        print(f"[eval] loading pi_S from {args.pi_S_adapter} ...")
        pi_S_lora_cfg = infer_lora_config_from_adapter(args.pi_S_adapter)
        pi_S = LoRAPolicy(PolicyConfig(
            model_name_or_path=args.model_path,
            device=args.pi_S_device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens_system,
            attn_implementation=args.attn_implementation,
            generation_use_cache=bool(args.generation_use_cache),
            gradient_checkpointing=False,
            **pi_S_lora_cfg,
        ))
        pi_S.load_adapter(args.pi_S_adapter)
        pi_S_adapter_path = args.pi_S_adapter
    pi_S.eval_mode()

    pi_S_raw = None
    if int(args.system_raw_first_turns) > 0 or bool(args.system_raw_on_soft_interest):
        print(
            "[eval] loading raw base pi_S for hybrid eval "
            f"(raw_first_turns={int(args.system_raw_first_turns)}, "
            f"raw_on_soft_interest={bool(args.system_raw_on_soft_interest)}) ..."
        )
        pi_S_raw = LoRAPolicy(PolicyConfig(
            model_name_or_path=args.model_path,
            device=args.pi_S_device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens_system,
            attn_implementation=args.attn_implementation,
            generation_use_cache=bool(args.generation_use_cache),
            gradient_checkpointing=False,
        ), adapter_init=False)
        pi_S_raw.eval_mode()

    print(f"[eval] loading pi_U ({args.user_mode}) from {pi_U_adapter_path} ...")
    pi_U_lora_cfg = infer_lora_config_from_adapter(pi_U_adapter_path)
    pi_U = LoRAPolicy(PolicyConfig(
        model_name_or_path=args.model_path,
        device=args.pi_U_device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens_user,
        attn_implementation=args.attn_implementation,
        generation_use_cache=bool(args.generation_use_cache),
        gradient_checkpointing=False,
        **pi_U_lora_cfg,
    ))
    pi_U.load_adapter(pi_U_adapter_path)
    pi_U.eval_mode()

    # --------------------- rollout loop -----------------------
    metrics = DialogMetrics()
    episode_logs: List[Dict] = []
    progress_finals: List[float] = []
    rationalities: List[float] = []

    # Deterministic-ish sampling of mind-prior entries across the pool.
    # We cycle through the mind prior so each episode picks a different user.
    prior_entries = list(mind_prior.entries)
    import random
    random.Random(args.seed).shuffle(prior_entries)

    print(f"[eval] running {args.num_episodes} episodes in "
          f"user_mode={args.user_mode} ...")
    pbar = tqdm(range(args.num_episodes), desc=f"[eval/{args.user_mode}]")
    for local_i in pbar:
        i = int(args.episode_offset) + int(local_i)
        entry = prior_entries[i % len(prior_entries)]
        ep = None
        attempts = max(int(args.episode_retries), 0) + 1
        for attempt in range(attempts):
            try:
                ep = _rollout_one_episode(
                    env=env,
                    pi_S=pi_S,
                    pi_U=pi_U,
                    prior_entry=entry,
                    max_turns=args.max_turns,
                    temperature_system=args.temperature_system,
                    temperature_user=args.temperature_user,
                    top_p=args.top_p,
                    max_new_tokens_system=args.max_new_tokens_system,
                    max_new_tokens_user=args.max_new_tokens_user,
                    system_use_bdi_hint=bool(args.system_use_bdi_hint),
                    system_postprocess=bool(args.system_postprocess),
                    system_strategy_hint=strategy_hint,
                    pi_S_raw=pi_S_raw,
                    system_raw_first_turns=int(args.system_raw_first_turns),
                    system_raw_on_soft_interest=bool(args.system_raw_on_soft_interest),
                    system_scripted_policy=args.system_scripted_policy,
                    system_num_candidates=int(args.system_num_candidates),
                    system_candidate_selector=args.system_candidate_selector,
                    system_rollout_user_samples=int(args.system_rollout_user_samples),
                    system_rollout_temperature_user=float(args.system_rollout_temperature_user),
                    system_close_inject=bool(args.system_close_inject),
                    system_close_inject_min_turn=int(args.system_close_inject_min_turn),
                    system_close_inject_style=args.system_close_inject_style,
                    system_refusal_stop_after=int(args.system_refusal_stop_after),
                    task_name=args.task,
                )
                break
            except Exception as e:  # noqa: BLE001
                if attempt + 1 >= attempts:
                    print(f"[eval] episode {i} failed after {attempts} attempts: {e}")
                    break
                print(f"[eval] episode {i} failed on attempt {attempt + 1}/{attempts}: {e}; retrying")
                import time
                time.sleep(max(float(args.episode_retry_sleep), 0.0))
        if ep is None:
            continue

        metrics.add(
            success=ep["success"],
            turns=ep["turns"],
            reward=ep["cum_reward_S"],
        )
        progress_finals.append(ep["progress_final"])
        rationalities.append(ep["avg_rationality"])
        episode_logs.append({
            "idx": int(i),
            "session_id": entry.session_id,
            "method": eval_method_name,
            "dataset": args.task,
            "split": "test",
            "seed": int(args.seed),
            "user_type": eval_user_type,
            "user_group": eval_user_group,
            "max_turns": int(args.max_turns),
            **ep,
        })

        running = metrics.summary(args.max_turns)
        pbar.set_postfix({
            "SR": f"{running['SR']:.3f}",
            "AT": f"{running['AT']:.2f}",
            "prog": f"{sum(progress_finals) / max(len(progress_finals), 1):.3f}",
        })

    # --------------------- aggregate -----------------------
    summary = metrics.summary(args.max_turns)
    summary["avg_progress_final"] = float(
        sum(progress_finals) / max(len(progress_finals), 1)
    )
    summary["avg_rationality"] = float(
        sum(rationalities) / max(len(rationalities), 1)
    )
    summary["user_mode"] = args.user_mode
    summary["method"] = eval_method_name
    summary["user_type"] = eval_user_type
    summary["user_group"] = eval_user_group
    summary["system_adapter"] = args.system_adapter
    summary["system_use_bdi_hint"] = bool(args.system_use_bdi_hint)
    summary["system_postprocess"] = bool(args.system_postprocess)
    summary["system_strategy"] = args.system_strategy
    summary["system_raw_first_turns"] = int(args.system_raw_first_turns)
    summary["system_raw_on_soft_interest"] = bool(args.system_raw_on_soft_interest)
    summary["system_scripted_policy"] = args.system_scripted_policy
    summary["system_num_candidates"] = int(args.system_num_candidates)
    summary["system_candidate_selector"] = args.system_candidate_selector
    summary["system_text_selector_model"] = args.system_text_selector_model
    summary["system_stop_router_model"] = args.system_stop_router_model
    summary["system_rollout_user_samples"] = int(args.system_rollout_user_samples)
    summary["system_rollout_temperature_user"] = float(args.system_rollout_temperature_user)
    summary["system_close_inject"] = bool(args.system_close_inject)
    summary["system_close_inject_min_turn"] = int(args.system_close_inject_min_turn)
    summary["system_close_inject_style"] = args.system_close_inject_style
    summary["system_refusal_stop_after"] = int(args.system_refusal_stop_after)
    summary["pi_S_adapter"] = pi_S_adapter_path
    summary["pi_U_adapter"] = pi_U_adapter_path
    summary["num_episodes_requested"] = int(args.num_episodes)
    summary["num_episodes_completed"] = int(summary["n_episodes"])
    summary["test_mind_prior_size"] = int(len(mind_prior))
    summary["unique_episode_sessions"] = int(
        len({item["session_id"] for item in episode_logs})
    )
    summary["seed"] = int(args.seed)
    summary["task"] = args.task
    role_orientation = ""
    try:
        with open(args.test_path, "r", encoding="utf-8") as f:
            test_rows = json.load(f)
        if isinstance(test_rows, list) and test_rows:
            first = test_rows[0]
            if isinstance(first, dict):
                meta = first.get("meta") if isinstance(first.get("meta"), dict) else {}
                role_orientation = str(
                    first.get("role_orientation")
                    or meta.get("role_orientation")
                    or ""
                )
    except Exception:
        role_orientation = ""
    summary["role_orientation"] = role_orientation
    summary.update(_emotional_judge_summary(episode_logs, args.task, int(args.max_turns)))
    summary.update(_craigslist_judge_summary(episode_logs, args.task, int(args.max_turns)))
    norm_eps = normalize_episodes(
        episode_logs,
        RunMeta(
            method=eval_method_name,
            dataset=args.task,
            split="test",
            user_type=eval_user_type,
            user_group=eval_user_group,
            seed=int(args.seed),
            max_turns=int(args.max_turns),
        ),
    )
    summary["robustness_metrics"] = compute_metrics_for_method(
        norm_eps,
        T=int(args.max_turns),
        tau_q_values=(0.6, 0.7, 0.8),
    )

    ensure_dir(os.path.dirname(args.out_path) or ".")
    dump_json(summary, args.out_path)
    print("\n[eval] ===== summary =====")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if args.save_episodes:
        ep_path = os.path.splitext(args.out_path)[0] + ".episodes.json"
        dump_json(episode_logs, ep_path)
        print(f"[eval] per-episode details -> {ep_path}")


if __name__ == "__main__":
    main()
