"""
LoRAPolicy — a thin wrapper around a compatible causal LM + a LoRA adapter.

Used for BOTH the system policy `π_S` and the adversarial user policy `π_U`.
The two policies differ only in:
  * which LoRA adapter is loaded
  * which chat prompt is built
  * which reference copy is used for KL

Design
------
We use HF + PEFT. The base model is loaded once and LoRA-injected. For
efficient self-play we can load the same base on multiple devices (one for
actor, one for reference) or share weights on a single device with two LoRA
adapter sets via `set_adapter`. For simplicity this class owns its own base
model instance so `π_S` and `π_U` can live on different GPUs.

Training
--------
The policy exposes two core operations:

  1. `generate(prompt)` — used at rollout time (no grad).
  2. `log_prob(prompt, target_ids)` — used by PPO / REINFORCE to compute
     per-token log-probs of a previously generated response (with grad).

PPO updates are handled by `masp.rl.ppo`.

NOTE: At train time we pass `gradient_checkpointing=True` to keep memory
under control. On A100 80GB this is optional for a compatible causal LM but still
recommended when 4 copies coexist during self-play.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.dtype import pick_dtype


# ---------------------------------------------------------------------- config

@dataclass
class PolicyConfig:
    model_name_or_path: str
    device: str = "cuda:0"
    dtype: str = "bf16"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    max_prompt_len: int = 1024
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    no_repeat_ngram_size: int = 4
    gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    attn_implementation: str = "flash_attention_2"
    generation_use_cache: bool = True


def infer_lora_config_from_adapter(
    adapter_path: str,
    *,
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
) -> Dict[str, object]:
    """Return PolicyConfig LoRA kwargs from a PEFT adapter directory.

    Phase 1 may save adapters with a non-default rank (for example r=64).
    Phase 2 and eval must instantiate LoRA layers with the same rank and
    target modules before loading adapter weights, otherwise PEFT raises
    shape mismatches while loading lora_A/lora_B tensors.
    """
    import json
    import os

    adapter_dir = adapter_path if os.path.isdir(adapter_path) else os.path.dirname(adapter_path)
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    raw: Dict[str, object] = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

    kwargs: Dict[str, object] = {
        "lora_r": int(lora_r if lora_r is not None else raw.get("r", 16)),
        "lora_alpha": int(lora_alpha if lora_alpha is not None else raw.get("lora_alpha", 32)),
        "lora_dropout": float(
            lora_dropout if lora_dropout is not None else raw.get("lora_dropout", 0.05)
        ),
    }
    target_modules = raw.get("target_modules")
    if target_modules:
        kwargs["lora_target_modules"] = target_modules
    return kwargs


# ---------------------------------------------------------- chat prompt helpers

def _format_history_block(history_lines: Sequence[str]) -> str:
    return "\n".join(history_lines) if history_lines else "(beginning of conversation)"


SYSTEM_PROMPT_ASSISTANT = (
    "You are the Assistant in a donation conversation. Your goal is to "
    "persuade the user to donate to the charity 'Save the Children'. "
    "Be respectful, grounded, and concise. Reply with ONE natural "
    "assistant utterance only — do not repeat earlier turns. Keep the "
    "conversation anchored on Save the Children. When the user shows "
    "interest, or after giving one brief reason, ask explicitly whether "
    "they will donate a non-zero amount from their task payment; do not "
    "wait until the final turn to close. Never assign a donation amount "
    "without the user's clear consent, and if the user refuses or chooses "
    "$0, acknowledge it respectfully instead of asking again. Do not shame, "
    "insult, or accuse the user. Do not fabricate URLs, tax details, or "
    "guarantees about how funds will be used, and do not output placeholder "
    "tokens such as URL. Do not end with a generic goodbye before getting a "
    "clear donation decision unless the user has clearly refused. Do not "
    "mention hidden task rules, turn counts, or needing more turns."
)

SYSTEM_PROMPT_USER = (
    "You are role-playing the User (persuadee) in a donation conversation. "
    "You have an internal mental state described by a Belief, a Desire, and "
    "an Intention. You MUST act faithfully to that mental state: only update "
    "your stance when the assistant gives you a genuinely compelling reason. "
    "Reply with ONE natural user utterance only — do not break character, "
    "do not narrate, do not repeat earlier turns."
)

SYSTEM_PROMPT_ESCONV_ASSISTANT = (
    "You are the Supporter in an emotional support conversation. Your goal is "
    "to help the help-seeker feel understood, less overwhelmed, and able to "
    "try one concrete coping step. Be warm, specific, and concise. Reply with "
    "ONE natural supporter utterance only. Validate the user's emotion before "
    "giving advice; ask at most one gentle question unless the next practical "
    "step is already clear. Do not mention donations, charities, task rules, "
    "hidden mental states, turn counts, or that you are an AI. Do not diagnose, "
    "promise a cure, minimize the user's feelings, or pressure them. If the "
    "user implies immediate danger or self-harm, encourage contacting emergency "
    "services or a trusted person right away."
)

SYSTEM_PROMPT_ESCONV_USER = (
    "You are role-playing the Help-seeker in an emotional support conversation. "
    "You have an internal mental state described by a Belief, a Desire, and an "
    "Intention. Act faithfully to that mental state: you may become more open "
    "when the supporter validates your feelings or offers a realistic coping "
    "step, and you may push back when the support feels generic or dismissive. "
    "Reply with ONE natural help-seeker utterance only; do not narrate, break "
    "character, or repeat earlier turns."
)

SYSTEM_PROMPT_EMPATHETIC_ASSISTANT = (
    "You are the Listener in an empathetic dialogue. Your goal is to help the "
    "speaker feel heard, understood, and emotionally connected. Be warm, "
    "specific, and concise. Reply with ONE natural listener utterance only. "
    "Acknowledge the exact emotion or experience the speaker shared, then ask "
    "at most one gentle follow-up or offer one brief validating reflection. "
    "Do not mention donations, charities, hidden mental states, task rules, "
    "turn counts, or that you are an AI. Do not give generic advice, diagnose, "
    "minimize the user's feelings, or change the topic."
)

SYSTEM_PROMPT_EMPATHETIC_USER = (
    "You are role-playing the Speaker in an empathetic dialogue. You have an "
    "internal mental state described by a Belief, a Desire, and an Intention. "
    "Act faithfully to that mental state: you may open up, elaborate, or feel "
    "validated when the listener accurately acknowledges your emotion, and you "
    "may stay brief or push back when the response feels generic or dismissive. "
    "Reply with ONE natural speaker utterance only; do not narrate, break "
    "character, or repeat earlier turns."
)

SYSTEM_PROMPT_CRAIGSLIST_ASSISTANT = (
    "You are the Buyer in a Craigslist price bargaining conversation. Your "
    "goal is to buy the listed item at a reasonable low price while keeping "
    "the seller engaged. Be concise, natural, and transactional. Reply with "
    "ONE buyer utterance only. You may ask item questions, explain your "
    "budget, make a counteroffer, accept a fair price, or politely hold firm. Do not "
    "mention donations, charities, hidden mental states, task rules, turn "
    "counts, or that you are an AI. Do not fabricate external links or details "
    "not already implied by the conversation."
)

SYSTEM_PROMPT_CRAIGSLIST_USER = (
    "You are role-playing the Seller in a Craigslist price bargaining "
    "conversation. You have an internal mental state described by a Belief, a "
    "Desire, and an Intention. Act faithfully to that mental state: answer "
    "reasonable item questions, negotiate toward your target price, concede "
    "only when the buyer gives a credible reason, and accept only when the "
    "deal fits your constraints. Reply with ONE natural seller utterance only; do "
    "not narrate, break character, or repeat earlier turns."
)


def _task_key(task_name: str) -> str:
    return (task_name or "p4g").lower().strip()


def build_chat_prompt_for_policy(
    role: str,
    history_lines: Sequence[str],
    bdi_text: Optional[str] = None,
    belief_hint_text: Optional[str] = None,
    task_name: str = "p4g",
    assistant_strategy_hint: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build a chat-formatted message list for either the system or user policy.

    For the system policy we optionally inject the *inferred* BDI (ẑ) as a
    belief hint; this lets the policy consume mentalization output through
    natural language, which works across any HF chat model.

    For the user policy we inject the *true* sampled BDI (z*) so it can act
    faithfully to it.
    """
    task = _task_key(task_name)
    if role == "assistant":
        assistant_turn = sum(1 for line in history_lines if str(line).startswith("Assistant:")) + 1
        if task == "esconv":
            sys = SYSTEM_PROMPT_ESCONV_ASSISTANT
        elif task == "empathetic_dialogues":
            sys = SYSTEM_PROMPT_EMPATHETIC_ASSISTANT
        elif task == "craigslist_bargain":
            sys = SYSTEM_PROMPT_CRAIGSLIST_ASSISTANT
        else:
            sys = SYSTEM_PROMPT_ASSISTANT
        if belief_hint_text:
            sys += (
                "\n\nYour mental model of the user (do not repeat it to the "
                f"user):\n{belief_hint_text}"
            )
        if task == "esconv":
            closing_hint = (
                "Respond as an emotional supporter: reflect the user's specific "
                "feeling, then either ask one gentle follow-up or offer one "
                "concrete coping step that fits the situation."
            )
            if assistant_turn >= 6:
                closing_hint += (
                " This is a late turn, so help the user leave with one "
                "manageable next step and check whether it feels doable."
                )
        elif task == "empathetic_dialogues":
            closing_hint = (
                "Respond as an empathetic listener: reflect the speaker's "
                "specific feeling or situation, show that you understand why "
                "it matters, then ask one gentle follow-up if more sharing "
                "would help. Keep it natural and avoid advice unless the "
                "speaker explicitly asks for it."
            )
            if assistant_turn >= 6:
                closing_hint += (
                    " This is a late turn, so offer a concise validating "
                    "reflection that helps the speaker feel heard and gives "
                    "them room to close the conversation warmly."
                )
        elif task == "craigslist_bargain":
            closing_hint = (
                "Respond as the buyer: keep the negotiation moving with a "
                "specific question, counteroffer, acceptance, or firm but "
                "polite budget boundary. If the seller makes a concrete "
                "acceptable price, confirm the deal and the price. If the ask "
                "is too high, give one brief reason and a realistic lower "
                "counteroffer."
            )
            if assistant_turn >= 6:
                closing_hint += (
                    " This is a late turn, so try to close with a clear price "
                    "or a clear no-deal boundary."
                )
        else:
            closing_hint = (
                "If the user asks a factual question, answer it in one short "
                "sentence and then ask for a clear donation decision or a specific "
                "non-zero amount. If the user has already refused or chosen $0, "
                "do not pressure them or assign any amount; politely acknowledge "
                "their decision. If the user is positive, curious, or says they "
                "will look into it, do not say goodbye yet; ask what amount they "
                "will donate now from the task payment."
            )
            if assistant_turn >= 6:
                closing_hint += (
                    " This is a late turn, so do not open a new topic or offer more "
                    "background; close now."
                )
            if assistant_strategy_hint:
                closing_hint += f"\n{assistant_strategy_hint.strip()}"
        user_content = (
            "Conversation so far:\n"
            f"{_format_history_block(history_lines)}\n\n"
            f"Next assistant turn number: {assistant_turn} of at most 8.\n"
            f"{closing_hint}\n\n"
            "Write the next assistant turn."
        )
    elif role == "user":
        if task == "esconv":
            sys = SYSTEM_PROMPT_ESCONV_USER
        elif task == "empathetic_dialogues":
            sys = SYSTEM_PROMPT_EMPATHETIC_USER
        elif task == "craigslist_bargain":
            sys = SYSTEM_PROMPT_CRAIGSLIST_USER
        else:
            sys = SYSTEM_PROMPT_USER
        if bdi_text:
            sys += f"\n\nYour current internal mental state:\n{bdi_text}"
        user_content = (
            "Conversation so far:\n"
            f"{_format_history_block(history_lines)}\n\n"
            "Write the next user turn, staying in character."
        )
    else:
        raise ValueError(f"unknown role: {role}")
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------- policy

class LoRAPolicy(nn.Module):
    """
    A a compatible causal LM + LoRA policy that can generate, compute log-probs, and be
    cloned into a frozen reference for KL regularization.
    """

    def __init__(self, cfg: PolicyConfig, adapter_init: bool = True):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = pick_dtype(cfg.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=cfg.trust_remote_code,
            attn_implementation=cfg.attn_implementation,
        )
        if cfg.gradient_checkpointing:
            base.gradient_checkpointing_enable()
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()
            if hasattr(base, "config"):
                base.config.use_cache = False
        elif hasattr(base, "config"):
            base.config.use_cache = bool(cfg.generation_use_cache)

        if adapter_init:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=cfg.lora_target_modules,
            )
            self.model = get_peft_model(base, lora_cfg)
        else:
            self.model = base

        self.model.to(self.device)

    # -------------------------------------------------------------- utilities
    def _activate_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def save_adapter(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)

    def load_adapter(self, path: str) -> None:
        """
        Load LoRA weights from a directory previously written by
        `save_adapter`. We deliberately bypass `PeftModel.load_adapter`
        (which is fussy about adapter-name slot collisions when the model
        was constructed via `get_peft_model(...)` and already owns a
        "default" adapter) and instead load the raw weights via
        `set_peft_model_state_dict`. This works whether the on-disk file
        is `adapter_model.safetensors` or `adapter_model.bin`.
        """
        import os
        try:
            from peft import set_peft_model_state_dict
        except ImportError as e:
            raise RuntimeError("PEFT is required to load LoRA adapters") from e

        st_path = os.path.join(path, "adapter_model.safetensors")
        bin_path = os.path.join(path, "adapter_model.bin")
        if os.path.isfile(st_path):
            try:
                from safetensors.torch import load_file
                sd = load_file(st_path, device="cpu")
            except Exception:
                # Fall back to torch.load if safetensors is unavailable.
                sd = torch.load(st_path, map_location="cpu")
        elif os.path.isfile(bin_path):
            sd = torch.load(bin_path, map_location="cpu")
        elif os.path.isfile(path):
            # Caller passed a single weight file directly.
            sd = torch.load(path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"no adapter weights found under {path} "
                "(expected adapter_model.safetensors or adapter_model.bin)"
            )

        # set_peft_model_state_dict will silently ignore non-LoRA keys, so
        # the call is safe even if the saved bundle includes extras.
        set_peft_model_state_dict(self.model, sd, adapter_name="default")

    def eval_mode(self) -> None:
        self.model.eval()

    def train_mode(self) -> None:
        self.model.train()

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Toggle checkpointing around PPO updates.

        Rollout generation is fastest with KV cache enabled and checkpointing off,
        but PPO backward on 40GB cards can OOM. This lets the trainer enable
        checkpointing only for the backward pass, then restore fast generation.
        """
        enabled = bool(enabled)
        if enabled:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            base = getattr(self.model, "base_model", self.model)
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
            if hasattr(base, "config"):
                base.config.use_cache = False
        else:
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()
            base = getattr(self.model, "base_model", self.model)
            if hasattr(base, "gradient_checkpointing_disable"):
                base.gradient_checkpointing_disable()
            use_cache = bool(self.cfg.generation_use_cache)
            if hasattr(self.model, "config"):
                self.model.config.use_cache = use_cache
            if hasattr(base, "config"):
                base.config.use_cache = use_cache
        self.cfg.gradient_checkpointing = enabled

    # ------------------------------------------------------ prompt -> tokens
    def _apply_chat(self, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
        """Apply the tokenizer chat template.

        For Qwen3 instruct models we force `enable_thinking=False` —
        dialogue training must stay in non-thinking mode so no
        ``<think>...</think>`` content leaks into rollouts. For chat
        templates that don't expose the kwarg (Qwen3 base / non-Qwen3) we
        fall back transparently.
        """
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        try:
            return self.tokenizer.apply_chat_template(
                messages, **kwargs, enable_thinking=False,
            )
        except TypeError:
            # Template doesn't accept enable_thinking — try without it.
            try:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
            except Exception:
                pass
        except Exception:
            pass
        # Final fallback: naive concatenation.
        parts = []
        for m in messages:
            parts.append(f"<{m['role']}>\n{m['content']}\n</{m['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>\n")
        return "\n".join(parts)

    def _tokenize_prompt(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(
            text,
            truncation=True,
            max_length=self.cfg.max_prompt_len,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        return ids.to(self.device)

    def _tokenize_prompt_batch(
        self,
        texts: Sequence[str],
        padding_side: str = "left",
    ) -> Dict[str, torch.Tensor]:
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        try:
            enc = self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.cfg.max_prompt_len,
                return_tensors="pt",
                add_special_tokens=False,
            )
        finally:
            self.tokenizer.padding_side = old_padding_side
        return {k: v.to(self.device) for k, v in enc.items()}

    # ------------------------------------------------------------------ generate
    @torch.no_grad()
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        self._activate_device()
        prompt_text = self._apply_chat(messages, add_generation_prompt=True)
        prompt_ids = self._tokenize_prompt(prompt_text)
        attn = torch.ones_like(prompt_ids)
        temp = float(temperature if temperature is not None else self.cfg.temperature)
        sample = bool(do_sample) and temp > 0.0

        gen_kwargs = {
            "input_ids": prompt_ids,
            "attention_mask": attn,
            "max_new_tokens": int(max_new_tokens or self.cfg.max_new_tokens),
            "do_sample": sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": False,
            "use_cache": bool(self.cfg.generation_use_cache),
        }
        if float(self.cfg.repetition_penalty) > 1.0:
            gen_kwargs["repetition_penalty"] = float(self.cfg.repetition_penalty)
        if int(self.cfg.no_repeat_ngram_size) > 0:
            gen_kwargs["no_repeat_ngram_size"] = int(self.cfg.no_repeat_ngram_size)
        if sample:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = float(top_p if top_p is not None else self.cfg.top_p)

        out = self.model.generate(**gen_kwargs)
        seq = out.sequences[0]
        new_ids = seq[prompt_ids.shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return {
            "text": text,
            "prompt_ids": prompt_ids[0].detach(),
            "response_ids": new_ids.detach(),
        }

    @torch.no_grad()
    def generate_batch(
        self,
        messages_batch: Sequence[List[Dict[str, str]]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        self._activate_device()
        if not messages_batch:
            return []
        prompt_texts = [
            self._apply_chat(messages, add_generation_prompt=True)
            for messages in messages_batch
        ]
        enc = self._tokenize_prompt_batch(prompt_texts, padding_side="left")
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        temp = float(temperature if temperature is not None else self.cfg.temperature)
        sample = bool(do_sample) and temp > 0.0

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "max_new_tokens": int(max_new_tokens or self.cfg.max_new_tokens),
            "do_sample": sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": False,
            "use_cache": bool(self.cfg.generation_use_cache),
        }
        if float(self.cfg.repetition_penalty) > 1.0:
            gen_kwargs["repetition_penalty"] = float(self.cfg.repetition_penalty)
        if int(self.cfg.no_repeat_ngram_size) > 0:
            gen_kwargs["no_repeat_ngram_size"] = int(self.cfg.no_repeat_ngram_size)
        if sample:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = float(top_p if top_p is not None else self.cfg.top_p)

        out = self.model.generate(**gen_kwargs)
        prompt_width = int(input_ids.shape[1])
        eos_id = self.tokenizer.eos_token_id
        results: List[Dict[str, torch.Tensor]] = []
        for row in range(out.sequences.shape[0]):
            seq = out.sequences[row]
            new_ids = seq[prompt_width:].detach()
            if eos_id is not None:
                eos_pos = (new_ids == eos_id).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    new_ids = new_ids[: int(eos_pos[0].item()) + 1]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            prompt_ids = input_ids[row][attn[row].bool()].detach()
            results.append({
                "text": text,
                "prompt_ids": prompt_ids,
                "response_ids": new_ids,
            })
        return results

    # --------------------------------------------------------- log prob (train)
    def log_probs_of_response(
        self,
        messages: List[Dict[str, str]],
        response_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token log-probs of `response_ids` given the chat prompt.

        Returns:
            logp: (T,) tensor of per-token log-probs (with grad)
            mask: (T,) bool — always True here but reserved for padding
        """
        self._activate_device()
        prompt_text = self._apply_chat(messages, add_generation_prompt=True)
        prompt_ids = self._tokenize_prompt(prompt_text)[0]
        response_ids = response_ids.to(self.device).long()

        full = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0)
        attn = torch.ones_like(full)

        out = self.model(input_ids=full, attention_mask=attn, return_dict=True, use_cache=False)
        logits = out.logits  # (1, L, V)

        # For token t in the full sequence, logits at position t-1 predict it.
        # We only score positions that fall in the response segment.
        shift_logits = logits[0, :-1, :]             # (L-1, V)
        shift_tokens = full[0, 1:]                   # (L-1,)
        log_probs_all = F.log_softmax(shift_logits.float(), dim=-1)  # (L-1, V)

        # Response corresponds to positions [prompt_len - 1 .. L - 2] in
        # shift_logits, which predict tokens at positions [prompt_len .. L - 1].
        pl = int(prompt_ids.shape[0])
        resp_logp = log_probs_all[pl - 1: pl - 1 + response_ids.shape[0]].gather(
            1, response_ids.unsqueeze(-1)
        ).squeeze(-1)  # (T,)

        mask = torch.ones_like(resp_logp, dtype=torch.bool)
        return resp_logp, mask

    @torch.no_grad()
    def log_probs_of_responses_batch(
        self,
        messages_batch: Sequence[List[Dict[str, str]]],
        response_ids_batch: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        self._activate_device()
        if not messages_batch:
            return []
        if len(messages_batch) != len(response_ids_batch):
            raise ValueError("messages_batch and response_ids_batch must have the same length")

        prompt_texts = [
            self._apply_chat(messages, add_generation_prompt=True)
            for messages in messages_batch
        ]
        prompt_ids_list = [
            self._tokenize_prompt(text)[0]
            for text in prompt_texts
        ]
        response_ids_list = [resp.to(self.device).long() for resp in response_ids_batch]

        pad_id = self.tokenizer.pad_token_id
        full_lengths = [
            int(prompt_ids.shape[0] + response_ids.shape[0])
            for prompt_ids, response_ids in zip(prompt_ids_list, response_ids_list)
        ]
        max_full_len = max(full_lengths)
        full = torch.full(
            (len(prompt_ids_list), max_full_len),
            fill_value=pad_id,
            dtype=torch.long,
            device=self.device,
        )
        attn = torch.zeros_like(full)
        for row, (prompt_ids, response_ids) in enumerate(zip(prompt_ids_list, response_ids_list)):
            seq = torch.cat([prompt_ids, response_ids], dim=0)
            full[row, : seq.shape[0]] = seq
            attn[row, : seq.shape[0]] = 1

        out = self.model(input_ids=full, attention_mask=attn, return_dict=True, use_cache=False)
        logits = out.logits
        shift_logits = logits[:, :-1, :]
        log_probs_all = F.log_softmax(shift_logits.float(), dim=-1)

        results: List[torch.Tensor] = []
        for row, (prompt_ids, response_ids) in enumerate(zip(prompt_ids_list, response_ids_list)):
            pl = int(prompt_ids.shape[0])
            rl = int(response_ids.shape[0])
            resp_logp = log_probs_all[row, pl - 1: pl - 1 + rl].gather(
                1, response_ids.unsqueeze(-1)
            ).squeeze(-1)
            results.append(resp_logp.detach())
        return results

    def log_probs_of_responses_batch_train(
        self,
        messages_batch: Sequence[List[Dict[str, str]]],
        response_ids_batch: Sequence[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Batched train-mode response log-probs with gradients.

        This is used by PPO updates. Rollout still uses the no-grad batch
        method above to collect old log-probs cheaply.
        """
        if not messages_batch:
            return []
        self._activate_device()
        if len(messages_batch) != len(response_ids_batch):
            raise ValueError("messages_batch and response_ids_batch must have the same length")

        prompt_texts = [
            self._apply_chat(messages, add_generation_prompt=True)
            for messages in messages_batch
        ]
        prompt_ids_list = [
            self._tokenize_prompt(text)[0]
            for text in prompt_texts
        ]
        response_ids_list = [resp.to(self.device).long() for resp in response_ids_batch]

        pad_id = self.tokenizer.pad_token_id
        full_lengths = [
            int(prompt_ids.shape[0] + response_ids.shape[0])
            for prompt_ids, response_ids in zip(prompt_ids_list, response_ids_list)
        ]
        max_full_len = max(full_lengths)
        full = torch.full(
            (len(prompt_ids_list), max_full_len),
            fill_value=pad_id,
            dtype=torch.long,
            device=self.device,
        )
        attn = torch.zeros_like(full)
        for row, (prompt_ids, response_ids) in enumerate(zip(prompt_ids_list, response_ids_list)):
            seq = torch.cat([prompt_ids, response_ids], dim=0)
            full[row, : seq.shape[0]] = seq
            attn[row, : seq.shape[0]] = 1

        out = self.model(input_ids=full, attention_mask=attn, return_dict=True, use_cache=False)
        logits = out.logits
        shift_logits = logits[:, :-1, :]
        log_probs_all = F.log_softmax(shift_logits.float(), dim=-1)

        results: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for row, (prompt_ids, response_ids) in enumerate(zip(prompt_ids_list, response_ids_list)):
            pl = int(prompt_ids.shape[0])
            rl = int(response_ids.shape[0])
            resp_logp = log_probs_all[row, pl - 1: pl - 1 + rl].gather(
                1, response_ids.unsqueeze(-1)
            ).squeeze(-1)
            mask = torch.ones_like(resp_logp, dtype=torch.bool)
            results.append((resp_logp, mask))
        return results

    # ------------------------------------------------------- ref log prob (frozen)
    @torch.no_grad()
    def log_probs_ref(
        self,
        messages: List[Dict[str, str]],
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        logp, _ = self.log_probs_of_response(messages, response_ids)
        return logp.detach()
