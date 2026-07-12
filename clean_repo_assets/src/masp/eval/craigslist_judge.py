"""PPDPP/DialogXpert-aligned critic for Craigslist Bargain negotiations."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .p4g_judge import RewardOutput
from ..utils.llm_client import LLMClient


_DEAL_RE = re.compile(r"\b(deal|agreed|agreement|sold|works|sounds good|that works|ok(?:ay)?|accept)\b", re.I)
_NO_DEAL_RE = re.compile(r"\b(no deal|not (?:a )?deal|can't|cannot|won't|too low|too high|walk away|not interested)\b", re.I)
_PRICE_RE = re.compile(r"(?<![\w.])\$?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)(?!\w)")
_JUDGE_CANDIDATE_RE = re.compile(
    r"\b("
    r"deal|agreed|agreement|sold|sounds good|that works|works for me|"
    r"works for us|accept(?:ed)?|i accept|you accept|let'?s do|"
    r"we can do|i can do|i'?ll take|i will take|i'?ll buy|i will buy|"
    r"you got a deal"
    r")\b",
    re.I,
)


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None


def _case_prices(case_meta: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    meta = case_meta or {}
    buyer_target = _to_float(meta.get("buyer_target") or meta.get("buyer_price"))
    seller_target = _to_float(meta.get("seller_target") or meta.get("seller_price"))
    if buyer_target is not None and seller_target is not None:
        return buyer_target, seller_target
    agent_info = meta.get("agent_info") or []
    if isinstance(agent_info, list):
        for item in agent_info:
            if not isinstance(item, dict):
                continue
            role = str(item.get("Role", item.get("role", ""))).lower()
            target = _to_float(item.get("Target", item.get("target")))
            if target is None:
                continue
            if role == "buyer":
                buyer_target = target
            elif role == "seller":
                seller_target = target
    return buyer_target, seller_target


def _extract_price(text: str) -> Optional[float]:
    values = [_to_float(m.group(1)) for m in _PRICE_RE.finditer(text or "")]
    values = [v for v in values if v is not None]
    return values[-1] if values else None


def _buyer_system_history_lines(history_lines: List[str]) -> List[str]:
    converted: List[str] = []
    for line in history_lines:
        text = str(line)
        if text.startswith("Assistant:"):
            converted.append("Buyer:" + text[len("Assistant:"):])
        elif text.startswith("User:"):
            converted.append("Seller:" + text[len("User:"):])
        else:
            converted.append(text)
    return converted


def _normalize_judgment(text: str) -> Tuple[str, Optional[float]]:
    t = (text or "").strip()
    low = t.lower()
    if "have not reached a deal" in low or "not reached a deal" in low:
        return "no_deal", None
    if "have reached a deal" in low or "reached a deal" in low:
        return "deal", _extract_price(t)
    if _NO_DEAL_RE.search(t) and not _DEAL_RE.search(t):
        return "no_deal", None
    if _DEAL_RE.search(t):
        return "deal", _extract_price(t)
    return "unknown", _extract_price(t)


def _buyer_utility(deal_price: Optional[float], case_meta: Optional[Dict[str, Any]]) -> float:
    if deal_price is None:
        return 0.0
    buyer_target, seller_target = _case_prices(case_meta)
    if buyer_target is None or seller_target is None or buyer_target == seller_target:
        return 0.0
    return float((float(deal_price) - seller_target) / (buyer_target - seller_target))


def _needs_llm_judgment(
    history_lines: List[str],
    assistant_action: str,
    user_reply: str,
) -> bool:
    tail_lines = (_buyer_system_history_lines(history_lines) + [f"Buyer: {assistant_action}", f"Seller: {user_reply}"])[-4:]
    tail = "\n".join(tail_lines)
    if _JUDGE_CANDIDATE_RE.search(tail):
        return True
    if re.search(r"\bok(?:ay)?\b", tail, re.I) and _extract_price(tail) is not None:
        return True
    return False


def _aggregate(
    judgments: List[str],
    success_threshold: float,
    case_meta: Optional[Dict[str, Any]],
) -> RewardOutput:
    labels: List[str] = []
    prices: List[float] = []
    for item in judgments:
        label, price = _normalize_judgment(item)
        if label != "unknown":
            labels.append(label)
        if label == "deal" and price is not None:
            prices.append(float(price))

    deal_votes = labels.count("deal")
    no_deal_votes = labels.count("no_deal")
    # PPDPP/DialogXpert CB treats any "have not reached a deal" sample as a
    # no-deal veto, then uses the modal parsed deal price when no veto appears.
    if no_deal_votes:
        reward = -0.1
        label = "no_deal"
        success = False
    else:
        deal_price = max(set(prices), key=prices.count) if prices else None
        reward = _buyer_utility(deal_price, case_meta) if deal_price is not None else 0.0
        label = "deal" if deal_votes else "unknown"
        success = bool(reward >= float(success_threshold))
    return RewardOutput(
        reward=float(reward),
        success=success,
        label=label,
        raw_judgments=list(judgments),
        pos_hits=int(deal_votes),
        neg_hits=int(no_deal_votes),
    )


def score_craigslist_heuristic(
    history_lines: List[str],
    assistant_action: str,
    user_reply: str,
    case_meta: Optional[Dict[str, Any]] = None,
    success_threshold: float = 0.0,
) -> RewardOutput:
    role_history = _buyer_system_history_lines(history_lines)
    conv = "\n".join(role_history + [f"Buyer: {assistant_action}", f"Seller: {user_reply}"])
    tail = "\n".join((role_history + [f"Buyer: {assistant_action}", f"Seller: {user_reply}"])[-4:])
    has_deal = bool(_DEAL_RE.search(tail)) and not bool(_NO_DEAL_RE.search(user_reply or ""))
    if not has_deal:
        out = RewardOutput(
            reward=-0.1,
            success=False,
            label="no_deal",
            raw_judgments=["craigslist_heuristic:They have not reached a deal."],
            neg_hits=1,
        )
        return out
    price = _extract_price(tail) or _extract_price(conv)
    reward = _buyer_utility(price, case_meta)
    return RewardOutput(
        reward=float(reward),
        success=bool(reward >= float(success_threshold)),
        label="deal",
        raw_judgments=[f"craigslist_heuristic:They have reached a deal at {price if price is not None else 'unknown'}."],
        pos_hits=1,
    )


def _build_prompt(
    history_lines: List[str],
    assistant_action: str,
    user_reply: str,
    max_history_lines: int,
) -> str:
    role_history = _buyer_system_history_lines(history_lines)
    use_lines = role_history[-max_history_lines:] if max_history_lines > 0 else role_history
    conv = "\n".join(use_lines)
    return (
        "You are an expert in evaluating negotiations between a buyer and a seller. "
        "Your job is to determine if they have successfully reached a deal at the "
        "end of the conversation.\n\n"
        "Conversation so far:\n"
        f"{conv}\n"
        f"Buyer: {assistant_action}\n"
        f"Seller: {user_reply}\n\n"
        "Please determine whether the buyer and the seller have successfully "
        "reached a deal at the end of the conversation. If they have reached a "
        "deal, extract the final agreed price as a number and reply with this "
        "exact format: They have reached a deal at [price]. Otherwise, reply "
        "with exactly: They have not reached a deal."
    )


class DialogXpertCraigslistJudge:
    """LLM-backed Craigslist Bargain judge following PPDPP/DialogXpert CB."""

    def __init__(
        self,
        llm: LLMClient,
        success_threshold: float = 0.0,
        num_samples: int = 5,
        temperature: float = 1.0,
        max_tokens: int = 32,
        max_history_lines: int = 20,
    ):
        self.llm = llm
        self.success_threshold = float(success_threshold)
        self.num_samples = int(num_samples)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.max_history_lines = int(max_history_lines)

    def score(
        self,
        history_lines: List[str],
        assistant_action: str,
        user_reply: str,
        case_meta: Optional[Dict[str, Any]] = None,
    ) -> RewardOutput:
        if not _needs_llm_judgment(history_lines, assistant_action, user_reply):
            return RewardOutput(
                reward=-0.1,
                success=False,
                label="no_deal",
                raw_judgments=["craigslist_fast_path:They have not reached a deal."],
                neg_hits=1,
            )
        backend = str(getattr(getattr(self.llm, "cfg", None), "backend", "")).lower()
        if backend in {"heuristic", "none", "local_heuristic"}:
            return score_craigslist_heuristic(
                history_lines=history_lines,
                assistant_action=assistant_action,
                user_reply=user_reply,
                case_meta=case_meta,
                success_threshold=self.success_threshold,
            )
        prompt = _build_prompt(
            history_lines=history_lines,
            assistant_action=assistant_action,
            user_reply=user_reply,
            max_history_lines=self.max_history_lines,
        )
        try:
            judgments = self.llm.call_many(
                prompt,
                n=self.num_samples,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return _aggregate(
                judgments,
                success_threshold=self.success_threshold,
                case_meta=case_meta,
            )
        except Exception as exc:  # noqa: BLE001
            fallback = score_craigslist_heuristic(
                history_lines=history_lines,
                assistant_action=assistant_action,
                user_reply=user_reply,
                case_meta=case_meta,
                success_threshold=self.success_threshold,
            )
            fallback.raw_judgments = [f"craigslist_judge_fallback:{type(exc).__name__}:{exc}"]
            return fallback
