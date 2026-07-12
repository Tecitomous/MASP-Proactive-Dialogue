#!/usr/bin/env python3
"""
Convert `src`-pipeline BDI jsonl labels into the cache format expected by the
paper-style `masp` Phase 0/1/2 pipeline.

Input:
  - dataset split json, e.g. dataset/p4g/train.json
  - src-style labels, e.g. data/labels/p4g/train.bdi.jsonl

Output:
  - data_cache/p4g_bdi_train.json

The converter preserves per-turn pre-step BDI labels and reconstructs the
mind-prior `initial_bdi` by matching the history after the first K user turns
to the closest available pre-turn snapshot.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from masp.data.bdi_dataset import BDILabelCache, BDITurnEntry
from masp.data.p4g_loader import load_p4g_sessions
from masp.mind.bdi_schema import BDI


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _history_lines(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    text = str(v or "").strip()
    if not text:
        return []
    return [x.strip() for x in text.splitlines() if x.strip()]


def _bdi_from_row(row: dict[str, Any]) -> BDI:
    """
    Pull a 6-field BDI out of a src/-style row. The src/ labeling pipeline
    (`src/labeling/oracle_bdi_label.py` + `prompts/oracle_bdi_prompt.txt`)
    already produces all 6 fields, so we keep them all.
    """
    src = row.get("silver_bdi") or row.get("teacher_state_before") or row.get("bdi") or {}
    return BDI(
        belief=str(src.get("belief", "The user is uncertain.")),
        desire=str(src.get("desire", "The user wants to avoid loss.")),
        intention=str(src.get("intention", "The user has not committed.")),
        # 6-dim scalars (paper §3.2). BDI.from_dict-style clipping is applied
        # by BDI's tolerant constructor via the dataclass fields' defaults if
        # missing; here we pass values explicitly when present.
        receptivity=float(src.get("receptivity", 0.5)),
        confidence=float(src.get("confidence", 0.5)),
        valence=float(src.get("valence", 0.0)),
    )


def _pick_initial_bdi(
    sid: str,
    target_history: list[str],
    by_history: dict[tuple[str, tuple[str, ...]], BDI],
    fallback_entries: list[BDITurnEntry],
) -> BDI:
    exact = by_history.get((sid, tuple(target_history)))
    if exact is not None:
        return exact

    # If the exact hook point does not exist (for example the K-th user turn is
    # the final utterance), fall back to the closest earlier snapshot whose
    # history is a prefix of the target history.
    best_len = -1
    best_bdi: BDI | None = None
    target_tuple = tuple(target_history)
    for (hist_sid, hist), bdi in by_history.items():
        if hist_sid != sid:
            continue
        if len(hist) > len(target_tuple):
            continue
        if target_tuple[: len(hist)] != hist:
            continue
        if len(hist) > best_len:
            best_len = len(hist)
            best_bdi = bdi
    if best_bdi is not None:
        return best_bdi

    if fallback_entries:
        return fallback_entries[0].bdi

    return BDI(
        belief="The user is uncertain.",
        desire="The user wants to avoid loss.",
        intention="The user has not committed.",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="dataset/p4g/{split}.json")
    ap.add_argument("--src_bdi_jsonl", required=True, help="src-style {split}.bdi.jsonl")
    ap.add_argument("--out_path", required=True, help="masp cache json output path")
    ap.add_argument("--task", default="p4g")
    ap.add_argument("--initial_bdi_from_first_k_turns", type=int, default=2)
    args = ap.parse_args()

    sessions = load_p4g_sessions(args.data_path)
    src_rows = _read_jsonl(args.src_bdi_jsonl)

    by_sid_turn: dict[tuple[str, int], dict[str, Any]] = {}
    by_history: dict[tuple[str, tuple[str, ...]], BDI] = {}
    for row in src_rows:
        sid = str(row.get("dialogue_id", ""))
        turn_id = int(row.get("turn_id", 0))
        hist = _history_lines(row.get("history_lines", row.get("history", "")))
        bdi = _bdi_from_row(row)
        by_sid_turn[(sid, turn_id)] = {"history": hist, "bdi": bdi}
        by_history[(sid, tuple(hist))] = bdi

    cache = BDILabelCache(task=args.task)
    missing_turns = 0
    exact_initial = 0
    fallback_initial = 0

    for session in sessions:
        sid = session.session_id
        history: list[str] = []
        per_session_entries: list[BDITurnEntry] = []

        for turn_idx, turn in enumerate(session.turns):
            row = by_sid_turn.get((sid, turn_idx))
            if row is None:
                missing_turns += 1
                bdi = BDI(
                    belief="The user is uncertain.",
                    desire="The user wants to avoid loss.",
                    intention="The user has not committed.",
                )
                hist = list(history)
            else:
                bdi = row["bdi"]
                hist = row["history"] or list(history)

            last_speaker = "none"
            if hist:
                last_line = hist[-1]
                last_speaker = "assistant" if last_line.startswith("Assistant:") else "user"

            entry = BDITurnEntry(
                session_id=sid,
                turn_idx=int(turn_idx),
                history_upto=list(hist),
                bdi=bdi,
                last_speaker=last_speaker,
                next_speaker=turn.speaker,
            )
            cache.entries.append(entry)
            per_session_entries.append(entry)

            prefix = "Assistant" if turn.speaker == "assistant" else "User"
            history.append(f"{prefix}: {turn.text}")

        collected_user = 0
        init_history: list[str] = []
        for turn in session.turns:
            prefix = "Assistant" if turn.speaker == "assistant" else "User"
            init_history.append(f"{prefix}: {turn.text}")
            if turn.speaker == "user":
                collected_user += 1
                if collected_user >= args.initial_bdi_from_first_k_turns:
                    break

        init_bdi = _pick_initial_bdi(sid, init_history, by_history, per_session_entries)
        if by_history.get((sid, tuple(init_history))) is not None:
            exact_initial += 1
        else:
            fallback_initial += 1

        cache.initial_bdi[sid] = init_bdi
        # The original paper-style extractor stores an empty profile_text by
        # default, so we keep the same behavior for compatibility.
        cache.profile_text[sid] = ""

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    cache.save(args.out_path)
    print(
        json.dumps(
            {
                "sessions": len(cache.initial_bdi),
                "turn_entries": len(cache.entries),
                "missing_turn_rows": missing_turns,
                "exact_initial_bdi": exact_initial,
                "fallback_initial_bdi": fallback_initial,
                "out_path": args.out_path,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
