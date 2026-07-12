#!/usr/bin/env python3
"""
Phase 0a — extract silver BDI labels for every turn hook point in a split.

Usage:
    python extract_bdi_labels.py \
        --data_path dataset/p4g/train.json \
        --out_path data_cache/p4g_bdi_train.json \
        --task p4g \
        --backend azure \
        --workers 8

For P4G this processes ~1000 dialogues × ~20 turn hook points = ~20k LLM
calls per split. With --workers N, N sessions are processed in parallel
(turns within a session remain sequential since each depends on prior history).

You should run this once per split (train / valid / test). The mind prior for
self-play is built from the initial_bdi entries in the train-split cache.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from masp.data.bdi_dataset import BDILabelCache, BDITurnEntry
from masp.data.dataset_adapter import get_adapter
from masp.mind.bdi_extractor import BDIExtractor
from masp.mind.bdi_schema import BDI, TASK_CONFIGS
from masp.utils.llm_client import LLMClient, LLMConfig


@dataclass
class SessionResult:
    """Result of processing one session — collected by the main thread."""
    session_id: str
    entries: List[BDITurnEntry]
    initial_bdi: BDI
    profile_text: str


def _merge_session_result(cache: BDILabelCache, result: SessionResult) -> None:
    """Merge one completed session, replacing any older copy for resume safety."""
    sid = result.session_id
    cache.entries = [e for e in cache.entries if e.session_id != sid]
    cache.entries.extend(result.entries)
    cache.initial_bdi[sid] = result.initial_bdi
    cache.profile_text[sid] = result.profile_text


def _sort_cache_by_session_order(cache: BDILabelCache, session_order: Dict[str, int]) -> None:
    """Keep output deterministic even though parallel workers finish out of order."""
    cache.entries.sort(
        key=lambda e: (session_order.get(e.session_id, 10**12), int(e.turn_idx))
    )
    cache.initial_bdi = dict(
        sorted(cache.initial_bdi.items(), key=lambda kv: session_order.get(kv[0], 10**12))
    )
    cache.profile_text = dict(
        sorted(cache.profile_text.items(), key=lambda kv: session_order.get(kv[0], 10**12))
    )


def _save_progress(cache: BDILabelCache, out_path: str, session_order: Dict[str, int]) -> None:
    _sort_cache_by_session_order(cache, session_order)
    cache.save(out_path)


def _process_one_session(
    s,
    extractor: BDIExtractor,
    initial_bdi_from_first_k_turns: int,
) -> SessionResult:
    """Process all turns of one session sequentially (thread-safe)."""
    entries: List[BDITurnEntry] = []
    history: List[str] = []

    for idx, t in enumerate(s.turns):
        bdi = extractor.extract(history)
        if history:
            prev_line = history[-1]
            last_speaker = "assistant" if prev_line.startswith("Assistant:") else "user"
        else:
            last_speaker = "none"
        entries.append(BDITurnEntry(
            session_id=s.session_id,
            turn_idx=int(idx),
            history_upto=list(history),
            bdi=bdi,
            last_speaker=last_speaker,
            next_speaker=t.speaker,
        ))
        prefix = "Assistant" if t.speaker == "assistant" else "User"
        history.append(f"{prefix}: {t.text}")

    # --- initial BDI for the mind prior ---
    init_history: List[str] = []
    collected_user = 0
    for t in s.turns:
        prefix = "Assistant" if t.speaker == "assistant" else "User"
        init_history.append(f"{prefix}: {t.text}")
        if t.speaker == "user":
            collected_user += 1
            if collected_user >= initial_bdi_from_first_k_turns:
                break
    if init_history:
        init_bdi = extractor.extract(init_history)
    else:
        init_bdi = BDI(
            belief="The user is neutral and does not know much about the charity.",
            desire="The user wants to avoid financial burden.",
            intention="The user is not currently inclined to donate.",
            receptivity=0.5,
            confidence=0.5,
            valence=0.0,
        )

    return SessionResult(
        session_id=s.session_id,
        entries=entries,
        initial_bdi=init_bdi,
        profile_text="",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--task", type=str, default="p4g", choices=list(TASK_CONFIGS.keys()))
    p.add_argument("--max_sessions", type=int, default=None)
    p.add_argument("--backend", type=str, default="azure", choices=["openai", "azure", "local"])
    p.add_argument("--model", type=str, default="")
    p.add_argument("--api_base", type=str, default="")
    p.add_argument("--api_key_env", type=str, default="")
    p.add_argument("--local_model_path", type=str, default="")
    p.add_argument("--local_device", type=str, default="cuda:0")
    p.add_argument("--local_dtype", type=str, default="bf16")
    # azure backend
    p.add_argument("--azure_endpoint", type=str, default="",
                   help="Azure OpenAI endpoint URL")
    p.add_argument("--azure_api_version", type=str, default="2024-03-01-preview")
    p.add_argument("--azure_thinking_budget", type=int, default=0,
                   help="Thinking budget for Gemini models. 0=disable (recommended for BDI extraction).")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_tokens", type=int, default=320,
                   help="Generation budget for the 6-field BDI schema "
                        "(Belief/Desire/Intention + Receptivity/"
                        "Confidence/Valence) plus headroom.")
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--retry_sleep_sec", type=float, default=1.0)
    p.add_argument("--initial_bdi_from_first_k_turns", type=int, default=2,
                   help="Use the first K user turns to define z_init for the mind prior.")
    p.add_argument("--workers", type=int, default=8,
                   help="Number of parallel session workers. Each worker "
                        "processes one session at a time (turns within a "
                        "session stay sequential). 1 = original serial mode.")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Resume from --out_path by skipping sessions already saved. Default: true.")
    p.add_argument("--save_every", type=int, default=1,
                   help="Atomically save progress every N completed sessions. Default 1 = realtime.")
    args = p.parse_args()

    task_cfg = TASK_CONFIGS[args.task]
    adapter = get_adapter(args.task)
    sessions = adapter.load_sessions(args.data_path, max_sessions=args.max_sessions)
    session_order = {s.session_id: i for i, s in enumerate(sessions)}
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    cache = BDILabelCache(task=args.task)
    completed_sessions = set()
    if args.resume and os.path.exists(args.out_path):
        cache = BDILabelCache.load(args.out_path)
        completed_sessions = set(cache.initial_bdi.keys())
        sessions = [s for s in sessions if s.session_id not in completed_sessions]
        print(
            f"resume: found {len(completed_sessions)} completed sessions in {args.out_path}; "
            f"remaining={len(sessions)}",
            flush=True,
        )
    else:
        # Truncate/initialize the output early so users can tail/inspect it.
        cache.save(args.out_path)

    print(f"loaded {len(sessions)} sessions from {args.data_path} (adapter={adapter.task_name})")

    api_key_env = args.api_key_env
    if api_key_env and not os.getenv(api_key_env) and len(api_key_env) > 20:
        # Backward-compatible convenience: users sometimes pass the actual key to
        # --api_key_env. The LLM client expects an env-var name, so materialize it.
        os.environ.setdefault("MASP_INLINE_API_KEY", api_key_env)
        api_key_env = "MASP_INLINE_API_KEY"

    llm = LLMClient(
        LLMConfig(
            backend=args.backend,
            model=args.model,
            api_base=args.api_base,
            api_key_env=api_key_env,
            local_model_path=args.local_model_path,
            local_device=args.local_device,
            local_dtype=args.local_dtype,
            max_retries=args.max_retries,
            retry_sleep_sec=args.retry_sleep_sec,
            azure_endpoint=args.azure_endpoint,
            azure_api_version=args.azure_api_version,
            azure_thinking_budget=args.azure_thinking_budget,
            verbose=True,
        )
    )

    # Each worker gets its own BDIExtractor (with its own cache dict) to
    # avoid lock contention. The LLMClient itself is thread-safe (TLS
    # connections per thread).
    def _make_extractor() -> BDIExtractor:
        return BDIExtractor(
            llm=llm,
            task_cfg=task_cfg,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    workers = max(1, args.workers)
    save_every = max(int(args.save_every), 1)
    completed_since_save = 0
    failed_sessions: List[Tuple[str, str]] = []
    failed_path = f"{args.out_path}.failed_sessions.jsonl"

    def _record_failure(session_id: str, err: Exception) -> None:
        msg = repr(err)
        failed_sessions.append((session_id, msg))
        print(f"[extract-bdi] ERROR session={session_id} failed: {msg}", flush=True)
        with open(failed_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"session_id": session_id, "error": msg},
                ensure_ascii=False,
            ) + "\n")

    if not sessions:
        _save_progress(cache, args.out_path, session_order)
        print(f"nothing to do; cache already has {len(cache.entries)} BDI entries + "
              f"{len(cache.initial_bdi)} initial BDIs at {args.out_path}")
        return

    if workers == 1:
        # --- serial mode (original behavior) ---
        extractor = _make_extractor()
        for s in tqdm(sessions, desc="BDI extract"):
            try:
                result = _process_one_session(s, extractor, args.initial_bdi_from_first_k_turns)
            except Exception as e:  # noqa: BLE001
                _record_failure(s.session_id, e)
                continue
            _merge_session_result(cache, result)
            completed_since_save += 1
            if completed_since_save >= save_every:
                _save_progress(cache, args.out_path, session_order)
                completed_since_save = 0
    else:
        # --- parallel mode ---
        # Thread-local extractors to avoid cache dict contention.
        _tls = threading.local()

        def _get_extractor() -> BDIExtractor:
            if not hasattr(_tls, "extractor"):
                _tls.extractor = _make_extractor()
            return _tls.extractor

        def _worker(s):
            ext = _get_extractor()
            return _process_one_session(s, ext, args.initial_bdi_from_first_k_turns)

        print(f"using {workers} parallel workers for {len(sessions)} sessions")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_session = {
                pool.submit(_worker, s): s for s in sessions
            }
            with tqdm(total=len(sessions), desc="BDI extract") as pbar:
                for future in as_completed(future_to_session):
                    s = future_to_session[future]
                    try:
                        result = future.result()
                    except Exception as e:  # noqa: BLE001
                        _record_failure(s.session_id, e)
                        pbar.update(1)
                        continue
                    _merge_session_result(cache, result)
                    completed_since_save += 1
                    if completed_since_save >= save_every:
                        _save_progress(cache, args.out_path, session_order)
                        completed_since_save = 0
                    pbar.update(1)

    _save_progress(cache, args.out_path, session_order)
    print(f"saved {len(cache.entries)} BDI entries + {len(cache.initial_bdi)} "
          f"initial BDIs to {args.out_path}")
    if failed_sessions:
        print(
            f"failed {len(failed_sessions)} sessions; details appended to {failed_path}. "
            "Rerun with --resume to retry them.",
            flush=True,
        )


if __name__ == "__main__":
    main()
