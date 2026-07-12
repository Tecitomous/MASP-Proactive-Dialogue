"""
Unified LLM client used by:
  1) Oracle Belief Updater (OBU) for BDI extraction and goal labeling
  2) Rationality classifier for the user-side adversarial reward
  3) DialogXpert-style P4G success critic

Backends supported:
  - "openai"  : any OpenAI-compatible HTTP endpoint
  - "azure"   : an Azure OpenAI compatible endpoint
  - "local"   : a locally loaded HF causal LM (no API calls)

The client is stateless; all calls are blocking and retried with exponential
back-off. Tokenizer / local model is loaded lazily.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional


def _pick_api_key(api_key_env: Optional[str] = None) -> str:
    if api_key_env:
        v = os.getenv(api_key_env, "").strip()
        if v:
            return v
    for key_name in ("OPENAI_API_KEY", "AZURE_API_KEY"):
        v = os.getenv(key_name, "").strip()
        if v:
            return v
    return ""


@dataclass
class LLMConfig:
    backend: str = "openai"         # openai / azure / local
    model: str = ""
    api_base: str = ""
    api_key_env: str = ""
    timeout_sec: float = 60.0
    max_retries: int = 3
    retry_sleep_sec: float = 1.0
    # local backend only
    local_model_path: str = ""
    local_device: str = "cuda:0"
    local_dtype: str = "bf16"
    local_max_new_tokens: int = 256
    # azure backend specific
    azure_endpoint: str = ""
    azure_api_version: str = "2024-03-01-preview"
    azure_thinking_budget: int = 0  # 0 = disable thinking (clean output for BDI extraction)
    parallel_workers: int = 1
    name: str = "llm"
    verbose: bool = False
    log_output_chars: int = 160


class LLMClient:
    """Very small abstraction over one chat-style LLM call."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._openai_client = None
        self._local_tok = None
        self._local_model = None
        import threading

        self._rate_lock = threading.Lock()
        self._next_call_time = 0.0
        self._thread_local = threading.local()

    def _tls(self):
        st = self._thread_local
        if not hasattr(st, "openai_client"):
            st.openai_client = None
        return st

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(f"[llm:{self.cfg.name}] {msg}", flush=True)

    def _rate_wait(self) -> None:
        interval = float(os.getenv("MASP_LLM_MIN_INTERVAL_SEC", "0") or 0.0)
        if interval <= 0:
            return
        with self._rate_lock:
            now = time.perf_counter()
            wait_s = max(0.0, self._next_call_time - now)
            self._next_call_time = max(now, self._next_call_time) + interval
        if wait_s > 0:
            time.sleep(wait_s)

    # ------------------------------------------------------------------ local
    def _lazy_local(self):
        if self._local_model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .dtype import pick_dtype

        path = self.cfg.local_model_path or self.cfg.model
        if not path:
            raise RuntimeError("local backend needs cfg.local_model_path or cfg.model")
        self._local_tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self._local_tok.pad_token is None:
            self._local_tok.pad_token = self._local_tok.eos_token
        self._local_model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=pick_dtype(self.cfg.local_dtype),
            trust_remote_code=True,
        ).to(self.cfg.local_device)
        self._local_model.eval()

    def _call_local(self, prompt: str, temperature: float, max_tokens: int) -> str:
        import torch

        self._lazy_local()
        messages = [{"role": "user", "content": prompt}]
        text = self._local_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self._local_tok(text, return_tensors="pt").to(self.cfg.local_device)
        with torch.no_grad():
            out = self._local_model.generate(
                **enc,
                max_new_tokens=int(max_tokens),
                do_sample=temperature > 0,
                temperature=max(float(temperature), 1e-5),
                top_p=0.95,
                pad_token_id=self._local_tok.pad_token_id,
            )
        new_ids = out[0, enc["input_ids"].shape[1]:]
        return self._local_tok.decode(new_ids, skip_special_tokens=True).strip()

    # ----------------------------------------------------------------- openai
    def _lazy_openai(self):
        st = self._tls()
        if st.openai_client is not None:
            return st.openai_client
        from openai import OpenAI  # type: ignore

        api_key = _pick_api_key(self.cfg.api_key_env)
        if not api_key:
            raise RuntimeError("missing API key for openai backend")
        kwargs = {"api_key": api_key, "timeout": self.cfg.timeout_sec}
        if self.cfg.api_base:
            kwargs["base_url"] = self.cfg.api_base
        st.openai_client = OpenAI(**kwargs)
        return st.openai_client

    def _call_openai(self, prompt: str, temperature: float, max_tokens: int) -> str:
        openai_client = self._lazy_openai()
        completion = openai_client.chat.completions.create(  # type: ignore
            model=self.cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        return (completion.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------- azure
    def _lazy_azure(self):
        st = self._tls()
        if st.openai_client is not None:
            return st.openai_client
        from openai import AzureOpenAI  # type: ignore

        api_key = _pick_api_key(self.cfg.api_key_env)
        if not api_key:
            raise RuntimeError("missing API key for azure backend")
        endpoint = self.cfg.azure_endpoint or self.cfg.api_base
        if not endpoint:
            raise RuntimeError("azure backend needs cfg.azure_endpoint or cfg.api_base")
        st.openai_client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=self.cfg.azure_api_version,
            timeout=self.cfg.timeout_sec,
        )
        return st.openai_client

    def _call_azure(self, prompt: str, temperature: float, max_tokens: int) -> str:
        azure_client = self._lazy_azure()
        extra_body = {}
        budget = int(self.cfg.azure_thinking_budget)
        if budget > 0:
            extra_body["thinking"] = {"include_thoughts": False, "budget_tokens": budget}
        else:
            extra_body["thinking"] = {"budget_tokens": 0}

        completion = azure_client.chat.completions.create(  # type: ignore
            model=self.cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            extra_body=extra_body,
        )
        return (completion.choices[0].message.content or "").strip()

    # --------------------------------------------------------------- external

    @staticmethod
    def _is_rate_limit(err: Exception) -> bool:
        """Detect 429 / rate-limit errors across backends."""
        cls_name = type(err).__name__
        if "RateLimitError" in cls_name:
            return True
        if hasattr(err, "status_code") and getattr(err, "status_code", 0) == 429:
            return True
        if "429" in str(err) or "rate" in str(err).lower():
            return True
        return False

    def call(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        last_err: Optional[Exception] = None
        start_all = time.perf_counter()
        self._log(
            f"start backend={self.cfg.backend} model={self.cfg.model or self.cfg.local_model_path} "
            f"prompt_chars={len(prompt)} temp={temperature} max_tokens={max_tokens}"
        )
        max_attempts = self.cfg.max_retries + 4  # extra headroom for 429s
        for i in range(max_attempts):
            start_try = time.perf_counter()
            try:
                if self.cfg.backend in {"openai", "azure"}:
                    self._rate_wait()
                if self.cfg.backend == "openai":
                    out = self._call_openai(prompt, temperature, max_tokens)
                elif self.cfg.backend == "azure":
                    out = self._call_azure(prompt, temperature, max_tokens)
                elif self.cfg.backend == "local":
                    out = self._call_local(prompt, temperature, max_tokens)
                else:
                    raise ValueError(f"unknown backend: {self.cfg.backend}")
                elapsed = time.perf_counter() - start_try
                preview = (out or "").replace("\n", " ")[: int(self.cfg.log_output_chars)]
                self._log(
                    f"ok try={i + 1}/{max_attempts} cost={elapsed:.2f}s "
                    f"output={preview!r}"
                )
                return out
            except Exception as e:  # noqa: BLE001
                last_err = e
                elapsed = time.perf_counter() - start_try
                is_rl = self._is_rate_limit(e)
                self._log(
                    f"fail try={i + 1}/{max_attempts} cost={elapsed:.2f}s "
                    f"{'[429 rate-limit] ' if is_rl else ''}err={repr(e)}"
                )
                if is_rl:
                    # QPM limit: exponential backoff 5s, 10s, 20s, 40s …
                    sleep = min(5.0 * (2 ** i), 120.0)
                else:
                    if i + 1 >= self.cfg.max_retries:
                        break  # non-429 errors respect original max_retries
                    sleep = self.cfg.retry_sleep_sec * float(i + 1)
                self._log(f"  sleeping {sleep:.1f}s before retry")
                time.sleep(sleep)
        total = time.perf_counter() - start_all
        self._log(f"exhausted total_cost={total:.2f}s err={repr(last_err)}")
        raise RuntimeError(f"LLM call failed after retries: {repr(last_err)}")

    def call_many(
        self,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> List[str]:
        n = max(int(n), 1)
        workers = max(int(self.cfg.parallel_workers), 1)
        start = time.perf_counter()
        self._log(
            f"batch_start n={n} workers={min(n, workers)} temp={temperature} max_tokens={max_tokens}"
        )
        if n == 1 or workers == 1 or self.cfg.backend == "local":
            outs: List[str] = []
            errors: List[Exception] = []
            for _ in range(n):
                try:
                    outs.append(self.call(prompt, temperature=temperature, max_tokens=max_tokens))
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                    self._log(f"batch_item_fail err={repr(exc)}")
            if not outs:
                err = errors[-1] if errors else RuntimeError("empty LLM batch")
                self._log(f"batch_failed n={n} err={repr(err)}")
                raise RuntimeError(f"LLM batch failed after retries: {repr(err)}") from err
            if errors:
                self._log(f"batch_partial ok={len(outs)} failed={len(errors)}")
            self._log(f"batch_done n={n} cost={time.perf_counter() - start:.2f}s")
            return outs

        with ThreadPoolExecutor(max_workers=min(n, workers)) as pool:
            futures = [
                pool.submit(
                    self.call,
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                for _ in range(n)
            ]
            outs = []
            errors = []
            for f in futures:
                try:
                    outs.append(f.result())
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                    self._log(f"batch_item_fail err={repr(exc)}")
        if not outs:
            err = errors[-1] if errors else RuntimeError("empty LLM batch")
            self._log(f"batch_failed n={n} err={repr(err)}")
            raise RuntimeError(f"LLM batch failed after retries: {repr(err)}") from err
        if errors:
            self._log(f"batch_partial ok={len(outs)} failed={len(errors)}")
        self._log(f"batch_done n={n} cost={time.perf_counter() - start:.2f}s")
        return outs
