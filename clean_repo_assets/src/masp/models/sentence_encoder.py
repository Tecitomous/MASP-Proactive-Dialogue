"""
Frozen sentence encoder φ.

We deliberately keep φ *simple and frozen* so the whole BDI latent space is
stable across Phase 0 / Phase 1 / Phase 2. Any of the following are fine:
  - a BGE / E5 sentence embedding model
  - the mean-pooled last hidden state of any HF CausalLM / encoder LM
  - (fallback) the mean-pooled word embeddings of the tokenizer

The default backs onto HuggingFace AutoModel with mean pooling, which works
equally well for BERT-family and decoder-only LLMs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ..utils.dtype import pick_dtype


_EMPTY_TEXT = "(empty)"


@dataclass
class SentenceEncoderConfig:
    model_name_or_path: str
    device: str = "cuda:0"
    dtype: str = "bf16"
    max_len: int = 128
    trust_remote_code: bool = True


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask: (B, T) — average over valid tokens."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    s = (hidden * m).sum(dim=1)
    d = m.sum(dim=1).clamp(min=1.0)
    return s / d


class SentenceEncoder(nn.Module):
    def __init__(self, cfg: SentenceEncoderConfig):
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

        self.model = AutoModel.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=cfg.trust_remote_code,
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Discover hidden size without failing on decoder-only vs encoder-only.
        hidden = getattr(self.model.config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(self.model.config, "d_model", None)
        if hidden is None:
            raise RuntimeError("could not determine hidden_size from model.config")
        self._hidden_size = int(hidden)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, self._hidden_size, device=self.device)
        clean_texts = []
        for text in texts:
            s = str(text or "").strip()
            clean_texts.append(s if s else _EMPTY_TEXT)
        enc = self.tokenizer(
            clean_texts,
            padding=True,
            truncation=True,
            max_length=max(int(self.cfg.max_len), 1),
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**enc, output_hidden_states=False, return_dict=True)
        hidden = out.last_hidden_state  # (B, T, H)
        pooled = _mean_pool(hidden, enc["attention_mask"])
        pooled = pooled.float()
        # L2 normalise so that cosine / L2 distances are directly comparable.
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled
