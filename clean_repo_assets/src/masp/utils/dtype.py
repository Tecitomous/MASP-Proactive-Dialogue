import torch


def pick_dtype(name: str) -> torch.dtype:
    name = (name or "").lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")
