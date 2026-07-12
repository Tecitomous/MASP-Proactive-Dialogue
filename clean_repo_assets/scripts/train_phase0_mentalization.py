#!/usr/bin/env python3
"""
Phase 0b — pretrain the Mentalization Module `M_ψ`.

Input:
    - Silver BDI label cache produced by `extract_bdi_labels.py`
    - A frozen sentence encoder (default: a compatible causal LM base — any HF LM works)

Loss:
    L_M(ψ) = ||M_ψ(history) - z*||^2

where z* is the embedding of the BDI label passed through the same frozen
sentence encoder used at inference time. Component-wise normalization means
this loss is equivalent (up to a constant) to `3 - cos_sim_sum`.

Outputs:
    - checkpoints/phase0/mentalization.pt
    - checkpoints/phase0/train_log.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from masp.data.bdi_dataset import BDILabelCache, build_bdi_turn_dataset
from masp.mind.bdi_schema import BDI, TASK_CONFIGS
from masp.models.mentalization import (
    MentalizationConfig,
    MentalizationModule,
    TeacherMentalizationModule,
)
from masp.models.sentence_encoder import SentenceEncoder, SentenceEncoderConfig
from masp.utils.io import dump_json, ensure_dir
from masp.utils.seed import set_seed


def _collate(batch: List[Dict]) -> Dict:
    """Forward all fields needed by both code paths:
      - teacher path uses history_text + profile_text  (paper eq 31, F_ω(h, p_u, g))
      - legacy silver-label path uses BDI text + scalars (encode_bdi)
    Earlier this function only carried the 4 BDI text fields, which crashed
    the moment a teacher_ckpt was supplied.
    """
    return {
        "history_text":     [x["history_text"]              for x in batch],
        "profile_text":     [x.get("profile_text", "")      for x in batch],
        "bdi_belief":       [x["bdi_belief"]                for x in batch],
        "bdi_desire":       [x["bdi_desire"]                for x in batch],
        "bdi_intention":    [x["bdi_intention"]             for x in batch],
        "bdi_receptivity":  [float(x.get("bdi_receptivity", 0.5)) for x in batch],
        "bdi_confidence":   [float(x.get("bdi_confidence",  0.5)) for x in batch],
        "bdi_valence":      [float(x.get("bdi_valence",     0.0)) for x in batch],
    }


def _encode_silver_bdi_batch(
    batch: Dict,
    encoder: SentenceEncoder,
    task_cfg,
) -> torch.Tensor:
    bdis = [
        BDI(
            belief=b or "Unknown.",
            desire=d or "Unknown.",
            intention=i or "Unknown.",
            receptivity=float(rho),
            confidence=float(conf),
            valence=float(val),
        )
        for b, d, i, rho, conf, val in zip(
            batch["bdi_belief"],
            batch["bdi_desire"],
            batch["bdi_intention"],
            batch["bdi_receptivity"],
            batch["bdi_confidence"],
            batch["bdi_valence"],
        )
    ]
    if not bdis:
        return torch.zeros(0, 3 * encoder.hidden_size + 3, device=encoder.device)

    with torch.no_grad():
        b_emb = encoder.encode([b.belief for b in bdis])
        d_emb = encoder.encode([b.desire for b in bdis])
        i_emb = encoder.encode([b.intention for b in bdis])
    text = torch.cat([b_emb, d_emb, i_emb], dim=-1).float()
    import math as _math
    scalars = torch.tensor(
        [
            [
                _math.sqrt(max(task_cfg.alpha_rho, 0.0)) * float(b.receptivity),
                _math.sqrt(max(task_cfg.alpha_c, 0.0)) * float(b.confidence),
                _math.sqrt(max(task_cfg.alpha_v, 0.0)) * float(b.valence),
            ]
            for b in bdis
        ],
        dtype=text.dtype,
        device=text.device,
    )
    z = torch.cat([text, scalars], dim=-1)
    return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _bdi_loss(z_pred: torch.Tensor, z_target: torch.Tensor, scalar_loss_weight: float) -> torch.Tensor:
    if z_pred.shape != z_target.shape:
        raise ValueError(f"shape mismatch: {z_pred.shape} vs {z_target.shape}")
    if float(scalar_loss_weight) < 0:
        return MentalizationModule.bdi_regression_loss(z_pred, z_target)
    text_loss = torch.nn.functional.mse_loss(z_pred[:, :-3], z_target[:, :-3])
    scalar_loss = torch.nn.functional.mse_loss(z_pred[:, -3:], z_target[:, -3:])
    return text_loss + float(scalar_loss_weight) * scalar_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_cache", type=str, required=True)
    p.add_argument("--valid_cache", type=str, default="")
    p.add_argument("--encoder_model", type=str, required=True,
                   help="Path to a compatible causal LM (or any HF model) used as frozen "
                        "sentence encoder φ.")
    p.add_argument("--encoder_device", type=str, default="cuda:0")
    p.add_argument("--encoder_dtype", type=str, default="bf16")
    p.add_argument("--encoder_max_len", type=int, default=128)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--proj_hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_history_lines", type=int, default=30)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--scalar_loss_weight", type=float, default=-1.0,
                   help="If >= 0, use text_mse + weight * scalar_mse instead "
                        "of plain mean MSE over 3d+3 dims. This prevents "
                        "rho/confidence/valence from being diluted by text dims.")
    p.add_argument("--task", type=str, default="p4g", choices=list(TASK_CONFIGS.keys()),
                   help="Used to fetch (alpha_rho, alpha_c, alpha_v) for encode_bdi.")
    p.add_argument("--teacher_ckpt", type=str, default="",
                   help="If set, train the student to mimic the FROZEN teacher "
                        "F_ω (paper §3.2 / eq 31). The teacher's profile + goal "
                        "context is loaded from --task. If empty, falls back to "
                        "the legacy behavior of regressing directly to silver "
                        "labels (B/D/I encode_bdi).")
    args = p.parse_args()
    task_cfg = TASK_CONFIGS[args.task]

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    print("[phase0] loading BDI cache...")
    train_cache = BDILabelCache.load(args.train_cache)
    valid_cache = BDILabelCache.load(args.valid_cache) if args.valid_cache else None
    print(f"[phase0] train entries = {len(train_cache.entries)}")

    train_ds = build_bdi_turn_dataset(train_cache, max_history_lines=args.max_history_lines)
    valid_ds = build_bdi_turn_dataset(valid_cache, max_history_lines=args.max_history_lines) if valid_cache else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
        drop_last=True,
    )
    valid_loader = None
    if valid_ds is not None:
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate,
        )

    print("[phase0] loading frozen sentence encoder...")
    encoder = SentenceEncoder(
        SentenceEncoderConfig(
            model_name_or_path=args.encoder_model,
            device=args.encoder_device,
            dtype=args.encoder_dtype,
            max_len=args.encoder_max_len,
        )
    )
    mcfg = MentalizationConfig(
        hidden_size=encoder.hidden_size,
        proj_hidden=args.proj_hidden,
        dropout=args.dropout,
        alpha_rho=task_cfg.alpha_rho,
        alpha_c=task_cfg.alpha_c,
        alpha_v=task_cfg.alpha_v,
    )
    mentalizer = MentalizationModule(encoder, mcfg).to(encoder.device)

    # ---- optional: load FROZEN teacher for paper eq 31 (student mimics teacher) ----
    teacher: TeacherMentalizationModule | None = None
    goal_text_for_teacher = ""
    if args.teacher_ckpt:
        if not os.path.isfile(args.teacher_ckpt):
            raise FileNotFoundError(f"--teacher_ckpt not found: {args.teacher_ckpt}")
        print(f"[phase0] loading frozen teacher from {args.teacher_ckpt}")
        teacher = TeacherMentalizationModule(encoder, mcfg).to(encoder.device)
        teacher.load(args.teacher_ckpt, map_location=str(encoder.device))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        goal_text_for_teacher = task_cfg.goal_bdi.to_text(include_scalars=True)
        print("[phase0] student will be trained to MIMIC frozen teacher (paper eq 31).")
    else:
        print("[phase0] --teacher_ckpt unset → falling back to direct silver-label "
              "regression (legacy behaviour).")

    optimizer = torch.optim.AdamW(
        mentalizer.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_log: List[Dict] = []
    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "mentalization_best.pt")
    step = 0

    def run_validation() -> float:
        if valid_loader is None:
            return float("inf")
        mentalizer.eval()
        vloss = 0.0
        vn = 0
        with torch.no_grad():
            for vb in valid_loader:
                if teacher is not None:
                    zt = teacher.forward_with_context(
                        vb["history_text"], vb["profile_text"], goal_text_for_teacher,
                    ).detach()
                else:
                    zt = _encode_silver_bdi_batch(vb, encoder, task_cfg)
                zp = mentalizer(vb["history_text"])
                vl = _bdi_loss(zp, zt, args.scalar_loss_weight)
                if math.isfinite(float(vl.item())):
                    vloss += float(vl.item()) * len(vb["history_text"])
                    vn += len(vb["history_text"])
        mentalizer.train()
        return vloss / max(vn, 1)

    for epoch in range(args.num_epochs):
        mentalizer.train()
        it = tqdm(train_loader, desc=f"[phase0] epoch {epoch + 1}/{args.num_epochs}")
        for batch in it:
            step += 1
            # Build target z*. Two modes:
            #   (a) --teacher_ckpt set  → target = sg(teacher(h, p_u, g))  [paper eq 31]
            #   (b) --teacher_ckpt unset → target = encode_bdi(silver_label)  [legacy]
            if teacher is not None:
                with torch.no_grad():
                    z_target = teacher.forward_with_context(
                        batch["history_text"],
                        batch["profile_text"],
                        goal_text_for_teacher,
                    ).detach()                                # (B, 3d + 3)
            else:
                z_target = _encode_silver_bdi_batch(batch, encoder, task_cfg)

            z_pred = mentalizer(batch["history_text"])              # (B, 3d + 3) — must match
            loss = _bdi_loss(z_pred, z_target, args.scalar_loss_weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mentalizer.trainable_parameters(), 1.0)
            optimizer.step()

            it.set_postfix({"loss": f"{float(loss.item()):.4f}"})
            if step % args.log_every == 0:
                train_log.append({
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                })

            if valid_loader is not None and step % args.val_every == 0:
                vloss = run_validation()
                train_log.append({"step": step, "val_loss": float(vloss)})
                if vloss < best_val:
                    best_val = vloss
                    mentalizer.save(best_path)

    if valid_loader is not None and not math.isfinite(best_val):
        vloss = run_validation()
        train_log.append({"step": step, "val_loss": float(vloss), "final_eval": True})
        best_val = vloss
        mentalizer.save(best_path)
    elif not os.path.exists(best_path):
        mentalizer.save(best_path)

    mentalizer.save(os.path.join(args.out_dir, "mentalization_last.pt"))
    dump_json(
        {
            "train_log": train_log,
            "best_val": float(best_val),
            "scalar_loss_weight": args.scalar_loss_weight,
        },
        os.path.join(args.out_dir, "train_log.json"),
    )
    print(f"[phase0] done. best val loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
