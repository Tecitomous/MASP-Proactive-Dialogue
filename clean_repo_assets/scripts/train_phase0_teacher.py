#!/usr/bin/env python3
"""
Phase 0b — Train the TEACHER mentalizer F_ω (paper §3.2 / eq 30).

Loss:
    L_F(ω) = λ_reg * Σ_t ||F_ω(h_t, p_u, g) - z̄_t||²_A
           + λ_dyn * Σ_t ||Δz^F_t - Δz̄_t||²_A
where
    Δz^F_t = F_ω(h_{t+1}, p_u, g) - F_ω(h_t, p_u, g)
    Δz̄_t   = z̄_{t+1} - z̄_t
and z̄_t is the silver oracle BDI from `data_cache/p4g_bdi_*.json`.

Inputs:
    --train_cache  data_cache/p4g_bdi_train.json
    --valid_cache  data_cache/p4g_bdi_valid.json
    --task         p4g  (drives task_cfg.alpha_*  + task_cfg.goal_bdi)
    --encoder_model  path to frozen sentence encoder (a compatible causal LM)

Outputs:
    out_dir/teacher_best.pt
    out_dir/teacher_train_log.json

After training the teacher is FROZEN and used in Phase 2 self-play to
compute the reward state z_t. The student M_φ (trained separately by
train_phase0_mentalization.py) mimics this teacher.

Usage on A100:
    DEVICE=cuda:4 OUT_DIR=checkpoints/phase0 \
      bash new_config_file/h20_phase0_teacher.sh
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from masp.data.bdi_dataset import BDILabelCache
from masp.mind.bdi_schema import BDI, TASK_CONFIGS
from masp.models.mentalization import (
    MentalizationConfig,
    TeacherMentalizationModule,
)
from masp.models.sentence_encoder import SentenceEncoder, SentenceEncoderConfig
from masp.utils.io import dump_json, ensure_dir
from masp.utils.seed import set_seed


# ----------------------------------------------------------------------- data

def _build_session_groups(cache: BDILabelCache) -> Dict[str, List[int]]:
    """session_id → ordered list of indices into cache.entries"""
    groups: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for idx, e in enumerate(cache.entries):
        groups[e.session_id].append((e.turn_idx, idx))
    out: Dict[str, List[int]] = {}
    for sid, lst in groups.items():
        lst.sort(key=lambda x: x[0])
        out[sid] = [idx for _, idx in lst]
    return out


def _build_pair_examples(cache: BDILabelCache) -> List[Dict]:
    """
    For dynamic loss we need consecutive (turn_t, turn_{t+1}) pairs from the
    same session. We also keep the singleton turns (no pair available) for
    the regression-only loss term.
    """
    groups = _build_session_groups(cache)
    pairs: List[Dict] = []
    for sid, idxs in groups.items():
        profile_text = cache.profile_text.get(sid, "")
        for i in range(len(idxs)):
            ent_t = cache.entries[idxs[i]]
            history_t = "\n".join(ent_t.history_upto[-30:]) or "(no turns yet)"
            ent_tp1 = cache.entries[idxs[i + 1]] if i + 1 < len(idxs) else None
            history_tp1 = (
                "\n".join(ent_tp1.history_upto[-30:]) if ent_tp1 is not None else None
            )
            pairs.append({
                "session_id": sid,
                "profile_text": profile_text,
                "history_t": history_t,
                "history_tp1": history_tp1,           # may be None
                "bdi_t": ent_t.bdi,
                "bdi_tp1": ent_tp1.bdi if ent_tp1 is not None else None,
            })
    return pairs


def _collate(batch: List[Dict]) -> Dict:
    return {
        "session_id":   [b["session_id"]   for b in batch],
        "profile_text": [b["profile_text"] for b in batch],
        "history_t":    [b["history_t"]    for b in batch],
        "history_tp1":  [b["history_tp1"]  for b in batch],
        "bdi_t":        [b["bdi_t"]        for b in batch],
        "bdi_tp1":      [b["bdi_tp1"]      for b in batch],
    }


# ----------------------------------------------------------------------- loss

def _encode_targets(bdis: List[BDI], encoder: SentenceEncoder, alphas: Tuple[float, float, float]) -> torch.Tensor:
    """Build z̄_t target batch via the same encode_bdi used by the env."""
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
                _math.sqrt(max(alphas[0], 0.0)) * float(b.receptivity),
                _math.sqrt(max(alphas[1], 0.0)) * float(b.confidence),
                _math.sqrt(max(alphas[2], 0.0)) * float(b.valence),
            ]
            for b in bdis
        ],
        dtype=text.dtype,
        device=text.device,
    )
    z = torch.cat([text, scalars], dim=-1)
    return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _diagnostic_metrics(z_pred: torch.Tensor, z_targ: torch.Tensor, d: int) -> Dict[str, float]:
    """
    Compute interpretable metrics alongside MSE.
    z_pred, z_targ: (B, 3d+3) — first 3d dims are 3 L2-normalized text parts,
                                  last 3 dims are √α-scaled scalars.
    """
    B = z_pred.shape[0]
    metrics: Dict[str, float] = {}

    # ---- Text cosine similarity (avg over B, D, I) ----
    cos_sims = []
    for k in range(3):
        p = z_pred[:, k*d:(k+1)*d]   # (B, d)
        t = z_targ[:, k*d:(k+1)*d]
        cos = F.cosine_similarity(p, t, dim=-1).mean()
        cos_sims.append(float(cos.item()))
    metrics["cos_B"] = cos_sims[0]
    metrics["cos_D"] = cos_sims[1]
    metrics["cos_I"] = cos_sims[2]
    metrics["cos_avg"] = sum(cos_sims) / 3.0

    # ---- Scalar MAE (last 3 dims: √α·ρ, √α·c, √α·v) ----
    scalar_pred = z_pred[:, -3:]
    scalar_targ = z_targ[:, -3:]
    mae = (scalar_pred - scalar_targ).abs().mean(dim=0)  # (3,)
    metrics["mae_rho"] = float(mae[0].item())
    metrics["mae_c"] = float(mae[1].item())
    metrics["mae_v"] = float(mae[2].item())

    # ---- Total MSE as SUM (not averaged over dims) for readable scale ----
    metrics["mse_sum"] = float(F.mse_loss(z_pred, z_targ, reduction="sum").item() / B)

    return metrics


def _bdi_loss(z_pred: torch.Tensor, z_target: torch.Tensor, scalar_loss_weight: float) -> torch.Tensor:
    if z_pred.shape != z_target.shape:
        raise ValueError(f"shape mismatch: {z_pred.shape} vs {z_target.shape}")
    if float(scalar_loss_weight) < 0:
        return TeacherMentalizationModule.bdi_regression_loss(z_pred, z_target)
    text_loss = F.mse_loss(z_pred[:, :-3], z_target[:, :-3])
    scalar_loss = F.mse_loss(z_pred[:, -3:], z_target[:, -3:])
    return text_loss + float(scalar_loss_weight) * scalar_loss


def _step_loss(
    teacher: TeacherMentalizationModule,
    encoder: SentenceEncoder,
    batch: Dict,
    goal_text: str,
    alphas: Tuple[float, float, float],
    lambda_reg: float,
    lambda_dyn: float,
    scalar_loss_weight: float,
    compute_diagnostics: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # ---- regression L_reg (always computable) ----
    z_pred_t = teacher.forward_with_context(batch["history_t"], batch["profile_text"], goal_text)
    z_targ_t = _encode_targets(batch["bdi_t"], encoder, alphas).to(z_pred_t.device)
    loss_reg = _bdi_loss(z_pred_t, z_targ_t, scalar_loss_weight)

    # ---- dynamic L_dyn (only on rows that have a t+1 sibling) ----
    valid_pair_idx = [i for i, h in enumerate(batch["history_tp1"]) if h is not None]
    if lambda_dyn > 0 and valid_pair_idx:
        sub_hist_tp1 = [batch["history_tp1"][i] for i in valid_pair_idx]
        sub_prof    = [batch["profile_text"][i] for i in valid_pair_idx]
        sub_bdi_tp1 = [batch["bdi_tp1"][i]    for i in valid_pair_idx]

        z_pred_tp1 = teacher.forward_with_context(sub_hist_tp1, sub_prof, goal_text)
        z_targ_tp1 = _encode_targets(sub_bdi_tp1, encoder, alphas).to(z_pred_tp1.device)
        z_pred_t_sub = z_pred_t[valid_pair_idx]
        z_targ_t_sub = z_targ_t[valid_pair_idx]
        loss_dyn = _bdi_loss(
            z_pred_tp1 - z_pred_t_sub,
            z_targ_tp1 - z_targ_t_sub,
            scalar_loss_weight,
        )
    else:
        loss_dyn = torch.zeros((), device=z_pred_t.device)

    total = lambda_reg * loss_reg + lambda_dyn * loss_dyn

    parts: Dict[str, float] = {
        "loss_reg": float(loss_reg.item()),
        "loss_dyn": float(loss_dyn.item()),
        "loss_total": float(total.item()),
    }

    if compute_diagnostics:
        d = int(encoder.hidden_size)
        diag = _diagnostic_metrics(z_pred_t.detach(), z_targ_t.detach(), d)
        parts.update(diag)

    return total, parts


# ----------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_cache", type=str, required=True)
    p.add_argument("--valid_cache", type=str, default="")
    p.add_argument("--encoder_model", type=str, required=True,
                   help="Path to a compatible causal LM (or any HF model) used as frozen φ.")
    p.add_argument("--encoder_device", type=str, default="cuda:4")
    p.add_argument("--encoder_dtype", type=str, default="bf16")
    p.add_argument("--encoder_max_len", type=int, default=384,
                   help="Composite [GOAL]+[PROFILE]+[HISTORY] input is longer "
                        "than history alone.")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--proj_hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_history_lines", type=int, default=30)
    p.add_argument("--lambda_reg", type=float, default=1.0,
                   help="Paper §A.1 eq 30: weight on regression loss.")
    p.add_argument("--lambda_dyn", type=float, default=0.5,
                   help="Paper §A.1 eq 30: weight on dynamic delta-loss.")
    p.add_argument("--scalar_loss_weight", type=float, default=-1.0,
                   help="If >= 0, use text_mse + weight * scalar_mse instead "
                        "of plain mean MSE over 3d+3 dims. This prevents "
                        "rho/confidence/valence from being diluted by text dims.")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--task", type=str, default="p4g", choices=list(TASK_CONFIGS.keys()))
    args = p.parse_args()
    task_cfg = TASK_CONFIGS[args.task]
    alphas = (task_cfg.alpha_rho, task_cfg.alpha_c, task_cfg.alpha_v)
    goal_text = task_cfg.goal_bdi.to_text(include_scalars=True)

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    # ---- data ----
    print(f"[teacher] loading train cache: {args.train_cache}")
    train_cache = BDILabelCache.load(args.train_cache)
    valid_cache = BDILabelCache.load(args.valid_cache) if args.valid_cache else None
    print(f"[teacher] train sessions={len(train_cache.initial_bdi)} entries={len(train_cache.entries)}")

    train_examples = _build_pair_examples(train_cache)
    valid_examples = _build_pair_examples(valid_cache) if valid_cache else None
    print(f"[teacher] train pairs (with t+1)={sum(1 for x in train_examples if x['history_tp1'] is not None)}")
    print(f"[teacher] train singletons     ={sum(1 for x in train_examples if x['history_tp1'] is None)}")

    train_loader = DataLoader(train_examples, batch_size=args.batch_size,
                              shuffle=True, collate_fn=_collate, num_workers=0)
    valid_loader = (
        DataLoader(valid_examples, batch_size=args.batch_size, shuffle=False,
                   collate_fn=_collate, num_workers=0)
        if valid_examples else None
    )

    # ---- model ----
    print(f"[teacher] loading frozen encoder: {args.encoder_model} on {args.encoder_device}")
    encoder = SentenceEncoder(SentenceEncoderConfig(
        model_name_or_path=args.encoder_model,
        device=args.encoder_device,
        dtype=args.encoder_dtype,
        max_len=args.encoder_max_len,
    ))
    mcfg = MentalizationConfig(
        hidden_size=encoder.hidden_size,
        proj_hidden=args.proj_hidden,
        dropout=args.dropout,
        alpha_rho=task_cfg.alpha_rho,
        alpha_c=task_cfg.alpha_c,
        alpha_v=task_cfg.alpha_v,
    )
    teacher = TeacherMentalizationModule(encoder, mcfg).to(encoder.device)

    optimizer = torch.optim.AdamW(
        teacher.trainable_parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    train_log: List[Dict] = []
    best_val = float("inf")
    step = 0

    for epoch in range(args.num_epochs):
        teacher.train()
        it = tqdm(train_loader, desc=f"[teacher] epoch {epoch + 1}/{args.num_epochs}")
        for batch in it:
            step += 1
            # Compute diagnostics every log_every steps (cheap; just extra .detach())
            do_diag = (step % args.log_every == 0)
            loss, parts = _step_loss(
                teacher, encoder, batch,
                goal_text=goal_text, alphas=alphas,
                lambda_reg=args.lambda_reg, lambda_dyn=args.lambda_dyn,
                scalar_loss_weight=args.scalar_loss_weight,
                compute_diagnostics=do_diag,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.trainable_parameters(), 1.0)
            optimizer.step()

            # Show interpretable metrics in progress bar
            postfix = {"reg": f"{parts['loss_reg']:.2e}", "dyn": f"{parts['loss_dyn']:.2e}"}
            if do_diag:
                postfix["cos"] = f"{parts.get('cos_avg', 0):.3f}"
                postfix["mae_s"] = f"{parts.get('mae_rho', 0):.3f}"
            it.set_postfix(postfix)

            if do_diag:
                train_log.append({"step": step, "epoch": epoch, **parts})

            if valid_loader is not None and step % args.val_every == 0:
                teacher.eval()
                vloss = 0.0; vn = 0
                val_diag_accum: Dict[str, float] = defaultdict(float)
                with torch.no_grad():
                    for vb in valid_loader:
                        l, vparts = _step_loss(
                            teacher, encoder, vb,
                            goal_text=goal_text, alphas=alphas,
                            lambda_reg=args.lambda_reg, lambda_dyn=args.lambda_dyn,
                            scalar_loss_weight=args.scalar_loss_weight,
                            compute_diagnostics=True,
                        )
                        bs = len(vb["history_t"])
                        if math.isfinite(float(l.item())):
                            vloss += float(l.item()) * bs
                            vn += bs
                            for k, v in vparts.items():
                                if k.startswith("cos_") or k.startswith("mae_") or k == "mse_sum":
                                    val_diag_accum[k] += v * bs
                vloss = vloss / max(vn, 1)
                val_diag = {k: v / max(vn, 1) for k, v in val_diag_accum.items()}
                train_log.append({"step": step, "val_loss": float(vloss), **val_diag})
                if vloss < best_val:
                    best_val = vloss
                    teacher.save(os.path.join(args.out_dir, "teacher_best.pt"))
                    print(f"[teacher] new best val={vloss:.2e} "
                          f"cos_avg={val_diag.get('cos_avg', 0):.3f} "
                          f"mae_ρ={val_diag.get('mae_rho', 0):.3f} "
                          f"mae_c={val_diag.get('mae_c', 0):.3f} "
                          f"mae_v={val_diag.get('mae_v', 0):.3f} "
                          f"-> teacher_best.pt")
                teacher.train()

    # always also save the final state
    teacher.save(os.path.join(args.out_dir, "teacher_final.pt"))
    if best_val == float("inf"):
        teacher.save(os.path.join(args.out_dir, "teacher_best.pt"))

    dump_json({"log": train_log, "best_val": best_val,
               "lambda_reg": args.lambda_reg, "lambda_dyn": args.lambda_dyn,
               "scalar_loss_weight": args.scalar_loss_weight},
              os.path.join(args.out_dir, "teacher_train_log.json"))
    print(f"[teacher] done. best_val={best_val:.2e}")


if __name__ == "__main__":
    main()
