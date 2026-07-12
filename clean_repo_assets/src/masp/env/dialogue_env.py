"""
Partially-Observable Belief Game (POBG) dialogue environment.

An episode proceeds as follows:

  1. `reset(seed)`
       - samples a user mind `z_init = (B, D, I)` from MindPrior
       - stores `z_init` as the hidden "committed" user state
       - resets history and turn counter
       - returns an observation dict

  2. `step(system_action, user_action)`
       - appends the two turns to history
       - calls the OBU to update the ground-truth BDI `z*_{t+1}`
       - calls the mentalization module to get `ẑ_t` from the pre-step history
       - calls the rationality judge (LLM) on (assistant_turn, user_turn)
       - calls the DialogXpert P4G success judge
       - computes system / user rewards using the BDI progress delta
       - returns reward bundle + next observation

Self-play driver lives in `masp.rl.rollout`.  The environment is deliberately
*agent-agnostic*: it does not generate any text itself, it only maintains the
belief state and scores actions. The rollout driver is responsible for
calling the two policies.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..eval.p4g_judge import DialogXpertP4GJudge, RewardOutput
from ..mind.bdi_extractor import BDIExtractor
from ..mind.bdi_schema import BDI, BDITaskConfig, encode_bdi
from ..mind.mind_prior import MindPrior, MindPriorEntry
from ..models.mentalization import MentalizationModule, TeacherMentalizationModule
from ..models.outcome import OutcomeEnsemble, OutcomeRewardConfig
from ..models.sentence_encoder import SentenceEncoder
from ..rl.rewards import (
    RationalityJudge,
    RewardBundle,
    RewardConfig,
    fidelity_cosine,
    progress_score,
    a_weighted_sq_distance,
    system_close_quality,
    system_reward,
    system_safety_penalty,
    user_reward,
)


_EMPTY_TURN_TEXT = "(empty)"


def _nonempty_turn_text(text: str) -> str:
    s = str(text or "").strip()
    return s if s else _EMPTY_TURN_TEXT


# ----------------------------------------------------------- env config

@dataclass
class EnvStepInfo:
    history_lines: List[str]
    turn_idx: int
    done: bool
    success: bool
    reward_bundle: Optional[RewardBundle] = None


# ----------------------------------------------------------- environment

class POBGDialogueEnv:
    """
    The POBG environment. See module docstring for semantics.
    """

    def __init__(
        self,
        task_cfg: BDITaskConfig,
        mind_prior: MindPrior,
        sentence_encoder: SentenceEncoder,
        mentalization: MentalizationModule,
        bdi_extractor: BDIExtractor,
        p4g_judge: DialogXpertP4GJudge,
        rationality_judge: RationalityJudge,
        reward_cfg: RewardConfig,
        max_turns: int = 8,
        parallel_env_calls: bool = True,
        env_call_workers: int = 3,
        teacher_mentalization: Optional[TeacherMentalizationModule] = None,
        outcome_ensemble: Optional[OutcomeEnsemble] = None,
        outcome_reward_cfg: Optional[OutcomeRewardConfig] = None,
        local_obu_feedback: bool = False,
        local_rationality_feedback: bool = False,
        precomputed_z_goal: Optional[torch.Tensor] = None,
        precomputed_prior_entries: Optional[Sequence[MindPriorEntry]] = None,
        precomputed_prior_z: Optional[torch.Tensor] = None,
        session_meta_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.task_cfg = task_cfg
        self.mind_prior = mind_prior
        self.sentence_encoder = sentence_encoder
        self.mentalization = mentalization
        self.bdi_extractor = bdi_extractor
        self.p4g_judge = p4g_judge
        self.rationality_judge = rationality_judge
        self.cfg = reward_cfg
        self.max_turns = int(max_turns)
        self.parallel_env_calls = bool(parallel_env_calls)
        self.env_call_workers = max(int(env_call_workers), 1)

        # Optional frozen teacher F_ω (paper §3.2). When provided, the
        # reward-side state z_t is computed by the teacher from
        # (history, profile, goal) instead of going through the
        # oracle-LLM extract path. The oracle is still used for the
        # rationality-judge prompt (needs natural-language BDI text).
        self.teacher_mentalization = teacher_mentalization
        self._teacher_goal_text = task_cfg.goal_bdi.to_text(include_scalars=True)

        # Optional OGR ensemble (paper §A.2 / eq 28 / eq 37). When provided,
        # at terminal step we add λ_out·(w_p·p̄ + w_a·ā - w_σ·σ) to system reward.
        self.outcome_ensemble = outcome_ensemble
        self.outcome_reward_cfg = outcome_reward_cfg
        self.local_obu_feedback = bool(local_obu_feedback)
        self.local_rationality_feedback = bool(local_rationality_feedback)
        self._step_executor = (
            ThreadPoolExecutor(max_workers=self.env_call_workers)
            if self.parallel_env_calls and self.env_call_workers > 1
            else None
        )
        self.session_meta_by_id = dict(session_meta_by_id or {})
        self._episode_case_meta: Dict[str, Any] = {}

        # Paper §3.3 metric weights, hoisted from task_cfg for convenience.
        self._alpha_rho = float(task_cfg.alpha_rho)
        self._alpha_c = float(task_cfg.alpha_c)
        self._alpha_v = float(task_cfg.alpha_v)

        # Pre-compute z_goal / prior_z once on the template env and share them
        # with rollout clones. Re-encoding the full mind prior per episode clone
        # dominated phase2 startup latency (24 envs x hundreds of prior entries).
        if precomputed_z_goal is not None:
            self._z_goal = precomputed_z_goal.detach()
        else:
            self._z_goal = encode_bdi(
                task_cfg.goal_bdi, sentence_encoder,
                alpha_rho=self._alpha_rho, alpha_c=self._alpha_c, alpha_v=self._alpha_v,
            )
        self._prior_entries = (
            list(precomputed_prior_entries)
            if precomputed_prior_entries is not None
            else list(self.mind_prior.entries)
        )
        if precomputed_prior_z is not None:
            self._prior_z = precomputed_prior_z.detach().float()
        elif self._prior_entries:
            self._prior_z = torch.stack(
                [
                    encode_bdi(
                        e.bdi, self.sentence_encoder,
                        alpha_rho=self._alpha_rho, alpha_c=self._alpha_c, alpha_v=self._alpha_v,
                    )
                    for e in self._prior_entries
                ],
                dim=0,
            ).float()
        else:
            self._prior_z = torch.zeros(
                0, self._z_goal.shape[0], device=self._z_goal.device, dtype=torch.float32
            )

        # Episode state
        self.history_lines: List[str] = []
        self.turn_idx: int = 0
        self.done: bool = False
        self.success: bool = False

        self._committed_bdi: Optional[BDI] = None
        self._bdi_star_prev: Optional[BDI] = None
        self._z_init: Optional[torch.Tensor] = None
        self._z_init_text_emb: Optional[torch.Tensor] = None
        self._z_star_prev: Optional[torch.Tensor] = None
        self._prog_prev: float = 0.0
        self._episode_prior_entry: Optional[MindPriorEntry] = None
        self._cached_hist_key: Optional[str] = None
        self._cached_z_hat: Optional[torch.Tensor] = None

    # ------------------------------------------------------------ getters
    @property
    def committed_bdi(self) -> Optional[BDI]:
        return self._committed_bdi

    @property
    def current_bdi(self) -> Optional[BDI]:
        """Current pre-step user mind state used to condition π_U(u|h,z_t)."""
        return self._bdi_star_prev if self._bdi_star_prev is not None else self._committed_bdi

    @property
    def z_init(self) -> Optional[torch.Tensor]:
        return self._z_init

    @property
    def z_goal(self) -> torch.Tensor:
        return self._z_goal

    @property
    def z_star_prev(self) -> Optional[torch.Tensor]:
        """Current ground-truth BDI embedding at the pre-step hook point."""
        return self._z_star_prev

    def observation(self) -> Dict:
        return {
            "history_lines": list(self.history_lines),
            "turn_idx": int(self.turn_idx),
            "done": bool(self.done),
            "success": bool(self.success),
            "committed_bdi": self._committed_bdi,
        }

    # -------------------------------------------------------------- reset
    def reset(
        self,
        prior_entry: Optional[MindPriorEntry] = None,
        history_lines: Optional[Sequence[str]] = None,
        z_init_override: Optional[torch.Tensor] = None,
        z_init_text_emb_override: Optional[torch.Tensor] = None,
    ) -> Dict:
        entry = prior_entry if prior_entry is not None else self.mind_prior.sample()
        self._episode_prior_entry = entry
        self._episode_case_meta = dict(self.session_meta_by_id.get(entry.session_id, {}))
        self._committed_bdi = entry.bdi
        self._bdi_star_prev = entry.bdi
        # If rollout already batched the frozen teacher call, reuse that z_init.
        # Otherwise fall back to the original single-env path for eval/debug use.
        if z_init_override is not None:
            self._z_init = z_init_override.detach()
        elif self.teacher_mentalization is not None:
            with torch.no_grad():
                self._z_init = self.teacher_mentalization.forward_with_context(
                    ["(no turns yet)"], [entry.profile_text or ""], self._teacher_goal_text,
                )[0].detach()
        else:
            self._z_init = encode_bdi(
                entry.bdi, self.sentence_encoder,
                alpha_rho=self._alpha_rho, alpha_c=self._alpha_c, alpha_v=self._alpha_v,
            ).detach()
        if z_init_text_emb_override is not None:
            self._z_init_text_emb = z_init_text_emb_override.detach().float()
        else:
            with torch.no_grad():
                self._z_init_text_emb = self.sentence_encoder.encode([entry.bdi.to_text()])[0].detach().float()

        # If the caller supplies history_lines (e.g. mid-dialogue evaluation),
        # use them as the starting point, otherwise start from an empty
        # dialogue.
        self.history_lines = list(history_lines or [])
        self.turn_idx = 0
        self.done = False
        self.success = False

        # Initial ground-truth BDI = committed BDI (nothing has happened yet).
        self._z_star_prev = self._z_init.clone()
        self._prog_prev = float(
            progress_score(self._z_star_prev, self._z_init, self._z_goal).item()
        )
        self._cached_hist_key = None
        self._cached_z_hat = None

        return self.observation()

    # ------------------------------------------------------ mentalization I/O
    @staticmethod
    def _hist_key(history_lines: Sequence[str]) -> str:
        return "\n".join(history_lines[-40:])

    def infer_z_hat(self, history_lines: Sequence[str]) -> torch.Tensor:
        """
        Infer `ẑ_t` from history and cache the latest result so rollout prompt
        construction and env.step can share the same estimate.
        """
        key = self._hist_key(history_lines)
        if self._cached_hist_key == key and self._cached_z_hat is not None:
            return self._cached_z_hat
        with torch.no_grad():
            text = "\n".join(history_lines) if history_lines else "(no turns yet)"
            z_hat = self.mentalization.forward([text])[0].detach().float()
        self._cached_hist_key = key
        self._cached_z_hat = z_hat
        return z_hat

    def infer_bdi_hint_text(self, history_lines: Sequence[str]) -> str:
        """
        Convert the latent estimate `ẑ_t` into a natural-language BDI hint
        by nearest-neighbour retrieval over the mind prior.
        """
        if self._prior_z.shape[0] == 0:
            return "Belief: Unknown.\nDesire: Unknown.\nIntention: Unknown."
        z_hat = self.infer_z_hat(history_lines).to(self._prior_z.device)
        q = F.normalize(z_hat.unsqueeze(0), p=2, dim=-1)
        k = F.normalize(self._prior_z, p=2, dim=-1)
        sims = torch.matmul(k, q.transpose(0, 1)).squeeze(-1)  # (N,)
        idx = int(torch.argmax(sims).item())
        return self._prior_entries[idx].bdi.to_text()

    # ---------------------------------------------------------- core step
    def step(
        self,
        system_turn: str,
        user_turn: str,
    ) -> Tuple[Dict, RewardBundle]:
        bdi_star, rat, sig = self.query_remote_feedback(
            system_turn=system_turn,
            user_turn=user_turn,
        )
        return self.step_with_feedback(
            system_turn=system_turn,
            user_turn=user_turn,
            bdi_star=bdi_star,
            rat=rat,
            sig=sig,
        )

    def query_remote_feedback(
        self,
        system_turn: str,
        user_turn: str,
        pre_history: Optional[Sequence[str]] = None,
    ) -> Tuple[BDI, int, RewardOutput]:
        if self.done:
            raise RuntimeError("env is done; call reset()")
        if self._z_init is None:
            raise RuntimeError("env was not reset")

        pre_history = list(pre_history) if pre_history is not None else list(self.history_lines)
        history_after = list(pre_history) + [
            f"Assistant: {system_turn}",
            f"User: {user_turn}",
        ]
        current_bdi_text = self.current_bdi.to_text() if self.current_bdi else ""

        def _extract_bdi() -> BDI:
            if self.local_obu_feedback:
                bdi = self.current_bdi or self._committed_bdi
                if bdi is None:
                    raise RuntimeError("missing BDI state for local OBU feedback")
                return bdi
            return self.bdi_extractor.extract(history_after)

        def _judge_rationality() -> int:
            if self.local_rationality_feedback:
                return 1
            return self.rationality_judge(
                assistant_turn=system_turn,
                user_turn=user_turn,
                user_bdi_text=current_bdi_text,
            )

        def _score_success() -> RewardOutput:
            return self.p4g_judge.score(
                history_lines=pre_history,
                assistant_action=system_turn,
                user_reply=user_turn,
                case_meta=self._episode_case_meta,
            )

        # 1) Ground-truth BDI update via OBU.
        if self._step_executor is not None:
            fut_bdi = self._step_executor.submit(_extract_bdi)
            fut_rat = self._step_executor.submit(_judge_rationality)
            fut_sig = self._step_executor.submit(_score_success)
            bdi_star = fut_bdi.result()
            rat = fut_rat.result()
            sig = fut_sig.result()
        else:
            bdi_star = _extract_bdi()
            rat = _judge_rationality()
            sig = _score_success()
        return bdi_star, rat, sig

    def step_with_feedback(
        self,
        system_turn: str,
        user_turn: str,
        bdi_star: BDI,
        rat: int,
        sig: RewardOutput,
        pre_history: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict, RewardBundle]:
        if self.done:
            raise RuntimeError("env is done; call reset()")
        if self._z_init is None:
            raise RuntimeError("env was not reset")

        system_turn = _nonempty_turn_text(system_turn)
        user_turn = _nonempty_turn_text(user_turn)
        pre_history = list(pre_history) if pre_history is not None else list(self.history_lines)

        # Update visible history with the two new turns.
        self.history_lines.append(f"Assistant: {system_turn}")
        self.history_lines.append(f"User: {user_turn}")
        self.turn_idx += 1

        # Paper §3.2: prefer the teacher F_ω(h_post, p_u, g) to compute z_t.
        # When unavailable, fall back to encoding the oracle's BDI text directly
        # (legacy path; equivalent to using the oracle as a "teacher" with
        # zero training error but unbounded API cost).
        if self.teacher_mentalization is not None:
            history_after_text = "\n".join(
                self.history_lines[-self.bdi_extractor.max_history_lines:]
            ) or "(no turns yet)"
            profile_text = (
                self._episode_prior_entry.profile_text
                if self._episode_prior_entry is not None else ""
            )
            with torch.no_grad():
                z_star = self.teacher_mentalization.forward_with_context(
                    [history_after_text], [profile_text], self._teacher_goal_text,
                )[0].detach()
        else:
            z_star = encode_bdi(
                bdi_star, self.sentence_encoder,
                alpha_rho=self._alpha_rho, alpha_c=self._alpha_c, alpha_v=self._alpha_v,
            ).detach()
        prog_next = float(progress_score(z_star, self._z_init, self._z_goal).item())

        # 2) Mentalization estimate for the *pre-step* history (used for
        #    the system-side inference penalty).
        z_hat = self.infer_z_hat(pre_history).to(z_star.device)

        # Paper eq 23 second term: A-weighted ||ẑ_t - z_t||_A^2
        mental_err_sq = float(a_weighted_sq_distance(z_hat, self._z_star_prev).item())

        # 3) Voice fidelity — cosine between user utterance embedding and z_init.
        with torch.no_grad():
            u_emb = self.sentence_encoder.encode([user_turn])[0]  # (H,)
            if self._z_init_text_emb is not None:
                z_init_agg = F.normalize(self._z_init_text_emb, p=2, dim=-1)
            else:
                # _z_init now has shape (3d + 3,): three text-embedding blocks
                # of size d followed by 3 scalar dims (ρ, c, v). For voice
                # fidelity we only want the text part — slice it back out.
                d = self.sentence_encoder.hidden_size
                text_part = self._z_init[: 3 * d]
                z_init_components = text_part.view(3, d)
                z_init_agg = F.normalize(z_init_components.mean(dim=0), p=2, dim=-1)
            fid = fidelity_cosine(u_emb, z_init_agg)

        # 4) Rationality judge and 5) P4G success judge are computed above,
        #    overlapped with the OBU call whenever parallel_env_calls is on.
        success = bool(sig.success)
        self.success = success
        timeout = self.turn_idx >= self.max_turns
        self.done = bool(success or timeout)
        safety_pen = system_safety_penalty(system_turn, pre_history)
        close_quality = system_close_quality(
            system_turn,
            pre_history,
            turn_idx=self.turn_idx,
            success=success,
        )

        # 6) Rewards.
        r_s = system_reward(
            success=success,
            progress_t=self._prog_prev,
            progress_tp1=prog_next,
            mental_error_sq=mental_err_sq,
            step_penalty=self.cfg.step_penalty,
            safety_penalty=safety_pen,
            close_quality=close_quality,
            cfg=self.cfg,
            terminal=self.done,
            turn_idx=self.turn_idx,
            max_turns=self.max_turns,
            rationality_signal=int(rat),
        )

        # 6b) OGR terminal bump (paper §A.2 / eq 28). Only at terminal step,
        # only if an outcome ensemble was provided.
        ogr_reward = 0.0
        if self.done and self.outcome_ensemble is not None and self.outcome_reward_cfg is not None:
            full_history_text = "\n".join(self.history_lines)
            profile_text = (
                self._episode_prior_entry.profile_text
                if self._episode_prior_entry is not None else ""
            )
            ogr_reward = self.outcome_ensemble.reward(
                full_history_text, profile_text, self.outcome_reward_cfg,
            )
            r_s = float(r_s) + ogr_reward
        r_u = user_reward(
            progress_t=self._prog_prev,
            progress_tp1=prog_next,
            fidelity_cos=float(fid),
            rationality_signal=int(rat),
            cfg=self.cfg,
            success=success,
            terminal=self.done,
        )

        bundle = RewardBundle(
            r_system=float(r_s),
            r_user=float(r_u),
            success=bool(success),
            progress_delta=float(prog_next - self._prog_prev),
            rationality=int(rat),
            mental_error=float(mental_err_sq),
            fidelity=float(fid),
            task_label=str(sig.label),
            task_reward=float(sig.reward),
            safety_penalty=float(safety_pen),
            close_quality=float(close_quality),
            raw_judgments=list(sig.raw_judgments),
        )

        # Advance stored state.
        self._bdi_star_prev = bdi_star
        self._z_star_prev = z_star
        self._prog_prev = prog_next
        self._cached_hist_key = None
        self._cached_z_hat = None

        return self.observation(), bundle
