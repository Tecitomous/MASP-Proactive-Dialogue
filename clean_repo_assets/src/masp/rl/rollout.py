"""
Self-play rollout engine.

Given:
  - a POBG environment
  - the system policy π_S
  - the adversarial user policy π_U

this module runs N full dialogues (each up to `max_turns` turns) and returns:
  - a TrajectoryBuffer containing one step per generated turn per role,
    ready for PPO updates
  - per-episode success / turn counts
  - per-step diagnostic metrics

The rollout is *blocking* and single-process. Multi-GPU parallelism is
achieved by placing π_S and π_U on different devices (see README §6).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

from concurrent.futures import ThreadPoolExecutor
import re
import torch
import torch.nn.functional as F
import time

from ..eval.metrics import DialogMetrics
from ..env.dialogue_env import POBGDialogueEnv
from ..mind.mind_prior import MindPriorEntry
from ..models.policy import LoRAPolicy, build_chat_prompt_for_policy
from .buffer import TrajectoryBuffer, TrajectoryStep
from .rewards import RewardBundle


_DONATION_REFUSAL_RE = re.compile(
    r"\b("
    r"no[\s,]+thank(?:s| you)?|"
    r"no thank|not donate|not give|don'?t want to donate|do not want to donate|"
    r"wouldn'?t donate|wouldn'?t give|would do nothing|will donate 0|donate 0|"
    r"0 dollars|\$0|nothing at all|can'?t donate|cannot donate|not at this time|"
    r"don'?t plan on donating|do not plan on donating|would not like|would not do that"
    r")\b",
    re.IGNORECASE,
)
_COERCIVE_ASSIGN_RE = re.compile(
    r"\b(put you down|mark you down|sign you up|count you (?:in|for)|"
    r"i'?ll put|i will put|i'?ll still put|i will still put|record you for|"
    r"take (?:your )?refusal as (?:a )?yes)\b",
    re.IGNORECASE,
)
_GOODBYE_RE = re.compile(
    r"\b(goodbye|bye|have a (?:great|nice|wonderful)|enjoy the rest|"
    r"thanks for (?:listening|your time)|thank you for your time)\b",
    re.IGNORECASE,
)
_DONATION_DECISION_RE = re.compile(
    r"\b(how much|would you like to donate|will you donate|willing to donate|"
    r"what amount|donate a small|donate \$|donate [0-9]|make a donation|"
    r"make a small donation|will you make|would you make)\b",
    re.IGNORECASE,
)
_TASK_META_RE = re.compile(
    r"\b(need at least .*turns?|complete this task|cannot complete this task)\b",
    re.IGNORECASE,
)


def _last_user_turn(history_lines: List[str]) -> str:
    for line in reversed(history_lines):
        if str(line).startswith("User:"):
            return str(line)[5:].strip()
    return ""


def _last_user_refused_donation(history_lines: List[str]) -> bool:
    last_user = _last_user_turn(history_lines)
    if re.search(r"don'?t want to donate too little|do not want to donate too little", last_user, re.IGNORECASE):
        return False
    return bool(_DONATION_REFUSAL_RE.search(last_user))


def _recent_user_refused_donation(history_lines: List[str], max_lines: int = 6) -> bool:
    for line in reversed(history_lines[-max_lines:]):
        if not str(line).startswith("User:"):
            continue
        user_text = str(line)[5:].strip()
        if re.search(r"don'?t want to donate too little|do not want to donate too little", user_text, re.IGNORECASE):
            continue
        if _DONATION_REFUSAL_RE.search(user_text):
            return True
    return False


def _postprocess_system_turn(
    text: str,
    pre_history: List[str],
    task_name: str = "p4g",
) -> str:
    """
    Generation-time guardrail for P4G rollouts.

    This is deliberately narrow: it removes placeholder URLs, blocks assigning
    money without consent, and prevents generic goodbye turns when the user has
    not clearly refused or agreed. The edited text is re-tokenized before PPO
    bookkeeping so rollout text and stored response ids stay aligned.
    """
    raw = (text or "").strip()
    task = (task_name or "p4g").lower().strip()
    if task == "craigslist_bargain":
        cleaned = re.sub(r"\bhttps?://\S*URL\S*\b", "the listing details", raw)
        cleaned = re.sub(r"\bURL\b", "the listing details", cleaned)
        if (
            re.search(
                r"\b(save the children|charit(?:y|ies)|donat(?:e|ion)|"
                r"fundrais(?:e|er|ing)|task payment|\$0\.?25)\b",
                cleaned,
                re.IGNORECASE,
            )
            or _TASK_META_RE.search(cleaned)
            or not cleaned
        ):
            last_user = _last_user_turn(pre_history)
            price = re.findall(r"\$?\s*([0-9]+(?:\.[0-9]+)?)", last_user)
            if price:
                return (
                    f"{price[-1]} is still a bit high for my budget. Could you "
                    "come down a little so we can make this work?"
                )
            return "I'm interested, but I need the price to work for my budget. What is the best you can do?"
        if len(cleaned.split()) > 60:
            cleaned = " ".join(cleaned.split()[:60]).rstrip(" ,;:") + "."
        return cleaned
    if task != "p4g":
        cleaned = re.sub(r"\bhttps?://\S*URL\S*\b", "a trusted support resource", raw)
        cleaned = re.sub(r"\bURL\b", "a trusted support resource", cleaned)
        if (
            re.search(
                r"\b(save the children|charit(?:y|ies)|donat(?:e|ion)|"
                r"fundrais(?:e|er|ing)|task payment|\$0\.?25)\b",
                cleaned,
                re.IGNORECASE,
            )
            or _TASK_META_RE.search(cleaned)
            or not cleaned
        ):
            return (
                "I'm sorry you're carrying this. It makes sense that this "
                "would feel heavy, and we can take it one step at a time. "
                "What is one small thing that might make the next hour a "
                "little easier?"
            )
        if task == "esconv":
            empathy_re = re.compile(
                r"\b(sorry|understand|makes sense|hard|heavy|painful|"
                r"overwhelming|valid|not alone)\b",
                re.IGNORECASE,
            )
            concrete_re = re.compile(
                r"\b(one small|next hour|right now|today|try|step|"
                r"breath(?:e|ing)?|grounding|write|journal|walk|rest|"
                r"plan|call|text)\b",
                re.IGNORECASE,
            )
            generic_re = re.compile(
                r"\b(how are you|anything i can help|what seems to be|"
                r"tell me more|how do you feel about)\b",
                re.IGNORECASE,
            )
            is_generic = bool(generic_re.search(cleaned))
            if is_generic or not empathy_re.search(cleaned):
                cleaned = (
                    "I'm sorry you're carrying this. It makes sense that this "
                    f"would feel hard. {cleaned}"
                )
            if is_generic or not concrete_re.search(cleaned):
                cleaned = (
                    f"{cleaned} For one small step right now, try taking "
                    "three slow breaths and writing down the one worry that "
                    "needs attention first. Would you be willing to try that "
                    "today?"
                )
        return cleaned

    cleaned = re.sub(r"\bhttps?://\S*URL\S*\b", "the official Save the Children website", raw)
    cleaned = re.sub(r"\bURL\b", "the official Save the Children website", cleaned)

    user_refused = _last_user_refused_donation(pre_history)
    recent_user_refused = _recent_user_refused_donation(pre_history)
    if user_refused or recent_user_refused:
        if (
            _COERCIVE_ASSIGN_RE.search(cleaned)
            or _DONATION_DECISION_RE.search(cleaned)
            or _TASK_META_RE.search(cleaned)
        ):
            return (
                "I understand and will not choose an amount for you. "
                "Thank you for considering Save the Children."
            )
        return cleaned

    if _COERCIVE_ASSIGN_RE.search(cleaned):
        return (
            "I cannot choose an amount for you. Would you be willing to donate "
            "a small non-zero amount, such as $0.25, from your task payment to "
            "Save the Children?"
        )

    if _TASK_META_RE.search(cleaned):
        return (
            "Before we finish, would you be willing to donate a small non-zero "
            "amount, such as $0.25, from your task payment to Save the Children?"
        )

    if _GOODBYE_RE.search(cleaned) and not _DONATION_DECISION_RE.search(cleaned):
        return (
            "Before we finish, would you be willing to donate a small non-zero "
            "amount, such as $0.25, from your task payment to Save the Children?"
        )

    return cleaned


def _replace_policy_response_text(policy: LoRAPolicy, out: Dict, text: str) -> Dict:
    if text == out.get("text"):
        return out
    enc = policy.tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    new_ids = enc["input_ids"][0].to(policy.device)
    updated = dict(out)
    updated["text"] = text
    updated["response_ids"] = new_ids.detach()
    return updated


@dataclass
class RolloutConfig:
    num_episodes: int = 16
    rollout_batch_size: int = 4
    max_turns: int = 8
    gamma: float = 0.97
    advantage_norm: bool = True
    verbose: bool = True
    log_every: int = 1
    max_new_tokens_system: int = 80
    max_new_tokens_user: int = 80
    temperature_system: float = 0.9
    temperature_user: float = 0.9
    top_p: float = 0.95
    task_name: str = "p4g"


class SelfPlayRollout:
    def __init__(
        self,
        env: POBGDialogueEnv,
        pi_S: LoRAPolicy,
        pi_U: LoRAPolicy,
        cfg: RolloutConfig,
    ):
        self.env = env
        self.pi_S = pi_S
        self.pi_U = pi_U
        self.cfg = cfg
        self._remote_executor = ThreadPoolExecutor(
            max_workers=max(1, int(self.cfg.rollout_batch_size))
        )

    def _spawn_env(self) -> POBGDialogueEnv:
        tmpl = self.env
        # Carry over teacher_mentalization / outcome_ensemble: without these
        # the spawned env silently falls back to OBU-as-teacher and skips
        # OGR, breaking two invariants: the frozen teacher must drive z* in
        # RL, and lambda_out wiring must be explicit.
        return POBGDialogueEnv(
            task_cfg=tmpl.task_cfg,
            mind_prior=tmpl.mind_prior,
            sentence_encoder=tmpl.sentence_encoder,
            mentalization=tmpl.mentalization,
            bdi_extractor=tmpl.bdi_extractor,
            p4g_judge=tmpl.p4g_judge,
            rationality_judge=tmpl.rationality_judge,
            reward_cfg=tmpl.cfg,
            max_turns=tmpl.max_turns,
            parallel_env_calls=tmpl.parallel_env_calls,
            env_call_workers=tmpl.env_call_workers,
            teacher_mentalization=tmpl.teacher_mentalization,
            outcome_ensemble=tmpl.outcome_ensemble,
            outcome_reward_cfg=tmpl.outcome_reward_cfg,
            local_obu_feedback=tmpl.local_obu_feedback,
            local_rationality_feedback=tmpl.local_rationality_feedback,
            precomputed_z_goal=tmpl._z_goal,
            precomputed_prior_entries=tmpl._prior_entries,
            precomputed_prior_z=tmpl._prior_z,
        )

    def _infer_bdi_hints_batch(
        self,
        envs: List[POBGDialogueEnv],
        history_batch: List[List[str]],
    ) -> List[str]:
        if not envs:
            return []
        if envs[0]._prior_z.shape[0] == 0:
            return ["Belief: Unknown.\nDesire: Unknown.\nIntention: Unknown."] * len(envs)

        texts = ["\n".join(h) if h else "(no turns yet)" for h in history_batch]
        with torch.no_grad():
            z_hat_batch = envs[0].mentalization.forward(texts).detach().float()

        prior_z = envs[0]._prior_z
        q = F.normalize(z_hat_batch.to(prior_z.device), p=2, dim=-1)
        k = F.normalize(prior_z, p=2, dim=-1)
        sims = torch.matmul(q, k.transpose(0, 1))
        idxs = torch.argmax(sims, dim=-1).tolist()

        hints: List[str] = []
        for env, hist, z_hat, idx in zip(envs, history_batch, z_hat_batch, idxs):
            env._cached_hist_key = env._hist_key(hist)
            env._cached_z_hat = z_hat.detach()
            hints.append(env._prior_entries[int(idx)].bdi.to_text())
        return hints

    @staticmethod
    def _finalize_episode(
        steps: List[TrajectoryStep],
        turn_rewards_S: List[float],
        turn_rewards_U: List[float],
        gamma: float,
        advantage_norm: bool,
    ) -> None:
        returns_S = TrajectoryBuffer.compute_monte_carlo_returns(turn_rewards_S, gamma)
        returns_U = TrajectoryBuffer.compute_monte_carlo_returns(turn_rewards_U, gamma)

        if advantage_norm:
            adv_S = TrajectoryBuffer.normalize(returns_S)
            adv_U = TrajectoryBuffer.normalize(returns_U)
        else:
            adv_S = list(returns_S)
            adv_U = list(returns_U)

        idx_s = 0
        idx_u = 0
        for st in steps:
            if st.role == "system":
                st.ret = float(returns_S[idx_s])
                st.advantage = float(adv_S[idx_s])
                idx_s += 1
            else:
                st.ret = float(returns_U[idx_u])
                st.advantage = float(adv_U[idx_u])
                idx_u += 1

    # ------------------------------------------------------- one-episode
    def run_episode(
        self,
        prior_entry: MindPriorEntry = None,
        collect_S: bool = True,
        collect_U: bool = True,
        collect_ment: bool = False,
    ) -> Tuple[List[TrajectoryStep], Dict]:
        """
        Run one full dialogue. Returns:
            steps: all per-turn trajectory steps (system + user) with rewards
            info : diagnostic dict
        """
        self.pi_S.eval_mode()
        self.pi_U.eval_mode()

        obs = self.env.reset(prior_entry=prior_entry)
        steps: List[TrajectoryStep] = []
        turn_rewards_S: List[float] = []
        turn_rewards_U: List[float] = []
        turn_bundles: List[RewardBundle] = []
        ment_samples: List[Dict] = []

        while not self.env.done and self.env.turn_idx < self.cfg.max_turns:
            pre_history = list(obs["history_lines"])
            if collect_ment and self.env.z_star_prev is not None:
                history_text = "\n".join(pre_history) if pre_history else "(no turns yet)"
                ment_samples.append({
                    "history_text": history_text,
                    "z_target": self.env.z_star_prev.detach().cpu(),
                })

            # ----- 1. System generates its turn
            with torch.no_grad():
                # Convert current latent estimate ẑ_t into a natural-language
                # BDI hint so the policy is explicitly conditioned on its
                # own mentalization output.
                belief_hint_text = self.env.infer_bdi_hint_text(pre_history)
                sys_msgs = build_chat_prompt_for_policy(
                    role="assistant",
                    history_lines=pre_history,
                    belief_hint_text=belief_hint_text,
                    task_name=self.cfg.task_name,
                )
                s_out = self.pi_S.generate(
                    sys_msgs,
                    max_new_tokens=self.cfg.max_new_tokens_system,
                    temperature=self.cfg.temperature_system,
                    top_p=self.cfg.top_p,
                    do_sample=True,
                )
                system_turn = _postprocess_system_turn(
                    s_out["text"], pre_history, task_name=self.cfg.task_name
                )
                s_out = _replace_policy_response_text(self.pi_S, s_out, system_turn)

            # Compute the old log-prob of this response under π_S for PPO.
            with torch.no_grad():
                s_logp, _ = self.pi_S.log_probs_of_response(sys_msgs, s_out["response_ids"])
                s_logp = s_logp.detach().cpu()

            # ----- 2. User generates its turn (conditioned on committed BDI)
            with torch.no_grad():
                user_hist_lines = list(pre_history) + [f"Assistant: {system_turn}"]
                bdi = self.env.current_bdi
                bdi_text = bdi.to_text() if bdi is not None else ""
                usr_msgs = build_chat_prompt_for_policy(
                    role="user",
                    history_lines=user_hist_lines,
                    bdi_text=bdi_text,
                    task_name=self.cfg.task_name,
                )
                u_out = self.pi_U.generate(
                    usr_msgs,
                    max_new_tokens=self.cfg.max_new_tokens_user,
                    temperature=self.cfg.temperature_user,
                    top_p=self.cfg.top_p,
                    do_sample=True,
                )
                user_turn = u_out["text"]

            with torch.no_grad():
                u_logp, _ = self.pi_U.log_probs_of_response(usr_msgs, u_out["response_ids"])
                u_logp = u_logp.detach().cpu()

            # ----- 3. Environment step (BDI update + rewards)
            obs, bundle = self.env.step(system_turn=system_turn, user_turn=user_turn)
            turn_rewards_S.append(bundle.r_system)
            turn_rewards_U.append(bundle.r_user)
            turn_bundles.append(bundle)

            if collect_S:
                steps.append(TrajectoryStep(
                    role="system",
                    messages=sys_msgs,
                    response_ids=s_out["response_ids"].detach().cpu(),
                    old_logp=s_logp,
                    advantage=0.0,  # filled in below
                    ret=0.0,
                    reward=bundle.r_system,
                ))
            if collect_U:
                steps.append(TrajectoryStep(
                    role="user",
                    messages=usr_msgs,
                    response_ids=u_out["response_ids"].detach().cpu(),
                    old_logp=u_logp,
                    advantage=0.0,
                    ret=0.0,
                    reward=bundle.r_user,
                ))

        # ----- 4. Compute returns / advantages per role
        returns_S = TrajectoryBuffer.compute_monte_carlo_returns(turn_rewards_S, self.cfg.gamma)
        returns_U = TrajectoryBuffer.compute_monte_carlo_returns(turn_rewards_U, self.cfg.gamma)

        if self.cfg.advantage_norm:
            adv_S = TrajectoryBuffer.normalize(returns_S)
            adv_U = TrajectoryBuffer.normalize(returns_U)
        else:
            adv_S = list(returns_S)
            adv_U = list(returns_U)

        # Steps are appended in pairs (system, user). We walk in turn order and
        # assign advantages respectively.
        idx_s = 0
        idx_u = 0
        for st in steps:
            if st.role == "system":
                st.ret = float(returns_S[idx_s])
                st.advantage = float(adv_S[idx_s])
                idx_s += 1
            else:
                st.ret = float(returns_U[idx_u])
                st.advantage = float(adv_U[idx_u])
                idx_u += 1

        info = {
            "success": bool(self.env.success),
            "turns": int(self.env.turn_idx),
            "ep_reward_S": float(sum(turn_rewards_S)),
            "ep_reward_U": float(sum(turn_rewards_U)),
            "bundles": turn_bundles,
            "ment_samples": ment_samples,
        }
        return steps, info

    # ------------------------------------------------------- batch rollout
    def run_batch(
        self,
        n_episodes: int,
        collect_S: bool = True,
        collect_U: bool = True,
        collect_ment: bool = False,
        phase_name: str = "",
    ) -> Tuple[TrajectoryBuffer, DialogMetrics, List[Dict]]:
        buf = TrajectoryBuffer()
        metrics = DialogMetrics()
        ment_samples: List[Dict] = []
        n_episodes = int(n_episodes)
        batch_size = max(1, int(self.cfg.rollout_batch_size))
        start_batch = time.perf_counter()
        tag = phase_name or "rollout"
        if self.cfg.verbose:
            print(
                f"[rollout:{tag}] start episodes={n_episodes} rollout_batch_size={batch_size}",
                flush=True,
            )
        spawn_start = time.perf_counter()
        envs = [self._spawn_env() for _ in range(n_episodes)]
        if self.cfg.verbose:
            print(
                f"[rollout:{tag}] spawn episodes={n_episodes} "
                f"cost={time.perf_counter() - spawn_start:.1f}s",
                flush=True,
            )

        # Batched reset: the frozen teacher F_omega used to be called once per
        # episode here, which created a long silent stall before the first
        # rollout chunk. Compute all z_init values in one forward pass instead.
        init_start = time.perf_counter()
        prior_entries = [self.env.mind_prior.sample() for _ in range(n_episodes)]
        z_init_batch = None
        z_init_text_emb_batch = None
        if envs:
            tmpl_env = envs[0]
            if tmpl_env.teacher_mentalization is not None:
                with torch.no_grad():
                    z_init_batch = tmpl_env.teacher_mentalization.forward_with_context(
                        history_texts=["(no turns yet)"] * n_episodes,
                        profile_texts=[entry.profile_text or "" for entry in prior_entries],
                        goal_text=tmpl_env._teacher_goal_text,
                    ).detach()
            with torch.no_grad():
                z_init_text_emb_batch = tmpl_env.sentence_encoder.encode(
                    [entry.bdi.to_text() for entry in prior_entries]
                ).detach().float()

        states: List[Dict] = []
        for i, env in enumerate(envs):
            states.append({
                "obs": env.reset(
                    prior_entry=prior_entries[i],
                    z_init_override=(z_init_batch[i] if z_init_batch is not None else None),
                    z_init_text_emb_override=(
                        z_init_text_emb_batch[i] if z_init_text_emb_batch is not None else None
                    ),
                ),
                "steps": [],
                "turn_rewards_S": [],
                "turn_rewards_U": [],
                "turn_bundles": [],
                "ment_samples": [],
            })
        if self.cfg.verbose:
            print(
                f"[rollout:{tag}] init episodes={n_episodes} "
                f"cost={time.perf_counter() - init_start:.1f}s",
                flush=True,
            )

        chunk_idx = 0
        while True:
            active = [i for i, env in enumerate(envs) if not env.done and env.turn_idx < self.cfg.max_turns]
            if not active:
                break

            for start in range(0, len(active), batch_size):
                chunk_idx += 1
                chunk_start = time.perf_counter()
                batch_idxs = active[start:start + batch_size]
                batch_envs = [envs[i] for i in batch_idxs]
                batch_states = [states[i] for i in batch_idxs]
                pre_histories = [list(st["obs"]["history_lines"]) for st in batch_states]

                if collect_ment:
                    for env, st, pre_history in zip(batch_envs, batch_states, pre_histories):
                        if env.z_star_prev is not None:
                            history_text = "\n".join(pre_history) if pre_history else "(no turns yet)"
                            st["ment_samples"].append({
                                "history_text": history_text,
                                "z_target": env.z_star_prev.detach().cpu(),
                            })

                t_ment_0 = time.perf_counter()
                belief_hints = self._infer_bdi_hints_batch(batch_envs, pre_histories)
                ment_cost = time.perf_counter() - t_ment_0
                sys_msgs_batch = [
                    build_chat_prompt_for_policy(
                        role="assistant",
                        history_lines=pre_history,
                        belief_hint_text=belief_hint,
                        task_name=self.cfg.task_name,
                    )
                    for pre_history, belief_hint in zip(pre_histories, belief_hints)
                ]
                t_sys_gen_0 = time.perf_counter()
                s_outs = self.pi_S.generate_batch(
                    sys_msgs_batch,
                    max_new_tokens=self.cfg.max_new_tokens_system,
                    temperature=self.cfg.temperature_system,
                    top_p=self.cfg.top_p,
                    do_sample=True,
                )
                s_outs = [
                    _replace_policy_response_text(
                        self.pi_S,
                        s_out,
                        _postprocess_system_turn(
                            s_out["text"],
                            pre_history,
                            task_name=self.cfg.task_name,
                        ),
                    )
                    for pre_history, s_out in zip(pre_histories, s_outs)
                ]
                sys_gen_cost = time.perf_counter() - t_sys_gen_0
                t_sys_logp_0 = time.perf_counter()
                s_logps = self.pi_S.log_probs_of_responses_batch(
                    sys_msgs_batch,
                    [out["response_ids"] for out in s_outs],
                )
                sys_logp_cost = time.perf_counter() - t_sys_logp_0

                user_histories = [
                    list(pre_history) + [f"Assistant: {s_out['text']}"]
                    for pre_history, s_out in zip(pre_histories, s_outs)
                ]
                usr_msgs_batch = [
                    build_chat_prompt_for_policy(
                        role="user",
                        history_lines=user_history,
                        bdi_text=(env.current_bdi.to_text() if env.current_bdi is not None else ""),
                        task_name=self.cfg.task_name,
                    )
                    for env, user_history in zip(batch_envs, user_histories)
                ]
                t_usr_gen_0 = time.perf_counter()
                u_outs = self.pi_U.generate_batch(
                    usr_msgs_batch,
                    max_new_tokens=self.cfg.max_new_tokens_user,
                    temperature=self.cfg.temperature_user,
                    top_p=self.cfg.top_p,
                    do_sample=True,
                )
                usr_gen_cost = time.perf_counter() - t_usr_gen_0
                t_usr_logp_0 = time.perf_counter()
                u_logps = self.pi_U.log_probs_of_responses_batch(
                    usr_msgs_batch,
                    [out["response_ids"] for out in u_outs],
                )
                usr_logp_cost = time.perf_counter() - t_usr_logp_0

                remote_start = time.perf_counter()
                remote_futures = [
                    self._remote_executor.submit(
                        env.query_remote_feedback,
                        s_out["text"],
                        u_out["text"],
                        pre_history,
                    )
                    for env, pre_history, s_out, u_out in zip(
                        batch_envs, pre_histories, s_outs, u_outs
                    )
                ]
                remote_results = [f.result() for f in remote_futures]
                remote_cost = time.perf_counter() - remote_start

                t_local_0 = time.perf_counter()
                for env, st, pre_history, sys_msgs, s_out, s_logp, usr_msgs, u_out, u_logp, remote in zip(
                    batch_envs,
                    batch_states,
                    pre_histories,
                    sys_msgs_batch,
                    s_outs,
                    s_logps,
                    usr_msgs_batch,
                    u_outs,
                    u_logps,
                    remote_results,
                ):
                    bdi_star, rat, sig = remote
                    obs, bundle = env.step_with_feedback(
                        system_turn=s_out["text"],
                        user_turn=u_out["text"],
                        bdi_star=bdi_star,
                        rat=rat,
                        sig=sig,
                        pre_history=pre_history,
                    )
                    st["obs"] = obs
                    st["turn_rewards_S"].append(bundle.r_system)
                    st["turn_rewards_U"].append(bundle.r_user)
                    st["turn_bundles"].append(asdict(bundle))

                    if collect_S:
                        st["steps"].append(TrajectoryStep(
                            role="system",
                            messages=sys_msgs,
                            response_ids=s_out["response_ids"].detach().cpu(),
                            old_logp=s_logp.detach().cpu(),
                            advantage=0.0,
                            ret=0.0,
                            reward=bundle.r_system,
                        ))
                    if collect_U:
                        st["steps"].append(TrajectoryStep(
                            role="user",
                            messages=usr_msgs,
                            response_ids=u_out["response_ids"].detach().cpu(),
                            old_logp=u_logp.detach().cpu(),
                            advantage=0.0,
                            ret=0.0,
                            reward=bundle.r_user,
                        ))
                local_cost = time.perf_counter() - t_local_0
                chunk_cost = time.perf_counter() - chunk_start

                if self.cfg.verbose and (chunk_idx % max(int(self.cfg.log_every), 1) == 0):
                    done_eps = sum(1 for env in envs if env.done or env.turn_idx >= self.cfg.max_turns)
                    mean_turns = sum(env.turn_idx for env in envs) / max(len(envs), 1)
                    print(
                        f"[rollout:{tag}] chunk={chunk_idx} active={len(active)} "
                        f"done={done_eps}/{n_episodes} mean_turns={mean_turns:.2f} "
                        f"ment={ment_cost:.1f}s sys_gen={sys_gen_cost:.1f}s "
                        f"sys_logp={sys_logp_cost:.1f}s usr_gen={usr_gen_cost:.1f}s "
                        f"usr_logp={usr_logp_cost:.1f}s remote={remote_cost:.1f}s "
                        f"local={local_cost:.1f}s chunk_elapsed={chunk_cost:.1f}s "
                        f"elapsed={time.perf_counter() - start_batch:.1f}s",
                        flush=True,
                    )

        for local_episode_idx, (env, st) in enumerate(zip(envs, states)):
            self._finalize_episode(
                steps=st["steps"],
                turn_rewards_S=st["turn_rewards_S"],
                turn_rewards_U=st["turn_rewards_U"],
                gamma=self.cfg.gamma,
                advantage_norm=self.cfg.advantage_norm,
            )
            for s in st["steps"]:
                buf.add(s)
            prior_entry = getattr(env, "_episode_prior_entry", None)
            initial_bdi = ""
            prior_session_id = ""
            profile_text = ""
            if prior_entry is not None:
                prior_session_id = str(getattr(prior_entry, "session_id", ""))
                profile_text = str(getattr(prior_entry, "profile_text", ""))
                if getattr(prior_entry, "bdi", None) is not None:
                    initial_bdi = prior_entry.bdi.to_text()
            final_bdi = env.current_bdi.to_text() if env.current_bdi is not None else ""
            trace = {
                "episode_index": int(local_episode_idx),
                "prior_session_id": prior_session_id,
                "profile_text": profile_text,
                "initial_bdi": initial_bdi,
                "final_bdi": final_bdi,
                "success": bool(env.success),
                "turns": int(env.turn_idx),
                "total_reward_system": float(sum(st["turn_rewards_S"])),
                "total_reward_user": float(sum(st["turn_rewards_U"])),
                "turn_rewards_system": [float(x) for x in st["turn_rewards_S"]],
                "turn_rewards_user": [float(x) for x in st["turn_rewards_U"]],
                "reward_bundles": list(st["turn_bundles"]),
                "history_lines": list(env.history_lines),
            }
            metrics.add(
                success=env.success,
                turns=env.turn_idx,
                reward=float(sum(st["turn_rewards_S"])),
                trace=trace,
            )
            if collect_ment:
                ment_samples.extend(st["ment_samples"])
        if self.cfg.verbose:
            print(
                f"[rollout:{tag}] done episodes={n_episodes} "
                f"elapsed={time.perf_counter() - start_batch:.1f}s",
                flush=True,
            )
        return buf, metrics, ment_samples
