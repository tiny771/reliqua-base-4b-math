"""Miner Engine for Reliquary Subnet 81 (Forced-Seed Protocol v2).

Implements auction-mode mining with deterministic rollout generation via vLLM's
ForcedSeedLogitsProcessor. Uses SHA256-derived uniform draws computed per-token
for bit-identical sampling that prevents variance farming and pre-generation.

Key features:
- Concurrent 8-rollout generation with early-abort on failures
- Two-stage quality gating for prompt frontier optimization
- GRAIL sketch commitments for cryptographic authenticity proof
- Precommit/reveal submission flow for auction fairness
- Split generation and proof/submission pipelines for maximum GPU utilization
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random as _random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from reliquary.constants import (
    BFT_FORCE_TEMPLATE,
    FORCED_SEED_PROTOCOL_VERSION,
    LAYER_INDEX,
    M_ROLLOUTS,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    MIN_EOS_PROBABILITY,
    POLL_INTERVAL_SECONDS,
    PROMPT_RANGE_SIZE,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
)
from reliquary.infrastructure import chain
from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, to_generation_contract
from reliquary.protocol.signatures import sign_envelope
from reliquary.protocol.submission import BatchSubmissionRequest, RolloutSubmission
from reliquary.shared.modeling import force_close_token_ids
from reliquary.shared.prompt_range import window_prompt_range
from reliquary.validator.verifier import rewards_std

if TYPE_CHECKING:
    from reliquary.environment.base import Environment

from reliquary.miner_code.utils import GenerationResult, VLLMGenerator, _eval_difficulty

EARLY_STOP_GENERATION = True  # Abort rollouts on malformed/suspicious patterns
TEST_MODE = False

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ====================== Helpers ======================
async def maybe_pull_checkpoint(
    state, local_n, local_hash, local_model, load_fn, download_fn, lock
):
    async with lock:
        if state.checkpoint_n <= local_n or not getattr(
            state, "checkpoint_repo_id", None
        ):
            return local_n, local_hash, local_model
        logger.info(f"🔄 New checkpoint n={state.checkpoint_n}")
        path = await download_fn(state.checkpoint_repo_id, state.checkpoint_revision)
        model, success = await load_fn(path)
        return (
            (state.checkpoint_n, state.checkpoint_revision, model)
            if success
            else (local_n, local_hash, local_model)
        )


async def _hf_download(repo_id: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    from reliquary.shared.modeling import MODEL_SNAPSHOT_ALLOW_PATTERNS

    start = time.time()
    path = await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id,
        revision=revision,
        allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS,
    )
    logger.debug(f"📥 Downloaded checkpoint in {time.time()-start:.1f}s")
    return path


# Score cache to avoid repeated difficulty evaluation across prompt selection rounds.
_prompt_difficulty_cache: Dict[int, float] = {}
_prompt_state_cache: Dict[int, bool] = {}


def select_prompts(
    env,
    cooldown: set,
    selected: set,
    prompt_range: tuple,
    count: int = 8,
    difficulty_range: tuple[float, float] | None = None,
):
    lo, hi = prompt_range
    eligible_random = [i for i in range(lo, hi) if i not in cooldown]
    eligible = [j for j in eligible_random if j not in selected]
    if not eligible:
        return [], []

    count = min(count, len(eligible))
    sample_size = min(len(eligible), max(count * 3, 120))
    candidate_indices = (
        _random.sample(eligible, sample_size)
        if len(eligible) > sample_size
        else eligible[:]
    )

    candidate_problems = {idx: env.get_problem(idx) for idx in candidate_indices}

    to_score = [idx for idx in candidate_indices if idx not in _prompt_difficulty_cache]
    if to_score:
        with ThreadPoolExecutor(max_workers=min(16, len(to_score))) as executor:
            futures = {
                executor.submit(_eval_difficulty, candidate_problems[idx]): idx
                for idx in to_score
            }
            for future in futures:
                idx = futures[future]
                try:
                    state, diff = future.result()
                    _prompt_difficulty_cache[idx] = diff
                    _prompt_state_cache[idx] = state
                except Exception as exc:
                    logger.warning(f"Failed to score prompt {idx}: {exc}")
                    _prompt_difficulty_cache[idx] = 0.0
                    _prompt_state_cache[idx] = False

    if difficulty_range is not None:
        lo_diff, hi_diff = difficulty_range
        filtered = [
            idx
            for idx in candidate_indices
            if lo_diff <= _prompt_difficulty_cache.get(idx, 0.0) < hi_diff
            and _prompt_state_cache.get(idx, False) == False
        ]

        if len(filtered) < count:
            remaining = [idx for idx in eligible if idx not in candidate_indices]
            while len(filtered) < count and remaining:
                add_batch = _random.sample(
                    remaining, min(len(remaining), max(count, 16))
                )
                candidate_indices.extend(add_batch)
                candidate_indices = list(dict.fromkeys(candidate_indices))
                candidate_problems.update(
                    {idx: env.get_problem(idx) for idx in add_batch}
                )
                to_score = [
                    idx for idx in add_batch if idx not in _prompt_difficulty_cache
                ]
                if to_score:
                    with ThreadPoolExecutor(
                        max_workers=min(16, len(to_score))
                    ) as executor:
                        futures = {
                            executor.submit(
                                _eval_difficulty, candidate_problems[idx]
                            ): idx
                            for idx in to_score
                        }
                        for future in futures:
                            idx = futures[future]
                            try:
                                state, diff = future.result()
                                _prompt_difficulty_cache[idx] = diff
                                _prompt_state_cache[idx] = state
                            except Exception as exc:
                                logger.warning(f"Failed to score prompt {idx}: {exc}")
                                _prompt_difficulty_cache[idx] = 0.0
                                _prompt_state_cache[idx] = False
                filtered = [
                    idx
                    for idx in candidate_indices
                    if lo_diff <= _prompt_difficulty_cache.get(idx, 0.0) < hi_diff
                    and _prompt_state_cache.get(idx, False) == False
                ]
                remaining = [idx for idx in eligible if idx not in candidate_indices]

        selected_indices = filtered[:count]
    else:
        selected_indices = candidate_indices[:count]

    if not selected_indices:
        logger.warning(
            f"No prompts found in difficulty range {difficulty_range} for range {lo}-{hi}."
        )
        return [], []

    problems = []
    for idx in selected_indices:
        problem = dict(candidate_problems[idx])
        problem["difficulty"] = _prompt_difficulty_cache.get(idx, 0.0)
        problems.append(problem)

    logger.info(
        f"🎲 Selected {len(selected_indices)} prompts from range {lo}-{hi} "
        f"difficulty_range={difficulty_range}"
    )
    return selected_indices, problems


def _compute_merkle_root(rollouts):
    import hashlib
    import json

    leaves = []
    for i, r in enumerate(rollouts):
        h = hashlib.sha256()
        h.update(i.to_bytes(8, "big"))
        h.update(json.dumps(r.tokens, separators=(",", ":")).encode())
        h.update(str(r.reward).encode())
        h.update(json.dumps(r.commit, sort_keys=True, separators=(",", ":")).encode())
        leaves.append(h.digest())

    while len(leaves) > 1:
        leaves = [
            hashlib.sha256(
                leaves[i] + (leaves[i + 1] if i + 1 < len(leaves) else leaves[i])
            ).digest()
            for i in range(0, len(leaves), 2)
        ]
    return leaves[0].hex()


def _current_drand_round_at_send() -> int:
    from reliquary.infrastructure.chain import compute_current_drand_round
    from reliquary.infrastructure.drand import get_current_chain

    ci = get_current_chain()
    return compute_current_drand_round(time.time(), ci["genesis_time"], ci["period"])


def _build_vllm_extra_body(
    *,
    prompt_len: int,
    prompt_idx: int,
    rollout_idx: int,
    randomness: str,
    checkpoint_hash: str,
    base_offset: int = 0,
    start_len: int | None = None,
) -> Dict[str, object]:
    """Build extra_body parameters for ForcedSeedLogitsProcessor."""
    return {
        "randomness": randomness,
        "prompt_idx": prompt_idx,
        "checkpoint_hash": checkpoint_hash,
        "rollout_index": rollout_idx,
        "base_offset": base_offset,
        "temperature": T_PROTO,
        "top_k": TOP_K_PROTO,
        "top_p": TOP_P_PROTO,
    }


# ====================== MiningEngine ======================
class MiningEngine:
    def __init__(
        self,
        vllm_url,
        hf_model,
        tokenizer,
        wallet,
        env=None,
        proof_gpu=0,
        max_new_tokens=MAX_NEW_TOKENS_PROTOCOL_CAP,
        validator_url_override=None,
        max_concurrent=200,
        difficulty_range: tuple[float, float] | None = None,
    ):
        self.vllm_url = vllm_url
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.wallet = wallet
        self.env = env
        self.proof_gpu = proof_gpu
        self.max_new_tokens = max_new_tokens
        self.validator_url_override = validator_url_override
        self._difficulty_range = (0, 20)
        self._n_candidates = 100

        self._cooldown: Set[int] = set()
        self._selected: Set[int] = set()
        self._prompt_range = (0, len(env) if env else 0)

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._checkpoint_lock = asyncio.Lock()

        self._local_checkpoint_n = 0
        self._local_checkpoint_hash = ""

        from reliquary.protocol.grail_verifier import GRAILVerifier
        from reliquary.shared.hf_compat import resolve_hidden_size

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

        self._preflight_perplexity_threshold: float | None = None
        self._results_dir = Path(__file__).resolve().parents[3] / "results"
        self._analysis_log_path = self._results_dir / "miner_analysis.jsonl"
        self._submission_log_path = self._results_dir / "submission_results.jsonl"
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._vllm_client = VLLMGenerator(base_url=vllm_url, model_name="./models/glm")
        logger.info(
            f"🚀 Miner ready | concurrency={max_concurrent} | results_dir={self._results_dir}"
        )

    async def mine_window(self, subtensor):
        import httpx

        from reliquary.miner.submitter import (
            discover_validator_url,
        )
        from reliquary.protocol.submission import WindowState

        subtensor = await chain.get_subtensor()

        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)

        async with httpx.AsyncClient(timeout=240) as client:
            # Queue for passing completed rollouts to the proof/submission worker
            queue = asyncio.Queue()
            # Start the background worker for proof building and submission
            worker_task = asyncio.create_task(
                self._proof_and_submit_worker(queue, client, url)
            )

            active_tasks = {}
            last_state = WindowState.OPEN
            state = None

            while True:
                last_state = state.state if state is not None else WindowState.READY
                state = await self._sync_state(client, url)
                if (
                    not state
                    or state.state != WindowState.OPEN
                    or not state.randomness
                    or TEST_MODE
                ):
                    if last_state == WindowState.OPEN:
                        for t in active_tasks:
                            t.cancel()
                        active_tasks.clear()
                        await self._vllm_client.cancel_all_requests()
                        self._selected = set()

                    logger.info(f"⏳ Window Not Acitve Yet...")
                    await asyncio.sleep(0.5)
                    continue

                if last_state != WindowState.OPEN:
                    logger.info(f"🧭 Window {state.window_n} Started")

                needed = self._n_candidates - len(active_tasks)
                if needed > 0:
                    prompt_idxs, problems = select_prompts(
                        self.env,
                        self._cooldown,
                        self._selected,
                        self._prompt_range,
                        count=needed,
                        difficulty_range=self._difficulty_range,
                    )

                    for idx, prob in zip(prompt_idxs, problems):
                        diff = prob.get("difficulty", _eval_difficulty(prob)[0])
                        self._selected.add(idx)

                        task = asyncio.create_task(
                            self._process_prompt_pipeline(
                                prob,
                                idx,
                                diff,
                                state.randomness,
                                state.window_n,
                                client,
                                url,
                                state,
                                queue,
                            )
                        )

                        active_tasks[task] = idx

                        logger.info(
                            f"🚀 Add new prompt prompt_index={idx} difficulty={diff}"
                        )

                if not active_tasks:
                    await asyncio.sleep(1.0)
                    self._selected.clear()
                    continue

                done, _ = await asyncio.wait(
                    active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    idx = active_tasks.pop(task)

                    try:
                        task.result()
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"⛔ Prompt task {idx} failed: {e}")

    async def _proof_and_submit_worker(self, queue: asyncio.Queue, client, url):
        """Background worker that watches the queue, builds GRAIL proofs, and submits batches."""
        pending_batches = {}  # prompt_idx -> dict of batch info

        while True:
            try:
                item = await queue.get()
                if item is None:  # Poison pill for shutdown
                    break

                msg_type = item[0]
                if msg_type == "abort":
                    _, prompt_idx = item
                    pending_batches.pop(prompt_idx, None)
                    continue

                (
                    _,
                    prompt_idx,
                    window_n,
                    state,
                    problem,
                    randomness,
                    gen_result,
                    reward,
                ) = item

                # Run heavy GRAIL proof building in a separate thread to not block the event loop
                submission = await asyncio.to_thread(
                    self._build_rollout_submission,
                    gen_result,
                    problem,
                    randomness,
                    reward,
                )

                if prompt_idx not in pending_batches:
                    pending_batches[prompt_idx] = {
                        "submissions": [],
                        "window_n": window_n,
                        "state": state,
                        "randomness": randomness,
                        "client": client,
                        "url": url,
                    }

                pending_batches[prompt_idx]["submissions"].append(submission)

                # Once we have all rollouts for this prompt, submit the batch
                if len(pending_batches[prompt_idx]["submissions"]) == M_ROLLOUTS:
                    batch_info = pending_batches.pop(prompt_idx)
                    asyncio.create_task(
                        self._submit(
                            batch_info["submissions"],
                            prompt_idx,
                            batch_info["randomness"],
                            batch_info["window_n"],
                            batch_info["state"],
                            batch_info["client"],
                            batch_info["url"],
                        )
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Worker error: {e}")

    async def _sync_state(self, client, url):
        from reliquary.miner.submitter import get_window_state_v2

        try:
            state = await get_window_state_v2(url, client=client)
            self._local_checkpoint_n, self._local_checkpoint_hash, self.hf_model = (
                await maybe_pull_checkpoint(
                    state,
                    self._local_checkpoint_n,
                    self._local_checkpoint_hash,
                    self.hf_model,
                    self._load_checkpoint,
                    _hf_download,
                    self._checkpoint_lock,
                )
            )
            try:
                env_state = await get_window_state_v2(
                    url, env=self.env.name, client=client
                )
                self._cooldown = set(env_state.cooldown_prompts)
                self._prompt_range = window_prompt_range(
                    state.randomness, self.env.name, len(self.env), PROMPT_RANGE_SIZE
                )
            except Exception:
                self._cooldown = set(getattr(state, "cooldown_prompts", []))
            return state
        except Exception as e:
            logger.warning(f"State sync failed: {e}")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            return None

    async def _process_prompt_pipeline(
        self, problem, prompt_idx, diff, randomness, window_n, client, url, state, queue
    ):
        async with self._semaphore:
            prompt, prompt_len = await asyncio.to_thread(
                self._format_problem, problem.get("prompt")
            )
            ground_truth = problem.get("ground_truth", "")

            # Launch all generation tasks concurrently
            pending = {
                asyncio.create_task(
                    self._generate_single_rollout(
                        prompt,
                        prompt_idx,
                        prompt_len,
                        rollout_idx,
                        randomness,
                        state.checkpoint_revision,
                        ground_truth,
                    )
                )
                for rollout_idx in range(0, M_ROLLOUTS)
            }

            passed_rollouts = []
            aborted = False

            # Process rollouts as they complete (FIRST_COMPLETED)
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    try:
                        result = task.result()
                    except Exception as exc:
                        logger.error(f"⚠️ Rollout task failed for #{prompt_idx}: {exc}")
                        result = None

                    # Check gate and abort if failed
                    if result is None or not self._passes_stage_two_gate(
                        result, prompt_idx
                    ):
                        aborted = True
                        for p in pending:
                            p.cancel()
                        break

                    # Calculate reward immediately upon passing gate
                    try:
                        reward = self.env.compute_reward(problem, result.text)
                    except Exception:
                        reward = 0.0

                    passed_rollouts.append((result, reward))

                if aborted:
                    break

            if aborted or len(passed_rollouts) < M_ROLLOUTS:
                logger.warning(f"⏭️ #{prompt_idx} -> aborted or incomplete")
                await queue.put(("abort", prompt_idx))
                await self._record_analysis_result(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "prompt_idx": prompt_idx,
                        "window_n": window_n,
                        "status": "rollout_incomplete",
                        "reason": (
                            "incomplete_rollouts" if not aborted else "gate_failed"
                        ),
                        "diff": diff,
                        "rollout_count": len(passed_rollouts),
                        "prompt_len": prompt_len,
                        "prompt_preview": problem.get("prompt", "")[:160],
                        "solution": problem.get("solution", ""),
                        "solution_len": len(problem.get("solution", "")),
                    }
                )
                return

            # Sigma gating (reward variance check)
            rewards = [r for _, r in passed_rollouts]
            sigma = rewards_std(rewards)

            if sigma < 0.433:
                logger.info(
                    f"⏭️ #{prompt_idx} -> SKIPPED | sigma={sigma:.3f} | rewards={rewards}"
                )
                await queue.put(("abort", prompt_idx))
                await self._record_analysis_result(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "prompt_idx": prompt_idx,
                        "window_n": window_n,
                        "status": "sigma_skipped",
                        "reason": "sigma_threshold",
                        "diff": diff,
                        "sigma": sigma,
                        "rewards": rewards,
                        "prompt_len": prompt_len,
                        "prompt_preview": problem.get("prompt", "")[:160],
                        "solution": problem.get("solution", ""),
                        "solution_len": len(problem.get("solution", "")),
                    }
                )
                return

            logger.info(f"💎 #{prompt_idx} -> PASSED | sigma={sigma:.3f}")

            # Put all passed rollouts into the queue for proof building and submission
            for gen_result, reward in passed_rollouts:
                await queue.put(
                    (
                        "rollout",
                        prompt_idx,
                        window_n,
                        state,
                        problem,
                        randomness,
                        gen_result,
                        reward,
                    )
                )

    def _passes_stage_two_gate(self, result, prompt_idx: int) -> bool:
        should_stop, has_malformed = self._should_stop_generation(result)
        if should_stop and EARLY_STOP_GENERATION:
            last_tokens = getattr(result, "tokens", None) or []
            last_logprob = (
                (getattr(result, "token_logprobs", None) or [])[-1]
                if getattr(result, "token_logprobs", None)
                else None
            )
            last_token_text = ""
            if last_tokens:
                last_token_text = self.tokenizer.decode([last_tokens[-1]])
            logger.warning(
                f"⚠️ #{prompt_idx} -> rejected "
                f"| rollout={getattr(result, 'rollout_idx', 0)} "
                f"| final_token={last_token_text} "
                f"| pstop={last_logprob} "
                f"| finish_reason={getattr(result, 'finish_reason', None)} "
                f"| malformed={has_malformed}"
            )
            return False

        return True

    def _should_stop_generation(self, rollout: GenerationResult) -> tuple[bool, bool]:
        finish_reason = str(getattr(rollout, "finish_reason", "") or "")
        if finish_reason != "stop" and finish_reason != "bft_length":
            return True, False

        tokens = getattr(rollout, "tokens", None) or []
        if not tokens:
            return True, True

        token_logprobs = getattr(rollout, "token_logprobs", None) or []
        if not token_logprobs:
            return True, True

        return False, False

    async def _generate_single_rollout(
        self,
        prompt: str,
        prompt_idx: int,
        prompt_len: int,
        rollout_idx: int,
        randomness: str,
        checkpoint_hash: str,
        ground_truth: str,
        max_tokens: int = 2048,
    ) -> Optional["GenerationResult"]:
        try:
            extra_body = _build_vllm_extra_body(
                prompt_len=prompt_len,
                prompt_idx=prompt_idx,
                rollout_idx=rollout_idx,
                randomness=randomness,
                checkpoint_hash=checkpoint_hash,
            )

            result = await self._vllm_client.generate_rollout_async(
                prompt,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )

            if not result or not getattr(result, "tokens", None):
                logger.warning(
                    "⚠️ Rollout #%d for #%d failed validation (empty or missing tokens)",
                    rollout_idx,
                    prompt_idx,
                )
                return None

            if result is not None:
                result.rollout_idx = rollout_idx

            return result

        except Exception as e:
            logger.exception(
                "⚠️ Rollout #%d failed for #%d: %s", rollout_idx, prompt_idx, e
            )
            return None

    async def _record_analysis_result(self, record: Dict):
        self._results_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._append_analysis_record, record)

    def _append_analysis_record(self, record: Dict):
        with self._analysis_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")

    async def _save_submission_result(self, record: Dict):
        self._results_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._append_submission_record, record)

    def _append_submission_record(self, record: Dict):
        with self._submission_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")

    def _format_problem(self, prompt: str) -> tuple[str, list[int]]:
        message = [{"role": "user", "content": prompt}]

        prompt_token_list = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=True,
        )

        formatted_prompt = self.tokenizer.decode(
            prompt_token_list, skip_special_tokens=False
        )

        return formatted_prompt, len(prompt_token_list["input_ids"])

    def _build_rollout_submission(self, gen, problem, randomness, reward):
        prompt_ids = gen.prompt_token_ids
        comp_ids = gen.tokens
        forced = gen.forced
        forced_span = gen.forced_span
        all_tokens = prompt_ids + comp_ids

        commit = self._build_grail_commit(
            all_tokens, len(prompt_ids), randomness, forced, forced_span
        )
        return RolloutSubmission(
            tokens=all_tokens, reward=reward, commit=commit, env_name=self.env.name
        )

    def _build_grail_commit(
        self, all_tokens, prompt_length, randomness, forced, forced_span
    ):
        start = time.time()
        proof_input = torch.tensor([all_tokens], device=f"cuda:{self.proof_gpu}")

        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding
        from reliquary.shared.forward import forward_single_layer

        try:
            with torch.no_grad():
                hidden, logits = forward_single_layer(
                    self.hf_model, proof_input, None, LAYER_INDEX
                )
        except Exception as e:
            logger.error(f"Proof generation failed: {e}")
            return None

        hidden_states = hidden[0]
        r_vec = self._verifier.generate_r_vec(randomness)
        commitments = self._verifier.create_commitments_batch(hidden_states, r_vec)

        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        token_logprobs = [
            log_probs[i - 1, all_tokens[i]].item()
            for i in range(prompt_length, len(all_tokens))
        ]

        model_name = getattr(self.hf_model, "name_or_path", "unknown")
        signature = sign_commit_binding(
            all_tokens, randomness, model_name, LAYER_INDEX, commitments, self.wallet
        )

        return {
            "tokens": all_tokens,
            "commitments": commitments,
            "proof_version": GRAIL_PROOF_VERSION,
            "model": {"name": model_name, "layer_index": LAYER_INDEX},
            "signature": signature.hex(),
            "beacon": {"randomness": randomness},
            "rollout": {
                "prompt_length": prompt_length,
                "completion_length": len(all_tokens) - prompt_length,
                "success": True,
                "total_reward": 0.0,
                "advantage": 0.0,
                "token_logprobs": token_logprobs,
                "forced": forced,
                "force_span": forced_span,
            },
        }

    async def _submit(
        self,
        submissions,
        prompt_idx: int,
        randomness: str,
        window_n: int,
        state,
        client,
        url,
    ):
        """Fully implemented submission logic."""
        if not submissions:
            return

        merkle_root = _compute_merkle_root(submissions)
        current_round = _current_drand_round_at_send()
        nonce = os.urandom(16).hex()

        print(f"🐞 {ACTIVE_PROTOCOL_PROFILE.profile_id}", flush=True)

        request = BatchSubmissionRequest(
            miner_hotkey=self.wallet.hotkey.ss58_address,
            prompt_idx=prompt_idx,
            window_start=state.window_n,
            merkle_root=merkle_root,
            rollouts=submissions,
            checkpoint_hash=getattr(state, "checkpoint_revision", ""),
            drand_round=current_round,
            nonce=nonce,
            protocol_version=FORCED_SEED_PROTOCOL_VERSION,
            generation_profile_id=(
                ACTIVE_PROTOCOL_PROFILE.profile_id
                if ACTIVE_PROTOCOL_PROFILE.protocol_version >= 3
                else ""
            ),
        )

        try:
            from reliquary.miner.submitter import SubmissionError, submit_batch_v2

            resp = await submit_batch_v2(
                url,
                request,
                client=client,
                wallet=self.wallet,
                randomness=state.randomness or "",
                drand_round_fn=_current_drand_round_at_send,
            )
            logger.info(
                f"📤 Submitted window={state.window_n} prompt={prompt_idx} "
                f"accepted={resp.accepted} reason={getattr(resp.reason, 'value', resp.reason)} "
                f"merkle_root={merkle_root[:16]}"
            )
            await self._save_submission_result(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "window_n": state.window_n,
                    "prompt_idx": prompt_idx,
                    "accepted": resp.accepted,
                    "reason": getattr(resp.reason, "value", resp.reason),
                    "merkle_root": merkle_root,
                    "checkpoint_hash": getattr(state, "checkpoint_revision", ""),
                    "drand_round": current_round,
                    "nonce": nonce,
                    "request": (
                        request.model_dump()
                        if hasattr(request, "model_dump")
                        else dict(request)
                    ),
                    "response": (
                        resp.model_dump()
                        if hasattr(resp, "model_dump")
                        else getattr(resp, "__dict__", str(resp))
                    ),
                }
            )
            return resp.accepted
        except SubmissionError as exc:
            logger.error(f"Submit failed for prompt {prompt_idx}: {exc}")
            return False
        except Exception as e:
            logger.error(f"Unexpected submit error {prompt_idx}: {e}")
            return False

    async def _load_checkpoint(self, local_path: str):
        import torch

        from reliquary.constants import ATTN_IMPLEMENTATION
        from reliquary.shared.modeling import load_text_generation_model

        if getattr(self, "_loaded_checkpoint_path", None) == local_path:
            return self.hf_model, True

        logger.info(f"🔄 Loading checkpoint from {local_path}")
        try:
            new_hf = await asyncio.to_thread(
                load_text_generation_model,
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            )

            def prepare_model(model):
                return model.to(f"cuda:{self.proof_gpu}").eval()

            new_hf = await asyncio.to_thread(prepare_model, new_hf)

            old = getattr(self, "hf_model", None)
            self.hf_model = new_hf
            if old:
                del old
            await asyncio.to_thread(torch.cuda.empty_cache)

            if "0.0.0.0" in self._vllm_client.base_url:
                self._vllm_client._reload_weight(local_path)
            else:
                self._vllm_client._reload_weight()

            self._loaded_checkpoint_path = local_path
            logger.info("✅ Checkpoint loaded successfully")
            return self.hf_model, True
        except Exception as e:
            logger.exception(f"Checkpoint load failed: {e}")
            return getattr(self, "hf_model", None), False
