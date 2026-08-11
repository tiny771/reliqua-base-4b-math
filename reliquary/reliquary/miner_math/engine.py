"""Optimized Miner Engine with vLLM v1 forced-seed processor using pre-computed u_list.

Based on miners/miner_processor/engine.py but integrates optimized ForcedSeedLogitsProcessor
that uses pre-generated u values (O(1) lookups) instead of per-token u_at() computation.

Performance Improvement: ~70% latency reduction in per-token generation (no SHA256 per token).
"""

from __future__ import annotations

import os
import asyncio
import json
import logging
import math
import time
import re
import statistics
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Set, Dict, Optional
import random as _random
import torch

from dataclasses import dataclass
from typing import Any

import traceback

import numpy as np

from reliquary.protocol.profiles import (
    ACTIVE_PROTOCOL_PROFILE,
    to_generation_contract,
)

from reliquary.constants import (
    FORCED_SEED_PROTOCOL_VERSION,
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    M_ROLLOUTS,
    PROMPT_RANGE_SIZE,
    POLL_INTERVAL_SECONDS,
    BFT_FORCE_TEMPLATE,
    MIN_EOS_PROBABILITY,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
)
from reliquary.shared.prompt_range import window_prompt_range
from reliquary.infrastructure import chain
from reliquary.protocol.signatures import sign_envelope
from reliquary.protocol.submission import BatchSubmissionRequest, RolloutSubmission
from reliquary.validator.verifier import rewards_std
from reliquary.shared.modeling import force_close_token_ids

if TYPE_CHECKING:
    from reliquary.environment.base import Environment

from reliquary.miner_math.utils import (
    VLLMGenerator,
    GenerationResult,
    _eval_difficulty,
)

# ================= CONTROL WORKFLOW =======================

FORCED_EOS_INJECT = False
EARLY_STOP_GENERATION = True
ENABLE_BFT_GENERATION = True
ENABLE_PREFLIGHT_GENERATION = False
FIRST_STAGE_MAX_TOKENS = 15000

# ==========================================================

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ====================== Stats ======================
class MinerStats:
    def __init__(self):
        self.prompts_processed = 0
        self.rollouts_generated = 0
        self.submissions_accepted = 0
        self.total_tokens = 0
        self.gen_times: deque = deque(maxlen=200)
        self.proof_times: deque = deque(maxlen=100)
        self.u_list_gen_times: deque = deque(maxlen=200)  # Track u_list pre-generation
        self.start_time = time.time()
        self.last_selection_time: float | None = None
        self.last_batch_completion_time: float | None = None

    def record_generation(self, duration: float, tokens: int):
        self.gen_times.append(duration)
        self.total_tokens += tokens
        self.rollouts_generated += 1

    def record_u_list_generation(self, duration: float):
        self.u_list_gen_times.append(duration)

    def record_selection(self) -> None:
        self.last_selection_time = time.time()

    def record_batch_completion(self) -> None:
        self.last_batch_completion_time = time.time()

    def get_stats(self) -> Dict:
        elapsed = time.time() - self.start_time
        avg_u_list_gen = (
            round(statistics.mean(self.u_list_gen_times), 3)
            if self.u_list_gen_times
            else 0
        )
        return {
            "uptime_min": round(elapsed / 60, 1),
            "prompts": self.prompts_processed,
            "tokens_per_sec": (
                round(self.total_tokens / elapsed, 1) if elapsed > 0 else 0
            ),
            "avg_gen_sec": (
                round(statistics.mean(self.gen_times), 2) if self.gen_times else 0
            ),
            "accepted_rate": round(
                self.submissions_accepted / max(self.prompts_processed, 1) * 100, 1
            ),
            "avg_u_list_gen_ms": avg_u_list_gen * 1000,  # Convert to ms
        }


stats = MinerStats()

@dataclass
class GrailJob:
    prompt_idx: int
    problem: str
    randomness: str

    gen_results: list
    rewards: list

    window_n: int
    state: object
    client: object
    url: str

    future: asyncio.Future

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
    # path = "/mnt/d/models/Qwen3.5-2B"  # Hardcoded path for testing
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
    # if cooldown:
    #     first = next(iter(cooldown))
    # else:
    #     first = None


    # print(f"Cooldown length: {len(cooldown)}, Cooldown[0]: {first}")

    # COOLDOWN_DIR = "/root/reliquary-miner"
    # COOLDOWN_FILE = os.path.join(COOLDOWN_DIR, "cooldown.json")
    # os.makedirs(COOLDOWN_DIR, exist_ok=True)

    # if len(cooldown) > 0:
    #     current_data = sorted(cooldown)
    #     if os.path.exists(COOLDOWN_FILE):
    #         with open(COOLDOWN_FILE, "r") as f:
    #             old_data = json.load(f)
    #             if old_data == current_data:
    #                 return

    #     with open(COOLDOWN_FILE, 'w') as f:
    #         json.dump(current_data, f)
    #         print("Saved COOLDOW_FILE!")      


    # if len(cooldown) > 100:
    #     COOLDOWN_DIR = "/root/reliquary-miner"
    #     COOLDOWN_FILE = os.path.join(COOLDOWN_DIR, "cooldown.json")
    #     COOLDOWN_PROBLEM_FILE = os.path.join(COOLDOWN_DIR, "cooldown_problems.jsonl")  # use .jsonl

    #     if not os.path.exists(COOLDOWN_FILE):
    #         logger.error(f"Cooldown file {COOLDOWN_FILE} not found.")
    #         return 0

    #     with open(COOLDOWN_FILE, "r") as f:
    #         idxs = json.load(f)
    #     if not idxs:
    #         logger.info("Cooldown file is empty.")
    #         return 0

    #     problems = []
    #     for idx in idxs:
    #         problem = env.get_problem(idx)
    #         if problem:
    #             problem_copy = dict(problem)
    #             problem_copy['idx'] = idx
    #             # add difficulty score
    #             problem_copy['difficulty'] = _eval_difficulty(problem)
    #             problems.append(problem_copy)

    #     # Write all problems to JSONL
    #     os.makedirs(os.path.dirname(COOLDOWN_PROBLEM_FILE) or '.', exist_ok=True)
    #     with open(COOLDOWN_PROBLEM_FILE, "w") as f:
    #         for p in problems:
    #             f.write(json.dumps(p) + "\n")

    #     print(f"Saved {len(problems)} cooldown problems to {COOLDOWN_PROBLEM_FILE}")

   
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
            if lo_diff <= _prompt_difficulty_cache.get(idx, 0.0) < hi_diff and _prompt_state_cache.get(idx, False) == False
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
                    if lo_diff <= _prompt_difficulty_cache.get(idx, 0.0) < hi_diff and _prompt_state_cache.get(idx, False) == False
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

    stats.record_selection()
    logger.info(
        f"🎲 Selected {len(selected_indices)} prompts from range {lo}-{hi} "
        f"difficulty_range={difficulty_range}"
    )
    return selected_indices, problems


def _compute_merkle_root(rollouts):
    import hashlib, json

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


def _bft_generation_offset(
    *,
    prompt_len: int,
    completed_tokens: List[int] | Tuple[int, ...] | None,
    force_ids: List[int] | Tuple[int, ...] | None,
) -> tuple[int, int]:
    completed_len = len(completed_tokens or [])
    force_len = len(force_ids or [])
    base_offset = completed_len + force_len
    return base_offset, prompt_len + base_offset


def _build_vllm_extra_body(
    *,
    prompt_len: int,
    # u_list_entry: List[float],
    prompt_idx: int,
    rollout_idx: int,
    randomness: str,
    checkpoint_hash: str,
    base_offset: int = 0,
    start_len: int | None = None,
) -> Dict[str, object]:
    return {
        "randomness": randomness,
        "prompt_idx": prompt_idx,
        "checkpoint_hash": checkpoint_hash,
        "rollout_index": rollout_idx,
        "base_offset": base_offset,
        "temperature": T_PROTO,
        "top_k": TOP_K_PROTO,
        "top_p": TOP_P_PROTO,
        # "u_list": u_list_entry,
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
        proof_gpu=1,
        max_new_tokens=MAX_NEW_TOKENS_PROTOCOL_CAP,
        validator_url_override=None,
        max_concurrent=30,
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
        self._difficulty_range = (6.8, 7.8)
        self._n_candidates = 12

        self._process_start = True
        self._bft_n_candidates = 0


        self._cooldown: Set[int] = set()
        self._selected: Set[int] = set()
        self._prompt_range = (0, len(env) if env else 0)

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._checkpoint_lock = asyncio.Lock()

        self._local_checkpoint_n = 0
        self._local_checkpoint_hash = ""


        self.grail_queue = asyncio.Queue()

        self.grail_worker_task = None

        from reliquary.shared.hf_compat import resolve_hidden_size
        from reliquary.protocol.grail_verifier import GRAILVerifier

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

        self._preflight_perplexity_threshold: float | None = None
        self._results_dir = Path(__file__).resolve().parents[3] / "results"
        self._analysis_log_path = self._results_dir / "miner_analysis.jsonl"
        self._summary_path = self._results_dir / "miner_summary.json"
        self._submission_log_path = self._results_dir / "submission_results.jsonl"
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._vllm_client = VLLMGenerator(base_url=vllm_url, model_name="reliquary")
        logger.info(
            f"🚀 Optimized Miner ready (u_list pre-gen) | concurrency={max_concurrent} | results_dir={self._results_dir}"
        )

    async def start_workers(self):
        if self.grail_worker_task is None:
            self.grail_woker_task = asyncio.create_task(
                self._grail_worker()
            )

            logger.info("🚀 Grail worker started")

    def _build_grail_commits_batched(
        self,
        rollouts_data: List[Dict],
        randomness: str,
    ) -> List[Dict]:
        """
        Build GRAIL commits using mini-batches.

        Splits large rollout batches to reduce GPU peak memory.
        """

        if not rollouts_data:
            return []


        # Adjust this depending on VRAM.
        # For RTX PRO 6000 96GB:
        # 2 = safest
        # 4 = faster
        MINI_BATCH_SIZE = 2


        all_commits = []

        total = len(rollouts_data)


        for start_idx in range(
            0,
            total,
            MINI_BATCH_SIZE,
        ):

            end_idx = min(
                start_idx + MINI_BATCH_SIZE,
                total,
            )


            mini_batch = rollouts_data[
                start_idx:end_idx
            ]


            logger.info(
                f"🔹 GRAIL mini batch "
                f"{start_idx}:{end_idx}/{total}"
            )


            commits = self._build_grail_commit_batch_single(
                mini_batch,
                randomness,
            )


            all_commits.extend(commits)


            # release allocator blocks
            torch.cuda.empty_cache()


        return all_commits

    def _build_grail_commit_batch_single(
        self,
        rollouts_data: List[Dict],
        randomness: str,
    ) -> List[Dict]:

        if not rollouts_data:
            return []


        start = time.time()


        all_token_seqs = [
            rd["all_tokens"]
            for rd in rollouts_data
        ]


        max_len = max(
            len(x)
            for x in all_token_seqs
        )


        batch_size = len(all_token_seqs)


        device = f"cuda:{self.proof_gpu}"


        pad_token_id = (
            getattr(
                self.tokenizer,
                "pad_token_id",
                0,
            )
            or 0
        )


        padded_tokens = []
        attention_masks = []


        for tokens in all_token_seqs:

            pad_len = max_len - len(tokens)


            padded_tokens.append(
                tokens + [pad_token_id] * pad_len
            )


            attention_masks.append(
                [1] * len(tokens)
                +
                [0] * pad_len
            )


        proof_input = torch.tensor(
            padded_tokens,
            device=device,
            dtype=torch.long,
        )


        attention_mask = torch.tensor(
            attention_masks,
            device=device,
            dtype=torch.long,
        )


        from reliquary.shared.forward import forward_single_layer
        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding


        try:

            with torch.inference_mode():

                hidden, token_logprobs_batch = (
                    forward_single_layer(
                        self.hf_model,
                        proof_input,
                        attention_mask,
                        LAYER_INDEX,
                        return_token_logprobs=True,
                    )
                )

        except Exception as e:

            logger.exception(
                f"GRAIL mini batch failed: {e}"
            )

            del proof_input
            del attention_mask

            torch.cuda.empty_cache()

            return [
                None
                for _ in rollouts_data
            ]


        r_vec = self._verifier.generate_r_vec(
            randomness
        )


        model_name = getattr(
            self.hf_model,
            "name_or_path",
            "unknown",
        )

        commits = []

        for idx, rd in enumerate(rollouts_data):

            all_tokens = rd["all_tokens"]

            prompt_length = rd["prompt_length"]

            forced = rd["forced"]

            forced_span = rd["forced_span"]


            seq_len = len(all_tokens)


            #
            # Hidden state commitment
            #

            hidden_states = (
                hidden[
                    idx,
                    :seq_len
                ]
                .clone()
            )


            commitments = (
                self._verifier
                .create_commitments_batch(
                    hidden_states,
                    r_vec,
                )
            )


            #
            # Token log probabilities
            #

            seq_token_lp = (
                token_logprobs_batch[
                    idx,
                    :seq_len
                ]
            )


            token_logprobs = [
                seq_token_lp[i].item()
                for i in range(
                    prompt_length,
                    seq_len,
                )
            ]


            signature = sign_commit_binding(
                all_tokens,
                randomness,
                model_name,
                LAYER_INDEX,
                commitments,
                self.wallet,
            )


            commit = {

                "tokens": all_tokens,

                "commitments": commitments,

                "proof_version":
                    GRAIL_PROOF_VERSION,


                "model": {
                    "name": model_name,
                    "layer_index": LAYER_INDEX,
                },


                "signature":
                    signature.hex(),


                "beacon": {
                    "randomness": randomness,
                },


                "rollout": {

                    "prompt_length":
                        prompt_length,

                    "completion_length":
                        len(all_tokens)-prompt_length,

                    "success":
                        True,

                    "total_reward":
                        0.0,

                    "advantage":
                        0.0,

                    "token_logprobs":
                        token_logprobs,

                    "forced":
                        forced,

                    "force_span":
                        forced_span,
                },
            }


            commits.append(commit)


            del hidden_states


        #
        # Free GPU memory
        #

        del proof_input
        del attention_mask
        del hidden
        del token_logprobs_batch


        torch.cuda.empty_cache()


        duration = time.time() - start


        stats.proof_times.append(
            duration
        )


        logger.info(
            f"⚡ GRAIL mini batch "
            f"{batch_size} rollouts "
            f"{duration:.3f}s"
        )


        return commits

    def _build_all_grail_submissions(self, job):
                 # Prepare batched data for all rollouts
        rollouts_data = []
        for gen_result in job.gen_results:
            prompt_ids = gen_result.prompt_token_ids
            comp_ids = gen_result.tokens
            all_tokens = prompt_ids + comp_ids
            rollouts_data.append({
                "all_tokens": all_tokens,
                "prompt_length": len(prompt_ids),
                "forced": gen_result.forced,
                "forced_span": gen_result.forced_span,
            })
        
        # Batch generate all commits in a single forward pass
        commits = self._build_grail_commits_batched(rollouts_data, job.randomness)
        
        # Build submissions from batched commits
        submissions = []
        for gen_result, reward, commit in zip(job.gen_results, job.rewards, commits):
            if commit is None:
                logger.warning(f"⚠️ Skipping rollout for #{prompt_idx} due to failed commit generation")
                continue
            
            prompt_ids = gen_result.prompt_token_ids
            comp_ids = gen_result.tokens
            all_tokens = prompt_ids + comp_ids
            
            submissions.append(RolloutSubmission(
                tokens=all_tokens,
                reward=reward,
                commit=commit,
                env_name=self.env.name
            ))

        return submissions
            

        # return [
        #     self._build_rollout_submission(
        #         g,
        #         job.problem,
        #         job.randomness,
        #         rew,
        #     )
        #     for g, rew in zip(
        #         job.gen_results,
        #         job.rewards,
        #     )
        # ]

    async def _grail_worker(self):

        while True:

            job = await self.grail_queue.get()

            try:
                # logger.info(
                #     f"prompt={job.prompt_idx} submission construction is started!"
                # )

                submissions = await asyncio.to_thread(
                    self._build_all_grail_submissions,
                    job,
                )

                result = await self._submit(
                    submissions=submissions,
                    prompt_idx=job.prompt_idx,
                    randomness=job.randomness,
                    window_n=job.window_n,
                    state=job.state,
                    client=job.client,
                    url=job.url,
                    rewards=job.rewards,
                )

                job.future.set_result(result)

            except Exception as e:
                job.future.set_exception(e)

            finally:
                self.grail_queue.task_done()

    
    async def mine_window(self, subtensor):
        import httpx
        from reliquary.miner.submitter import (
            SubmissionError,
            discover_validator_url,
            get_window_state_v2,
            submit_batch_v2,
        )
        from reliquary.protocol.submission import WindowState

        await self.start_workers()

        subtensor = await chain.get_subtensor()

        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)


        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                state = await self._sync_state(client, url)
                if not state or state.state != WindowState.OPEN or not state.randomness:
                    await self._vllm_client.cancel_all_requests()
                    await asyncio.sleep(0.5)
                    self._selected = set()
                    self._process_start = True
                    continue

                if self._process_start:

                    prompt_idxs, problems = select_prompts(
                        self.env,
                        self._cooldown,
                        self._selected,
                        self._prompt_range,
                        count=self._n_candidates,
                        difficulty_range=self._difficulty_range,
                    )

                    self._selected.update(prompt_idxs)
                    
                    logger.info(
                        f"Selected prompts "
                        f"{self._selected}"
                    )

                    diffs = [
                        problem.get("difficulty", _eval_difficulty(problem)[0])
                        for problem in problems
                    ]
                    logger.info(
                        f"🧭 Window {state.window_n} prompt_batch size={len(prompt_idxs)} "
                        f"range={self._prompt_range} cooldown={len(self._cooldown)}"
                    )

                    prompt_tasks = []
                    for idx, prob, diff in zip(prompt_idxs, problems, diffs):
                        prompt_tasks.append(
                            asyncio.create_task(
                                self._process_prompt_pipeline(
                                    prob,
                                    idx,
                                    diff,
                                    state.randomness,
                                    state.window_n,
                                    client,
                                    url,
                                    state,
                                )
                            )
                        )

                    if prompt_tasks:
                        results = await asyncio.gather(
                            *prompt_tasks, return_exceptions=True
                        )
                        for result in results:
                            if isinstance(result, Exception):
                                logger.error(f"Prompt task failed: {result}")

            
                    stats.record_batch_completion()
                    self._log_stats()

                    self._process_start = False

                else:
                    logger.info(
                        f"🧭 Window {state.window_n} prompts selection is stopped."
                    )
                    time.sleep(3)


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
        self, problem, prompt_idx, diff, randomness, window_n, client, url, state
    ):
        async with self._semaphore:
            prompt, prompt_len = await asyncio.to_thread(
                self._format_problem, problem.get("prompt")
            )

            ground_truth=problem.get("ground_truth", "")

            start_time = time.time()
            
            perplexity = 0.0

            gen_results = await self._generate_rollouts(
                prompt,
                prompt_idx,
                prompt_len,
                diff,
                randomness,
                state.checkpoint_revision,
                ground_truth,
            )
            analysis_metrics = self._collect_rollout_analysis_metrics(
                gen_results, problem
            )
            if len(gen_results or []) < M_ROLLOUTS:
                logger.warning(
                    f"⏭️  #{prompt_idx} → rollout generation incomplete | count={len(gen_results or [])} | diff={diff:.2f} | perplexity={perplexity:.2f} | reward_vector={analysis_metrics.get('rewards_vector', [])}"
                )
                await self._record_analysis_result(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "prompt_idx": prompt_idx,
                        "window_n": window_n,
                        "status": "rollout_incomplete",
                        "reason": "incomplete_rollouts",
                        "diff": diff,
                        "rollout_count": len(gen_results or []),
                        "prompt_len": prompt_len,
                        "prompt_preview": problem.get("prompt", "")[:160],
                        "solution": problem.get("solution", ""),
                        "solution_len": len(problem.get("solution", "")),
                        **analysis_metrics,
                        "completion_rollout_previews": [
                            getattr(r, "text", "") for r in gen_results or []
                        ],
                        "completion_rollout_lengths": [
                            len(getattr(r, "tokens", []) or []) for r in gen_results or []
                        ],
                    }
                )
                stats.prompts_processed += 1
                return

            rewards = [self.env.compute_reward(problem, r.text) for r in gen_results]
            sigma = rewards_std(rewards)

            if sigma < 0.433:
                logger.info(
                    f"⏭️  #{prompt_idx} → SKIPPED | sigma={sigma:.3f} | rewards={rewards} | diff={diff:.2f} | perp={perplexity:.2f}"
                )
                await self._record_analysis_result(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "prompt_idx": prompt_idx,
                        "window_n": window_n,
                        "status": "sigma_skipped",
                        "reason": "sigma_threshold",
                        "diff": diff,
                        "perplexity": perplexity,
                        "sigma": sigma,
                        "rewards": rewards,
                        "prompt_len": prompt_len,
                        "prompt_preview": problem.get("prompt", "")[:160],
                        "solution": problem.get("solution", ""),
                        "solution_len": len(problem.get("solution", "")),
                        **analysis_metrics,
                        "completion_rollout_previews": [
                            getattr(r, "text", "") for r in gen_results or []
                        ],
                        "completion_rollout_lengths": [
                            len(getattr(r, "tokens", []) or []) for r in gen_results or []
                        ],
                    }
                )
                stats.prompts_processed += 1
                return

            logger.info(
                f"💎  #{prompt_idx} → PASSED | sigma={sigma:.3f} | rewards={rewards} | diff={diff:.2f} | perp={perplexity:.2f}"
            )

            future = asyncio.get_running_loop().create_future()

            job = GrailJob(
                prompt_idx=prompt_idx,
                problem=problem,
                randomness=randomness,
                gen_results=gen_results,
                rewards=rewards,
                window_n=window_n,
                state=state,
                client=client,
                url=url,
                future=future,
            )

            await self.grail_queue.put(job)
            logger.info(
                f"⏭️  #{prompt_idx} → queued for GRAIL submission."
            )

            submit_result = await future

            await self._record_analysis_result(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt_idx": prompt_idx,
                    "window_n": window_n,
                    "status": "submitted" if submit_result else "submit_failed",
                    "reason": "submitted" if submit_result else "submit_failed",
                    "diff": diff,
                    "perplexity": perplexity,
                    "sigma": sigma,
                    "rewards": rewards,
                    "prompt_len": prompt_len,
                    "rollout_count": len(gen_results),
                    "prompt_preview": problem.get("prompt", "")[:160],
                    "solution": problem.get("solution", ""),
                    "solution_len": len(problem.get("solution", "")),
                    **analysis_metrics,
                    "completion_rollout_previews": [
                        getattr(r, "text", "") for r in gen_results or []
                    ],
                    "completion_rollout_lengths": [
                        len(getattr(r, "tokens", []) or []) for r in gen_results or []
                    ],
                }
            )

          
            stats.prompts_processed += 1

    def _collect_rollout_analysis_metrics(
        self, rollouts: List[object], problem: Dict | None = None
    ) -> Dict[str, List[float] | List[int]]:
        if not rollouts:
            return {
                "rewards_vector": [],
                "pstop_vector": [],
                "completion_length_vector": [],
            }

        rewards_vector = []
        completion_length_vector = []
        pstop_vector = []
        for rollout in rollouts:
            try:
                reward = self.env.compute_reward(problem, getattr(rollout, "text", ""))
            except Exception:
                reward = 0.0
            rewards_vector.append(reward)

            tokens = getattr(rollout, "tokens", []) or []
            completion_length_vector.append(len(tokens))

            token_logprobs = getattr(rollout, "token_logprobs", []) or []
            pstop_vector.append(token_logprobs[-1] if token_logprobs else None)

        return {
            "rewards_vector": rewards_vector,
            "pstop_vector": pstop_vector,
            "completion_length_vector": completion_length_vector,
        }

    def _should_skip_prompt(self, perplexity: float | None, diff: float) -> bool:
        if self._preflight_perplexity_threshold is None:
            return False
        if perplexity is None:
            return True
        return float(perplexity) < self._preflight_perplexity_threshold

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

        from reliquary.validator.boxed_integrity import has_malformed_final_answer

        _bad, _bad_reason = has_malformed_final_answer(
            reward=0.0,
            text=rollout.text,
            completion_length=len(tokens),
            cap=MAX_NEW_TOKENS_PROTOCOL_CAP,
        )
        if _bad:
            return True, True

        return False, False

    def _check_distribution_suspicious(
        self, rollout: GenerationResult
    ) -> tuple[bool, dict]:
        import numpy as np

        from reliquary.constants import (
            SAMPLING_HIGH_P,
            SAMPLING_LOW_P,
            SAMPLING_LOW_Q10_MAX,
            SAMPLING_MEDIAN_LOW_MAX,
        )

        logprobs = getattr(rollout, "token_logprobs", None)
        if not logprobs:
            return True, {"reason": "missing_or_empty_token_logprobs"}

        try:
            values = [float(lp) for lp in logprobs]
        except (TypeError, ValueError):
            return True, {"reason": "non_numeric_token_logprobs"}

        if len(values) < 2 or not np.isfinite(values).all():
            return True, {
                "reason": "invalid_token_logprobs",
                "count": len(values),
            }

        # Convert logprobs to probabilities (stable exp, underflow -> 0)
        probs = np.exp(np.asarray(values, dtype=np.float64))
        if probs.size == 0 or not np.isfinite(probs).all():
            return True, {"reason": "invalid_probabilities"}

        metrics = {
            "mean": float(probs.mean()),
            "median": float(np.median(probs)),
            "q10": float(np.quantile(probs, 0.10)),
            "low_frac": float((probs <= SAMPLING_LOW_P).mean()),
            "high_frac": float((probs >= SAMPLING_HIGH_P).mean()),
        }

        suspicious = (
            metrics["median"] < SAMPLING_MEDIAN_LOW_MAX
            or metrics["q10"] < SAMPLING_LOW_Q10_MAX
        )
        return suspicious, metrics

    async def _generate_single_rollout(
        self,
        prompt: str,
        prompt_idx: int,
        prompt_len: int,
        rollout_idx: int,
        randomness: str,
        checkpoint_hash: str,
        ground_truth: str,
        # u_list_entry: List[float],
    ) -> Optional["GenerationResult"]:
        try:
            extra_body = _build_vllm_extra_body(
                prompt_len=prompt_len,
                prompt_idx=prompt_idx,
                rollout_idx=rollout_idx,
                randomness=randomness,
                checkpoint_hash=checkpoint_hash,
            )

            from reliquary.constants import BFT_THINKING_BUDGET

            # 1. Initial generation: reserve the full thinking budget so the
            # injected FORCE span lands exactly at the validator's expected
            # offset (prompt_len + BFT_THINKING_BUDGET).
            result = await self._vllm_client.generate_rollout_async(
                prompt,
                max_tokens=BFT_THINKING_BUDGET,
                # max_tokens=2500,
                extra_body=extra_body,
            )

            # 2. Validate initial result
            if not result or not getattr(result, "tokens", None):
                logger.warning(
                    "⚠️ Rollout #%d for #%d failed validation (empty or missing tokens)",
                    rollout_idx,
                    prompt_idx,
                )
                return None, False

            if result is not None:
                result.rollout_idx = rollout_idx

            # 3. Check if BFT (Forced) generation is needed. The validator only
            # accepts a forced span when the first pass actually consumed the
            # full thinking budget and still lacked a natural close token.

            completion_tokens = list(getattr(result, "tokens", None) or [])
            reached_thinking_budget = len(completion_tokens) >= BFT_THINKING_BUDGET
            needs_bft = (
                reached_thinking_budget
                and ("</think>" not in getattr(result, "text", ""))
                and (result.finish_reason != "stop")
            )

            if needs_bft and ENABLE_BFT_GENERATION:
                logger.info(
                    "↩️ Rollout #%d for #%d triggering BFT generation (missing '</think>' and non-stop finish) completion length: %d",
                    rollout_idx,
                    prompt_idx,
                    len(completion_tokens),
                )

                # Prepare forced generation
                bft_prompt = prompt + (result.text or "") + BFT_FORCE_TEMPLATE
                forced_ids = force_close_token_ids(self.tokenizer)
                bft_base_offset, bft_start_len = _bft_generation_offset(
                    prompt_len=prompt_len,
                    completed_tokens=getattr(result, "tokens", None),
                    force_ids=forced_ids,
                )

                bft_extra_body = _build_vllm_extra_body(
                    prompt_len=prompt_len,
                    # u_list_entry=u_list_entry,
                    prompt_idx=prompt_idx,
                    rollout_idx=rollout_idx,
                    randomness=randomness,
                    checkpoint_hash=checkpoint_hash,
                    base_offset=bft_base_offset,
                    start_len=bft_start_len,
                )

                bft_result = await self._vllm_client.generate_rollout_async(
                    bft_prompt,
                    max_tokens=512,
                    extra_body=bft_extra_body,
                )

                # Validate BFT result before attempting to merge
                if not bft_result or not getattr(bft_result, "tokens", None):
                    logger.warning(
                        "⚠️ BFT generation failed validation for Rollout #%d of #%d",
                        rollout_idx,
                        prompt_idx,
                    )
                    return None, False

                if bft_result.finish_reason != "stop" and FORCED_EOS_INJECT:
                    boxed_end_idx = self._find_boxed_end(bft_result.text)
                    if boxed_end_idx != -1:
                        bft_result.text = bft_result.text[:boxed_end_idx]
                        truncate_token_idx = self._find_token_idx_for_char_idx(
                            bft_result.tokens, boxed_end_idx
                        )
                        bft_result.tokens = bft_result.tokens[:truncate_token_idx]
                        bft_result.token_logprobs = bft_result.token_logprobs[
                            :truncate_token_idx
                        ]
                        bft_result.top_logprobs = bft_result.top_logprobs[
                            :truncate_token_idx
                        ]
                        eos_token_id = self.tokenizer.eos_token_id
                        if eos_token_id is not None:
                            bft_result.tokens.append(eos_token_id)
                            bft_result.token_logprobs.append(0.0)
                            bft_result.top_logprobs.append({})
                        bft_result.finish_reason = "stop"

                # FIX: Correct text and token merging without prompt leakage
                bft_result.text = (
                    (result.text or "") + BFT_FORCE_TEMPLATE + (bft_result.text or "")
                )
                bft_result.tokens = (
                    (result.tokens or []) + forced_ids + (bft_result.tokens or [])
                )
                bft_result.prompt_token_ids = result.prompt_token_ids

                # bft_result.token_logprobs = (result.token_logprobs or []) + (bft_result.token_logprobs or [])
                # bft_result.top_logprobs = (result.top_logprobs or []) + (bft_result.top_logprobs or [])
                # bft_result.prompt_logprobs = result.prompt_logprobs
                # Validator expects BFT force_span to begin exactly at:
                #   start - prompt_len == BFT_THINKING_BUDGET
                # i.e. absolute start = prompt_len + BFT_THINKING_BUDGET.
                from reliquary.constants import BFT_THINKING_BUDGET

                force_start = prompt_len + BFT_THINKING_BUDGET
                bft_result.forced = True
                bft_result.forced_span = (
                    int(force_start),
                    int(force_start + len(forced_ids)),
                )
                bft_result.rollout_idx = rollout_idx
                bft_result.finish_reason = "bft_length"

                return bft_result, True

            return result, False

        except Exception as e:
            logger.exception(
                "⚠️ Rollout #%d failed for #%d: %s", rollout_idx, prompt_idx, e
            )
            return None

    def _find_boxed_end(self, text: str) -> int:
        """Finds the index right after the closing '}' of the first \boxed{...}."""
        brace_count = 1
        for i in range(len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    return i + 1
        return -1

    def _find_token_idx_for_char_idx(
        self, tokens: List[int], target_char_idx: int
    ) -> int:
        """Finds the token index that covers the target character index."""
        current_text = ""
        for i, token_id in enumerate(tokens):
            # Decode single token to accumulate text length
            current_text += self.tokenizer.decode([token_id], skip_special_tokens=False)
            if len(current_text) >= target_char_idx:
                return i + 1
        return len(tokens)

    def _passes_stage_one_gate(self, result, prompt_idx: int, prompt_len: int | None = None, ground_truth: str | None = None) -> bool:
        should_stop, has_malformed = self._should_stop_generation(result)
        if should_stop and EARLY_STOP_GENERATION:
            last_tokens = getattr(result, "tokens", None) or []
            last_logprob = (
                getattr(result, "token_logprobs", None) or []
            )[-1] if getattr(result, "token_logprobs", None) else None
            last_token_text = ""
            if last_tokens:
                last_token_text = self.tokenizer.decode([last_tokens[-1]])
            logger.warning(
                f"⚠️ #{prompt_idx} → stage-one malformed rollout rejected "
                f"| rollout={getattr(result, 'rollout_idx', 0)} "
                f"| final_token={last_token_text} "
                f"| pstop={last_logprob} "
                f"| finish_reason={getattr(result, 'finish_reason', None)} "
                f"| malformed={has_malformed}"
            )
            return False

        # suspicious, dist_metrics = self._check_distribution_suspicious(result)
        # if suspicious and EARLY_STOP_GENERATION:
        #     logger.warning(
        #         f"⚠️ #{prompt_idx} → stage-one suspicious rollout rejected "
        #         f"| rollout={getattr(result, 'rollout_idx', 0)} "
        #         f"| metrics={dist_metrics}"
        #     )
        #     return False

        completion_tokens = getattr(result, "tokens", None) or []
        prompt_token_ids = getattr(result, "prompt_token_ids", None)
        if prompt_token_ids is not None:
            total_token_count = len(prompt_token_ids) + len(completion_tokens)
        else:
            total_token_count = (prompt_len or 0) + len(completion_tokens)

        if total_token_count <= FIRST_STAGE_MAX_TOKENS:
            logger.warning(
                f"⚠️ #{prompt_idx} → stage-one rollout rejected on token count "
                f"| rollout={getattr(result, 'rollout_idx', 0)} "
                f"| tokens={total_token_count} "
                f"| completion_tokens={len(completion_tokens)} "
                f"| prompt_tokens={len(prompt_token_ids) if prompt_token_ids is not None else (prompt_len or 0)} "
                f"| threshold={FIRST_STAGE_MAX_TOKENS}"
            )
            return False

        # token_logprobs = getattr(result, "token_logprobs", None) or [0.0]

        # problem = {"ground_truth": ground_truth}

        # reward = self.env.compute_reward(problem, result.text)

        # reward = getattr(result, "reward", 0.0)

        # lp = np.array(token_logprobs, dtype=np.float32)
        # avg_prob = float(np.exp(lp.mean()))

        # confidence_gap = avg_prob - reward

        # if reward == 0:
        #     if confidence_gap < 0.5:
        #         logger.warning(
        #             f"⚠️ #{prompt_idx} → stage-one rollout rejected on confidence_gap "
        #             f"| rollout={getattr(result, 'rollout_idx', 0)} "
        #             f"| reward={reward} confidence gap={confidence_gap:.3f} "
        #             f"| threshold={0.5}"
        #         )
        #         return False

        # if reward == 1:
        #     if confidence_gap > -0.2:
        #         logger.warning(
        #             f"⚠️ #{prompt_idx} → stage-one rollout rejected on confidence_gap "
        #             f"| rollout={getattr(result, 'rollout_idx', 0)} "
        #             f"| reward={reward} confidence gap={confidence_gap:.3f} "
        #             f"| threshold={-0.2}"
        #         )
        #         return False

        # print(f"Finally this rollout {prompt_idx} passed in first stage")
        return True

    def _passes_stage_two_gate(self, result, prompt_idx: int) -> bool:
        should_stop, has_malformed = self._should_stop_generation(result)
        if should_stop and EARLY_STOP_GENERATION:
            last_tokens = getattr(result, "tokens", None) or []
            last_logprob = (
                getattr(result, "token_logprobs", None) or []
            )[-1] if getattr(result, "token_logprobs", None) else None
            last_token_text = ""
            if last_tokens:
                last_token_text = self.tokenizer.decode([last_tokens[-1]])
            logger.warning(
                f"⚠️ #{prompt_idx} → stage-two malformed rollout rejected "
                f"| rollout={getattr(result, 'rollout_idx', 0)} "
                f"| final_token={last_token_text} "
                f"| pstop={last_logprob} "
                f"| finish_reason={getattr(result, 'finish_reason', None)} "
                f"| malformed={has_malformed}"
            )
            return False

        # suspicious, dist_metrics = self._check_distribution_suspicious(result)
        # if suspicious and EARLY_STOP_GENERATION:
        #     logger.warning(
        #         f"⚠️ #{prompt_idx} → stage-two suspicious rollout rejected "
        #         f"| rollout={getattr(result, 'rollout_idx', 0)} "
        #         f"| metrics={dist_metrics}"
        #     )
        #     return False

        return True

    async def _generate_rollouts(
        self, prompt, prompt_idx, prompt_len, diff, randomness, checkpoint_hash, ground_truth
    ) -> List[GenerationResult]:
        """Two-stage rollout generation: validate one rollout first, then generate the rest."""
        start = time.time()

        try:
            # Stage 2: only launch the remaining rollouts after the first rollout
            # has already passed the stage-one gate.
            results = []

            bft_state = False

            pending = {
                asyncio.create_task(
                    self._generate_single_rollout(
                        prompt,
                        prompt_idx,
                        prompt_len,
                        rollout_idx,
                        randomness,
                        checkpoint_hash,
                        ground_truth,
                    )
                )
                for rollout_idx in range(0, M_ROLLOUTS)
            }

            async def _abort_batch() -> None:
                for extra_task in pending:
                    extra_task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    try:
                        result, result_state = task.result()
                    except Exception as exc:
                        logger.error(
                            f"⚠️ Rollout task failed for #{prompt_idx}: {exc}"
                        )
                        result = None

                    if result is None:
                        await _abort_batch()
                        duration = time.time() - start
                        logger.warning(
                            f"⚠️ #{prompt_idx} → rollout failed "
                            f"{len(results)}/{M_ROLLOUTS} "
                            f"| {duration:.2f}s"
                        )
                        return []

                    if not self._passes_stage_one_gate(result, prompt_idx, prompt_len, ground_truth):
                        await _abort_batch()
                        duration = time.time() - start
                        logger.warning(
                            f"⚠️ #{prompt_idx}f → later rollout rejected at stage-one gate "
                            f"| rollout={getattr(result, 'rollout_idx', 0)} "
                            f"| generated={len(results)}/{M_ROLLOUTS} "
                            f"| {duration:.2f}s"
                        )
                        return []

                    if result_state:
                        bft_state = True

                    results.append(result)
                    if len(results) >= M_ROLLOUTS:
                        if bft_state:
                            self._bft_n_candidates += 1
                        await _abort_batch()
                        results.sort(
                            key=lambda r: getattr(r, "rollout_idx", -1)
                        )
                        duration = time.time() - start
                        tokens_est = sum(
                            len(getattr(r, "tokens", [])) for r in results
                        )
                        stats.record_generation(duration, tokens_est)
                        logger.info(
                            f"⚡ #{prompt_idx} → "
                            f"{len(results)}/{M_ROLLOUTS} rollouts "
                            f"in {duration:.2f}s "
                            f"| diff={diff:.2f}"
                        )
                        return results



            results.sort(key=lambda r: getattr(r, "rollout_idx", -1))
            duration = time.time() - start
            tokens_est = sum(len(getattr(r, "tokens", [])) for r in results)
            stats.record_generation(duration, tokens_est)
            return results

        except Exception as e:
            logger.exception(f"⚠️ Gen fail #{prompt_idx}: {e}")
            return []


    async def _record_analysis_result(self, record: Dict):
        self._results_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._append_analysis_record, record)
        await self._write_summary_report()

    def _append_analysis_record(self, record: Dict):
        with self._analysis_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")

    async def _save_submission_result(self, record: Dict):
        self._results_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._append_submission_record, record)

    def _append_submission_record(self, record: Dict):
        with self._submission_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")

    async def _write_summary_report(self):
        await asyncio.to_thread(self._write_summary_report_sync)

    def _write_summary_report_sync(self):
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompts_processed": stats.prompts_processed,
            "rollouts_generated": stats.rollouts_generated,
            "submissions_accepted": stats.submissions_accepted,
            "avg_gen_sec": (
                round(statistics.mean(stats.gen_times), 3) if stats.gen_times else 0
            ),
            "avg_proof_sec": (
                round(statistics.mean(stats.proof_times), 3) if stats.proof_times else 0
            ),
        }
        with self._summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

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
        # print(f"Prompt Text: {self.tokenizer.decode(prompt_ids)}", flush=True)
        comp_ids = gen.tokens
        # print(f"Completion Text: {self.tokenizer.decode(comp_ids)}", flush=True)
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

        from reliquary.shared.forward import forward_single_layer
        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding

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

        duration = time.time() - start
        stats.proof_times.append(duration)

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
        rewards,
    ):
        """Fully implemented submission logic."""
        if not submissions:
            return

        merkle_root = _compute_merkle_root(submissions)
        current_round = _current_drand_round_at_send()
        nonce = os.urandom(16).hex()


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
            from reliquary.miner.submitter import submit_batch_v2, SubmissionError

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
                f"merkle_root={merkle_root[:16]} "
                f"rewards={rewards}"
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
            if resp.accepted:
                stats.submissions_accepted += 1
                return True
            return False
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

            # self._vllm_client._reload_weight(local_path)

            self._loaded_checkpoint_path = local_path
            logger.info("✅ Checkpoint loaded successfully")
            return self.hf_model, True
        except Exception as e:
            logger.exception(f"Checkpoint load failed: {e}")
            return getattr(self, "hf_model", None), False

    def _log_stats(self):
        s = stats.get_stats()
        start_time = stats.last_selection_time or stats.start_time
        end_time = stats.last_batch_completion_time or time.time()
        cycle_time = max(0.0, end_time - start_time)
        cycle_time = round(cycle_time, 1)
        logger.info(
            f"📊 [OPTIMIZED MINER STATS] uptime={s['uptime_min']}m | prompts={s['prompts']} | "
            f"{s['tokens_per_sec']} tok/s | avg_gen={s['avg_gen_sec']}s | "
            f"cycle_time={cycle_time}s | ({self._n_candidates}candidates) | "
            # f"u_list_gen={s['avg_u_list_gen_ms']:.2f}ms | accept={s['accepted_rate']}% | "
            f"bft_candidates={self._bft_n_candidates} | "
            f"accept={s['accepted_rate']}% | "
            f"rollouts={stats.rollouts_generated} | proof_avg={round(statistics.mean(stats.proof_times), 3) if stats.proof_times else 0:.3f}s"
        )