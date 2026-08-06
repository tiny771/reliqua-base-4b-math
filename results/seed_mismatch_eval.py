#!/usr/bin/env python3
"""Local seed-mismatch evaluator for Reliquary submission rollouts.

This script reads a submission_results.jsonl file produced by the miner and
replays the validator-side forced-seed consistency check using the same public
uniform derivation as the validator.

It is designed to be a standalone local analysis tool for inspecting whether
submitted rollouts match the protocol's forced-seed stream.

Usage:
    python3 results/seed_mismatch_eval.py \
        --submission-log results/submission_results.jsonl \
        --checkpoint-model ReliquaryForge/qwen3.5-4b-reliquary-v4 \
        --limit 50 \
        --device cuda:0 \
        --output-json results/seed_mismatch_report.json
        [--tokenizer-model <hf-model-or-local-path>] \
        [--device cuda:0] \
        [--limit 50] \
        [--output-json results/seed_mismatch_report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import torch

# Make repo importable when running from the workspace root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reliquary.environment.forced_sampling import u_at, seed_consistency_diagnostics
from reliquary.constants import (
    LAYER_INDEX,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
    FORCED_SEED_STOCHASTIC_MAXPROB,
    FORCED_SEED_CDF_BOUNDARY_EPSILON,
)
from reliquary.shared.forward import forward_single_layer
from reliquary.shared.modeling import load_text_generation_model, load_tokenizer


@dataclass
class RolloutEvalResult:
    prompt_idx: int
    rollout_idx: int
    accepted: bool
    checkpoint_hash: str
    randomness: str
    completion_length: int
    prompt_length: int
    n_positions: int
    n_stochastic: int
    n_exact_match: int
    n_boundary_match: int
    n_hard_mismatch: int
    n_deterministic_hard_mismatch: int
    n_miss_gt_0_01: int
    n_miss_gt_0_05: int
    n_miss_gt_0_10: int
    max_cdf_miss: float
    first_hard_mismatch_offset: Optional[int]
    exact_match_rate: float
    boundary_match_rate: float
    hard_mismatch_rate: float
    forced_span: Optional[list[int]]
    forced: bool
    token_count: int
    completion_preview: str


class SeedMismatchEvaluator:
    def __init__(self, model_path: str, tokenizer_path: Optional[str], device: str):
        self.device = torch.device(device)
        dtype = torch.bfloat16 if self.device.type != "cpu" else torch.float32
        if self.device.type == "cpu":
            device_map = None
        else:
            # Force a single-device load so the model stays colocated with the
            # input tensors and does not hit mixed-device errors from auto-sharding.
            device_map = {"": str(self.device)}

        self.model = load_text_generation_model(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
        )
        if self.device.type != "cpu":
            self.model = self.model.to(self.device).eval()
        else:
            self.model = self.model.eval()
        self.tokenizer = load_tokenizer(tokenizer_path or model_path)

    def evaluate_submission_log(self, submission_log: Path, limit: Optional[int] = None) -> list[RolloutEvalResult]:
        results: list[RolloutEvalResult] = []
        count = 0
        with submission_log.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if limit is not None and count >= limit:
                    break
                payload = json.loads(line)
                results.extend(self.evaluate_payload(payload))
                count += 1
        return results

    def evaluate_payload(self, payload: dict[str, Any]) -> list[RolloutEvalResult]:
        request = payload.get("request", {})
        prompt_idx = int(request.get("prompt_idx", -1))
        checkpoint_hash = str(request.get("checkpoint_hash", ""))

        evals: list[RolloutEvalResult] = []
        for rollout_idx, rollout in enumerate(request.get("rollouts", [])):
            commit = rollout.get("commit", {}) or {}
            rollout_meta = commit.get("rollout", {}) or {}
            tokens = list(commit.get("tokens") or [])
            if not tokens:
                continue

            prompt_length = int(rollout_meta.get("prompt_length", 0))
            completion_length = int(rollout_meta.get("completion_length", 0))
            force_span = rollout_meta.get("force_span")
            forced = bool(rollout_meta.get("forced", False))
            beacon = commit.get("beacon", {}) or {}
            randomness = str(beacon.get("randomness", "") or "")
            if isinstance(force_span, (list, tuple)):
                force_span = [int(force_span[0]), int(force_span[1])]
            else:
                force_span = None

            # Match validator's seed_u construction.
            seed_u_values = [
                u_at(
                    randomness,
                    prompt_idx,
                    checkpoint_hash,
                    rollout_idx,
                    j,
                )
                for j in range(completion_length)
            ]

            # Build the token list used by validator's seed-consistency logic.
            completion_tokens = tokens[prompt_length:prompt_length + completion_length]
            if not completion_tokens:
                continue

            # Exclude BFT-injected force_span tokens exactly like the validator.
            fs0, fs1 = 0, 0
            if forced and isinstance(force_span, list) and len(force_span) == 2:
                try:
                    fs0, fs1 = int(force_span[0]), int(force_span[1])
                except (TypeError, ValueError, OverflowError):
                    fs0, fs1 = 0, 0

            valid_t = [
                t for t in range(prompt_length, prompt_length + completion_length)
                if t > 0 and t - 1 < len(tokens) and not (fs0 <= t < fs1)
            ]
            if not valid_t:
                continue

            seed_tokens = [tokens[t] for t in valid_t]
            seed_u = [seed_u_values[t - prompt_length] for t in valid_t]
            seed_positions = [t - prompt_length for t in valid_t]
            logit_positions = [t - 1 for t in valid_t]

            # Replay the validator's logits forward pass on the full token sequence.
            input_ids = torch.tensor([tokens], device=self.device)
            with torch.no_grad():
                _, logits = forward_single_layer(self.model, input_ids, None, LAYER_INDEX)
            if logits.device != self.device:
                logits = logits.to(self.device)

            logits_gpu = logits[0]
            diagnostics = seed_consistency_diagnostics(
                logits_gpu,
                seed_tokens,
                seed_u,
                t=T_PROTO,
                top_k=TOP_K_PROTO,
                top_p=TOP_P_PROTO,
                stochastic_threshold=FORCED_SEED_STOCHASTIC_MAXPROB,
                boundary_epsilon=FORCED_SEED_CDF_BOUNDARY_EPSILON,
                position_offsets=seed_positions,
                logit_positions=logit_positions,
            )

            exact_match_rate = diagnostics.n_exact_match / diagnostics.n_stochastic if diagnostics.n_stochastic else 0.0
            boundary_match_rate = diagnostics.n_boundary_match / diagnostics.n_stochastic if diagnostics.n_stochastic else 0.0
            hard_mismatch_rate = diagnostics.n_hard_mismatch / diagnostics.n_stochastic if diagnostics.n_stochastic else 0.0

            evals.append(
                RolloutEvalResult(
                    prompt_idx=prompt_idx,
                    rollout_idx=rollout_idx,
                    accepted=bool(payload.get("accepted", False)),
                    checkpoint_hash=checkpoint_hash,
                    randomness=randomness,
                    completion_length=completion_length,
                    prompt_length=prompt_length,
                    n_positions=diagnostics.n_positions,
                    n_stochastic=diagnostics.n_stochastic,
                    n_exact_match=diagnostics.n_exact_match,
                    n_boundary_match=diagnostics.n_boundary_match,
                    n_hard_mismatch=diagnostics.n_hard_mismatch,
                    n_deterministic_hard_mismatch=diagnostics.n_deterministic_hard_mismatch,
                    n_miss_gt_0_01=diagnostics.n_miss_gt_0_01,
                    n_miss_gt_0_05=diagnostics.n_miss_gt_0_05,
                    n_miss_gt_0_10=diagnostics.n_miss_gt_0_10,
                    max_cdf_miss=diagnostics.max_cdf_miss,
                    first_hard_mismatch_offset=diagnostics.first_hard_mismatch_offset,
                    exact_match_rate=exact_match_rate,
                    boundary_match_rate=boundary_match_rate,
                    hard_mismatch_rate=hard_mismatch_rate,
                    forced_span=force_span,
                    forced=forced,
                    token_count=len(tokens),
                    completion_preview=self._preview_completion(tokens, prompt_length, completion_length),
                )
            )
        return evals

    @staticmethod
    def _preview_completion(tokens: list[int], prompt_length: int, completion_length: int, max_chars: int = 140) -> str:
        if not tokens:
            return ""
        completion = tokens[prompt_length:prompt_length + completion_length]
        return "..." if len(completion) > max_chars else str(completion)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local forced-seed mismatch evaluator")
    parser.add_argument("--submission-log", type=Path, required=True)
    parser.add_argument("--checkpoint-model", type=str, required=True)
    parser.add_argument("--tokenizer-model", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    evaluator = SeedMismatchEvaluator(
        model_path=args.checkpoint_model,
        tokenizer_path=args.tokenizer_model,
        device=args.device,
    )
    results = evaluator.evaluate_submission_log(args.submission_log, limit=args.limit)

    if args.output_json is not None:
        out_path = args.output_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(r) for r in results]
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {len(results)} rollout evaluations to {out_path}")
    else:
        print(f"Evaluated {len(results)} rollouts")

    if results:
        # Print a compact summary.
        print("\nSummary:")
        print(f"  rollouts={len(results)}")
        print(f"  exact-match-rate avg={sum(r.exact_match_rate for r in results)/len(results):.4f}")
        print(f"  boundary-match-rate avg={sum(r.boundary_match_rate for r in results)/len(results):.4f}")
        print(f"  hard-mismatch-rate avg={sum(r.hard_mismatch_rate for r in results)/len(results):.4f}")
        worst = min(results, key=lambda r: (r.exact_match_rate, r.hard_mismatch_rate))
        print(f"  worst exact-match-rate prompt={worst.prompt_idx} rollout={worst.rollout_idx} rate={worst.exact_match_rate:.4f}")

    return 0


if __name__ == "__main__":
    main()
