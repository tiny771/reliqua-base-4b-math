#!/usr/bin/env python3
"""
Local replay harness: re-run validator logic against stored miner submissions
to identify the exact rejection stage and reason for each recorded failure.

VALIDATION PIPELINE:
  1. Drand round check       - timing/staleness validation
  2. Checkpoint hash         - model checkpoint consistency
  3. Window state            - submission window activity
  4. Preflight termination   - EOS token validation
  5. Reward zone             - reward std in bounds
  6. Distribution suspicious - opposite-reward clone detection
  7. Prompt binding          - canonical vs miner token alignment
  8. Seed mismatch           - forced-seed consistency (requires proofs)

USAGE:
  python scripts/replay_submissions.py [--jsonl /path/to/submission_results.jsonl]

OUTPUT:
  JSON lines with enriched diagnostics showing which validator stage caused
  each rejection, plus the exact conditions that triggered it.
"""

import json
import sys
import time
import math
from pathlib import Path
from typing import Any, TypeAlias
from datetime import datetime


# ============================================================================
# TYPE ALIASES & CONSTANTS
# ============================================================================

CheckResult: TypeAlias = tuple[str | None, dict]  # (reject_reason, diagnostics)
ProofsDict: TypeAlias = dict[int, Any]            # rollout_idx → proof or error

# Model caching
_MODEL: Any = None
_TOKENIZER: Any = None

# Replay behavior
CONTINUE_VERIFY = False

# Model paths
MODEL_PATH = "/mnt/d/models/Qwen3.5-2B"
DEFAULT_ENV_NAME = "openmathinstruct"


# ============================================================================
# SECTION 1: MODEL & TOKENIZER UTILITIES
# ============================================================================

def get_eos_set_from_model(model: Any, tokenizer: Any) -> set[int]:
    """Resolve all stop tokens from model and tokenizer."""
    from reliquary.shared.modeling import resolve_eos_token_ids
    return resolve_eos_token_ids(model, tokenizer)


def get_shared_model_and_tokenizer() -> tuple[Any, Any]:
    """Load and cache the shared text-generation model and tokenizer."""
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        from reliquary.shared.modeling import load_text_generation_model, load_tokenizer
        _MODEL = load_text_generation_model(MODEL_PATH).to("cuda")
        _TOKENIZER = load_tokenizer(MODEL_PATH)
    return _MODEL, _TOKENIZER


# ============================================================================
# SECTION 2: CHEAP VALIDATION CHECKS (no model required)
# ============================================================================

def check_drand_round(
    request: dict,
    batcher_drand_tolerance: int = 0,
    record_timestamp: float | None = None,
) -> CheckResult:
    """Check drand round timing (server.py cheap reject path).
    
    If record_timestamp is provided (from stored submission), compute the
    current round as of that timestamp instead of "now". This lets offline
    replay classify submissions as accepted/rejected as they would have been
    at submission time.
    
    Args:
        request: Submission request dict
        batcher_drand_tolerance: Allowed staleness window
        record_timestamp: UNIX seconds | ISO 8601 string | None (use current time)
    
    Returns:
        (reject_reason, diagnostics) - None means pass
    """
    from reliquary.infrastructure.chain import compute_current_drand_round
    from reliquary.infrastructure.drand import get_current_chain
    
    try:
        chain_info = get_current_chain()
    except Exception:
        # Offline or network issue — assume pass for offline replay
        return None, {"drand_check": "skipped_offline"}
    
    submitted_round = int(request.get("drand_round", 0))
    if submitted_round == 0:
        return None, {"drand_round": 0, "status": "legacy"}
    
    # Parse record timestamp
    if record_timestamp is not None:
        if isinstance(record_timestamp, str):
            dt = datetime.fromisoformat(record_timestamp)
            eval_timestamp = dt.timestamp()
        else:
            eval_timestamp = float(record_timestamp)
    else:
        eval_timestamp = time.time()
    
    current_round = compute_current_drand_round(
        eval_timestamp,
        chain_info["genesis_time"],
        chain_info["period"],
    )
    delta = submitted_round - current_round
    
    if submitted_round > current_round:
        return "future_round", {
            "reason": "future_round",
            "submitted": submitted_round,
            "current_at_eval": current_round,
            "delta": delta,
            "eval_timestamp": eval_timestamp,
        }
    
    if submitted_round < current_round - batcher_drand_tolerance:
        return "stale_round", {
            "reason": "stale_round",
            "submitted": submitted_round,
            "current_at_eval": current_round,
            "delta": delta,
            "tolerance": batcher_drand_tolerance,
            "eval_timestamp": eval_timestamp,
        }
    
    return None, {
        "submitted": submitted_round,
        "current_at_eval": current_round,
        "delta": delta,
        "tolerance": batcher_drand_tolerance,
        "status": "ok",
        "eval_timestamp": eval_timestamp,
    }


def check_checkpoint_hash(request: dict, expected_hash: str | None = None) -> CheckResult:
    """Check checkpoint hash mismatch."""
    if expected_hash and request.get("checkpoint_hash") != expected_hash:
        return "wrong_checkpoint", {
            "submitted": request.get("checkpoint_hash"),
            "expected": expected_hash,
        }
    return None, {}


def check_window_state(request: dict) -> CheckResult:
    """Check if window is still active (simplified: assume OPEN for replay)."""
    return None, {"assumed_window_state": "OPEN"}


def check_reward_zone(request: dict) -> CheckResult:
    """Check if reward std is in zone (simplified)."""
    from reliquary.validator.verifier import rewards_std, is_in_zone
    
    rewards = [float(r["reward"]) for r in request["rollouts"]]
    sigma = rewards_std(rewards)
    
    if not is_in_zone(sigma, bootstrap=False):
        return "out_of_zone", {
            "sigma": sigma,
            "rewards": rewards,
        }
    return None, {}


# ============================================================================
# SECTION 3: MEDIUM VALIDATION CHECKS (uses tokenizer only)
# ============================================================================

def check_preflight_termination(
    request: dict,
    env_name: str = DEFAULT_ENV_NAME,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> CheckResult:
    """Pre-GRAIL HTTP-level termination check (server.py).
    
    Validates EOS token positioning and token logprob thresholds.
    """
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP, MIN_EOS_PROBABILITY
    from reliquary.validator.verifier import is_forced_bft_cap_termination

    if model is None or tokenizer is None:
        try:
            model, tokenizer = get_shared_model_and_tokenizer()
        except Exception as e:
            return "model_load_error", {"error": str(e)}
    
    eos_set = get_eos_set_from_model(model, tokenizer)
    if not eos_set:
        return "no_eos_set", {}
    
    for ri, rollout in enumerate(request["rollouts"]):
        diagnostics = _validate_rollout_termination(
            rollout, eos_set, ri, MAX_NEW_TOKENS_PROTOCOL_CAP, MIN_EOS_PROBABILITY
        )
        if diagnostics["reject_reason"]:
            return diagnostics["reject_reason"], diagnostics
    
    return None, {}


def _validate_rollout_termination(
    rollout: dict,
    eos_set: set[int],
    rollout_idx: int,
    max_tokens_cap: int,
    min_eos_prob: float,
) -> dict:
    """Validate termination for a single rollout."""
    from reliquary.validator.verifier import is_forced_bft_cap_termination
    
    commit = rollout["commit"]
    tokens = list(commit.get("tokens") or [])
    meta = commit.get("rollout", {}) or {}
    prompt_length = int(meta.get("prompt_length", 0))
    completion_length = int(meta.get("completion_length", 0))
    completion = tokens[prompt_length : prompt_length + completion_length]
    
    if not completion:
        return {
            "reject_reason": "bad_schema",
            "diagnostics": {f"rollout_{rollout_idx}": {"reason": "empty_completion"}},
        }
    
    eos_positions = [i for i, token in enumerate(completion) if int(token) in eos_set]
    
    if eos_positions:
        # Validate EOS position (should be at end, only one)
        if len(eos_positions) > 1 or eos_positions[0] != len(completion) - 1:
            return {
                "reject_reason": "bad_termination",
                "diagnostics": {
                    f"rollout_{rollout_idx}": {
                        "reason": "eos_padding",
                        "eos_positions": eos_positions,
                        "completion_length": len(completion),
                    }
                },
            }
        
        # Validate final token logprob
        claimed_lp = meta.get("token_logprobs")
        if claimed_lp and len(claimed_lp) > 0:
            final_lp = float(claimed_lp[-1])
            min_lp = math.log(min_eos_prob)
            if math.isfinite(final_lp) and final_lp < min_lp:
                return {
                    "reject_reason": "bad_termination",
                    "diagnostics": {
                        f"rollout_{rollout_idx}": {
                            "reason": "final_token_lp_too_low",
                            "claimed_lp": final_lp,
                            "min_lp": min_lp,
                        }
                    },
                }
    else:
        # No EOS found - check if cap termination or insufficient length
        total_length = prompt_length + completion_length
        if not is_forced_bft_cap_termination(commit) and total_length < max_tokens_cap:
            return {
                "reject_reason": "bad_termination",
                "diagnostics": {
                    f"rollout_{rollout_idx}": {
                        "reason": "no_natural_eos_no_cap",
                        "prompt_length": prompt_length,
                        "completion_length": completion_length,
                        "total_length": total_length,
                        "cap": max_tokens_cap,
                    }
                },
            }
    
    return {"reject_reason": None, "diagnostics": {}}


def check_distribution_suspicious(
    request: dict,
    tokenizer: Any | None = None,
) -> CheckResult:
    """Detect manufactured opposite-reward clone patterns across rollouts."""
    from reliquary.validator.rollout_patterns import detect_opposite_reward_clones

    if tokenizer is None:
        try:
            _, tokenizer = get_shared_model_and_tokenizer()
        except Exception as e:
            return "model_load_error", {"error": str(e)}

    completion_texts = []
    rewards = []
    for rollout in request.get("rollouts", []):
        commit = rollout.get("commit", {})
        rollout_meta = commit.get("rollout", {}) or {}
        prompt_length = int(rollout_meta.get("prompt_length", 0) or 0)
        tokens = list(commit.get("tokens") or [])
        completion_texts.append(tokenizer.decode(tokens[prompt_length:]))
        rewards.append(float(rollout.get("reward", 0.0)))

    metrics = detect_opposite_reward_clones(completion_texts, rewards)
    diagnostics = metrics.to_log_dict()

    if metrics.suspicious:
        return "distribution_suspicious", diagnostics
    return None, diagnostics


def check_prompt_binding(request: dict) -> CheckResult:
    """Check prompt binding (canonical vs miner tokens).
    
    Note: Skipped in offline replay mode.
    """
    return None, {"prompt_binding": "skipped_in_replay"}


# ============================================================================
# SECTION 4: HEAVY COMPUTATION - COMMITMENT PROOFS
# ============================================================================

def compute_commitment_proofs_for_request(
    request: dict,
    model: Any,
    tokenizer: Any,
    randomness: str,
) -> ProofsDict:
    """Compute commitment proofs for all rollouts at once (heavy operation).
    
    This is computed once and shared across multiple validation functions
    to avoid redundant expensive calls to verify_commitment_proofs.
    
    Args:
        request: Submission request dict
        model: Text generation model
        tokenizer: Tokenizer
        randomness: Beacon randomness value
    
    Returns:
        Dict[rollout_idx] → proof object or error dict with status field
    """
    from reliquary.environment.forced_sampling import u_at
    from reliquary.validator.batcher import _is_missing_kwarg_typeerror
    from reliquary.validator.verifier import verify_commitment_proofs

    checkpoint_hash = request.get("checkpoint_hash", "")
    miner_hotkey = request.get("miner_hotkey", "")
    prompt_idx = int(request.get("prompt_idx", 0))
    
    proofs = {}
    
    for ri, rollout in enumerate(request.get("rollouts", [])):
        commit = rollout.get("commit", {})
        rollout_meta = commit.get("rollout", {}) or {}
        completion_length = int(rollout_meta.get("completion_length", 0))
        
        if completion_length <= 0:
            proofs[ri] = {"status": "no_completion", "error": None}
            continue
        
        seed_u = [
            u_at(randomness, miner_hotkey, prompt_idx, checkpoint_hash, ri, j)
            for j in range(completion_length)
        ]
        
        proofs[ri] = _verify_commitment_with_fallback(
            commit, model, randomness, tokenizer, seed_u
        )
    
    return proofs


def _verify_commitment_with_fallback(
    commit: dict,
    model: Any,
    randomness: str,
    tokenizer: Any,
    seed_u: list,
) -> dict:
    """Try to verify commitment proofs with seed_u, falling back if needed."""
    from reliquary.validator.batcher import _is_missing_kwarg_typeerror
    from reliquary.validator.verifier import verify_commitment_proofs
    
    try:
        return verify_commitment_proofs(
            commit,
            model,
            randomness,
            tokenizer=tokenizer,
            seed_u_values=seed_u,
        )
    except TypeError as exc:
        if not _is_missing_kwarg_typeerror(exc, "seed_u_values"):
            return {"status": "verification_error", "error": str(exc)}
        
        try:
            return verify_commitment_proofs(
                commit,
                model,
                randomness,
                tokenizer=tokenizer,
            )
        except Exception as exc2:
            return {"status": "verification_error", "error": str(exc2)}
    except Exception as exc:
        return {"status": "verification_error", "error": str(exc)}


# ============================================================================
# SECTION 5: HEAVY VALIDATION CHECKS (uses proofs)
# ============================================================================

def check_seed_mismatch(
    request: dict,
    proofs: ProofsDict,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> CheckResult:
    """Check forced-seed consistency across rollouts.
    
    Args:
        request: Submission request dict
        proofs: Pre-computed commitment proofs for each rollout
        model: Text generation model (optional, for loading if needed)
        tokenizer: Tokenizer (optional, for loading if needed)
    
    Returns:
        (reject_reason, diagnostics) - None means pass
    """
    from reliquary.constants import (
        FORCED_SEED_ENFORCE,
        FORCED_SEED_CONSISTENCY_FLOOR,
        FORCED_SEED_MIN_STOCH_POSITIONS,
        FORCED_SEED_ROLLOUT_FLOOR,
        FORCED_SEED_ROLLOUT_MIN_STOCH,
    )

    if model is None or tokenizer is None:
        try:
            model, tokenizer = get_shared_model_and_tokenizer()
        except Exception as e:
            return "model_load_error", {"error": str(e)}

    protocol_version = int(request.get("protocol_version", 0))
    if protocol_version == 0:
        return None, {"status": "protocol_version_0"}

    checkpoint_hash = request.get("checkpoint_hash", "")
    randomness = _extract_randomness_from_request(request)
    if not randomness:
        return None, {"status": "no_randomness"}

    seed_enforce = bool(checkpoint_hash) and FORCED_SEED_ENFORCE
    
    diagnostics = {
        "seed_enforce": seed_enforce,
        "randomness": randomness,
        "checkpoint_hash": checkpoint_hash,
        "prompt_idx": request.get("prompt_idx"),
        "miner_hotkey": request.get("miner_hotkey"),
        "group_seed_n_stochastic": 0,
        "group_seed_n_match": 0,
        "rollouts": [],
    }

    if not seed_enforce:
        return None, diagnostics

    # Process all rollouts with pre-computed proofs
    grp_stoch, grp_match, rollout_data = _process_proof_rollouts(proofs, request)
    diagnostics["group_seed_n_stochastic"] = grp_stoch
    diagnostics["group_seed_n_match"] = grp_match
    diagnostics["rollouts"] = rollout_data

    if grp_stoch < FORCED_SEED_MIN_STOCH_POSITIONS:
        diagnostics["status"] = "thin_signal"
        return None, diagnostics

    group_reject = (grp_match / grp_stoch) < FORCED_SEED_CONSISTENCY_FLOOR
    rollout_reject = any(
        r["seed_n_stochastic"] >= FORCED_SEED_ROLLOUT_MIN_STOCH
        and (r["seed_n_match"] / r["seed_n_stochastic"]) < FORCED_SEED_ROLLOUT_FLOOR
        for r in rollout_data
    )

    if group_reject or rollout_reject:
        diagnostics["reason"] = "forced_seed"
        diagnostics["group_reject"] = group_reject
        diagnostics["rollout_reject"] = rollout_reject
        return "seed_mismatch", diagnostics

    diagnostics["status"] = "ok"
    return None, diagnostics


def _extract_randomness_from_request(request: dict) -> str | None:
    """Extract randomness from request rollouts."""
    for rollout in request.get("rollouts", []):
        beacon = rollout.get("commit", {}).get("beacon", {}) or {}
        randomness = beacon.get("randomness", "")
        if randomness:
            return randomness
    return None


def _process_proof_rollouts(proofs: ProofsDict, request: dict) -> tuple[int, int, list]:
    """Process all rollouts with pre-computed proofs.
    
    Returns (total_stoch, total_match, rollout_data_list)
    """
    grp_stoch = 0
    grp_match = 0
    rollout_data = []

    for ri, _ in enumerate(request.get("rollouts", [])):
        if ri not in proofs:
            rollout_data.append({
                "rollout_idx": ri,
                "seed_n_stochastic": 0,
                "seed_n_match": 0,
                "status": "no_rollout_index",
            })
            continue
        
        proof = proofs[ri]
        
        # Skip if proof computation failed or no completion
        if isinstance(proof, dict) and proof.get("status") in ("verification_error", "no_completion"):
            rollout_data.append({
                "rollout_idx": ri,
                "seed_n_stochastic": 0,
                "seed_n_match": 0,
                "status": proof.get("status", "unknown"),
            })
            continue
        
        grp_stoch += proof.seed_n_stochastic
        grp_match += proof.seed_n_match
        rollout_data.append({
            "rollout_idx": ri,
            "seed_n_stochastic": proof.seed_n_stochastic,
            "seed_n_match": proof.seed_n_match,
            "seed_n_positions": proof.seed_n_positions,
            "seed_n_miss_gt_0_01": proof.seed_n_miss_gt_0_01,
            "seed_n_miss_gt_0_05": proof.seed_n_miss_gt_0_05,
            "seed_n_miss_gt_0_10": proof.seed_n_miss_gt_0_10,
        })

    return grp_stoch, grp_match, rollout_data


# ============================================================================
# SECTION 6: REPLAY ORCHESTRATION
# ============================================================================

def replay_submission(
    record: dict,
    verbose: bool = False,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> dict:
    """Run all validation checks on one submission record.
    
    Executes the validation pipeline in order:
    1. Cheap checks (timing, hash, window)
    2. Medium checks (termination, rewards, distribution)
    3. Heavy checks (seed mismatch with proofs)
    
    Args:
        record: Stored submission record with request + metadata
        verbose: Enable verbose output
        model: Text generation model
        tokenizer: Tokenizer
    
    Returns:
        Result dict with rejection info and diagnostics per stage
    """
    request = record["request"]
    record_timestamp = record.get("timestamp")
    
    result = {
        "merkle_root": request.get("merkle_root"),
        "prompt_idx": request.get("prompt_idx"),
        "drand_round": request.get("drand_round"),
        "timestamp": record_timestamp,
        "stored_reason": record.get("reason", "unknown"),
        "accepted_recorded": record.get("accepted", False),
        "stages": {},
    }
    
    # ---- Cheap Checks (no models) ----
    _run_check(result, "drand_round", check_drand_round, request, record_timestamp=record_timestamp)
    if CONTINUE_VERIFY and result["stages"]["drand_round"]["reject"]:
        return _finalize_result(result, "drand_round")
    
    _run_check(result, "checkpoint", check_checkpoint_hash, request)
    if CONTINUE_VERIFY and result["stages"]["checkpoint"]["reject"]:
        return _finalize_result(result, "checkpoint")
    
    _run_check(result, "window_state", check_window_state, request)
    if CONTINUE_VERIFY and result["stages"]["window_state"]["reject"]:
        return _finalize_result(result, "window_state")
    
    # ---- Medium Checks (tokenizer only) ----
    _run_check(result, "termination_preflight", check_preflight_termination, request, model=model, tokenizer=tokenizer)
    if CONTINUE_VERIFY and result["stages"]["termination_preflight"]["reject"]:
        return _finalize_result(result, "preflight")
    
    _run_check(result, "zone", check_reward_zone, request)
    if CONTINUE_VERIFY and result["stages"]["zone"]["reject"]:
        return _finalize_result(result, "zone")
    
    _run_check(result, "distribution_suspicious", check_distribution_suspicious, request, tokenizer=tokenizer)
    if CONTINUE_VERIFY and result["stages"]["distribution_suspicious"]["reject"]:
        return _finalize_result(result, "distribution_suspicious")
    
    _run_check(result, "prompt_binding", check_prompt_binding, request)
    if CONTINUE_VERIFY and result["stages"]["prompt_binding"]["reject"]:
        return _finalize_result(result, "prompt_binding")
    
    # ---- Heavy Check (compute proofs once) ----
    randomness = _extract_randomness_from_request(request)
    proofs = {}
    if randomness and model is not None and tokenizer is not None:
        proofs = compute_commitment_proofs_for_request(
            request, model, tokenizer, randomness
        )
    
    _run_check(result, "seed_mismatch", check_seed_mismatch, request, proofs=proofs, model=model, tokenizer=tokenizer)
    if CONTINUE_VERIFY and result["stages"]["seed_mismatch"]["reject"]:
        return _finalize_result(result, "seed_mismatch")
    
    # No rejects - would proceed to GRAIL
    result["inferred_reason"] = "would_proceed_to_grail"
    result["inferred_stage"] = "grail_or_later"
    return result


def _run_check(result: dict, stage_name: str, check_fn, request, **kwargs) -> None:
    """Run a single check and store result."""
    reason, diagnostics = check_fn(request, **kwargs)
    result["stages"][stage_name] = {
        "reject": reason,
        "diagnostics": diagnostics,
    }


def _finalize_result(result: dict, stage_name: str) -> dict:
    """Finalize result when rejection found."""
    result["inferred_reason"] = result["stages"][stage_name]["reject"]
    result["inferred_stage"] = stage_name
    return result


# ============================================================================
# SECTION 7: MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point: parse args, process submissions, output summary."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Replay validator logic against stored submissions"
    )
    parser.add_argument(
        "--jsonl",
        default="results/submission_results.jsonl",
        help="Path to submission JSONL log",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file (default: replay_diagnostics.jsonl in same dir)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress to stdout",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N records (for testing)",
    )
    
    args = parser.parse_args()
    
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found", file=sys.stderr)
        sys.exit(1)
    
    output_path = args.output or jsonl_path.parent / "replay_diagnostics.jsonl"
    
    # Load model once
    model, tokenizer = get_shared_model_and_tokenizer()
    
    # Process submissions
    summary = _process_submissions(
        jsonl_path, output_path, model, tokenizer, args.limit, args.verbose
    )
    
    # Print summary
    _print_summary(output_path, summary)


def _process_submissions(
    jsonl_path: Path,
    output_path: Path,
    model: Any,
    tokenizer: Any,
    limit: int | None,
    verbose: bool,
) -> dict:
    """Process all submissions and return summary."""
    summary = {
        "total": 0,
        "accepted_recorded": 0,
        "rejected_recorded": 0,
        "stages": {},
        "inferred_stages": {},
    }
    
    with open(output_path, "w") as out:
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                result = replay_submission(
                    record,
                    verbose=verbose,
                    model=model,
                    tokenizer=tokenizer,
                )
                out.write(json.dumps(result) + "\n")
                
                # Update summary
                _update_summary(summary, result)
                
                if verbose and (i + 1) % 50 == 0:
                    print(f"Processed {i + 1} records...", file=sys.stderr)
    
    return summary


def _update_summary(summary: dict, result: dict) -> None:
    """Update summary with result from one submission."""
    summary["total"] += 1
    if result["accepted_recorded"]:
        summary["accepted_recorded"] += 1
    else:
        summary["rejected_recorded"] += 1
    
    inferred = result.get("inferred_stage", "unknown")
    summary["inferred_stages"][inferred] = summary["inferred_stages"].get(inferred, 0) + 1
    
    for stage_name, stage_result in result.get("stages", {}).items():
        if stage_result.get("reject"):
            key = f"{stage_name}:{stage_result['reject']}"
            summary["stages"][key] = summary["stages"].get(key, 0) + 1


def _print_summary(output_path: Path, summary: dict) -> None:
    """Print execution summary to stdout."""
    print(f"\nReplayed {summary['total']} submissions → {output_path}")
    print(f"  Accepted: {summary['accepted_recorded']}")
    print(f"  Rejected: {summary['rejected_recorded']}")
    print(f"\nInferred rejection stages:")
    for stage, count in sorted(summary["inferred_stages"].items(), key=lambda x: -x[1]):
        print(f"  {stage}: {count}")
    print(f"\nDetailed reject reasons:")
    for reason, count in sorted(summary["stages"].items(), key=lambda x: -x[1])[:20]:
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
