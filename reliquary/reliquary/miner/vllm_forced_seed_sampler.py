"""vLLM-compatible forced seed logits processor for Reliquary protocol-v2.

This processor works identically to the transformers-based forced_seed_sampler.py
but is designed for vLLM's batch-level logits processor API. It forces each
position's token to be sampled from the protocol's deterministic draw instead of
a local RNG, ensuring validator-verifiable generation.

Usage:
    from reliquary.miner.vllm_forced_seed_sampler import ForcedSeedVllmLogitsProcessor
    from vllm import LLM, SamplingParams
    
    # Initialize vLLM with the processor
    llm = LLM(
        model="Qwen/Qwen3.5-2B",
        logits_processors=["reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"]
    )
    
    # Generate with forced seed configuration in extra_args
    sampling_params = SamplingParams(
        temperature=1.0,  # These will be ignored; processor applies protocol values
        max_tokens=512,
        extra_args={
            "forced_seed_config": {
                "randomness": window_randomness,
                "prompt_idx": prompt_idx,
                "checkpoint_hash": checkpoint_hash,
                "rollout_index": 0,
                "base_offset": 0,
                "temperature": T_PROTO,
                "top_k": TOP_K_PROTO,
                "top_p": TOP_P_PROTO,
            }
        }
    )
    outputs = llm.generate(prompt, sampling_params)
"""
from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment.forced_sampling import pick, u_at, warp


class ForcedSeedVllmLogitsProcessor(LogitsProcessor):
    """vLLM batch-level logits processor that forces tokens from protocol draws.
    
    Each request in the batch may have its own forced seed configuration passed
    via SamplingParams.extra_args["forced_seed_config"]. The processor maintains
    per-request state and applies forced sampling only to requests that provide
    configuration.
    
    Configuration dict structure (all fields required if present):
        {
            "randomness": str,          # Window randomness seed
            "prompt_idx": int,          # Prompt index in environment
            "checkpoint_hash": str,     # HF checkpoint revision hash
            "rollout_index": int,       # Rollout index (0-7 typically)
            "base_offset": int,         # Starting completion offset (usually 0)
            "temperature": float,       # Protocol temperature (default: T_PROTO)
            "top_k": int,              # Protocol top_k (default: TOP_K_PROTO)
            "top_p": float,            # Protocol top_p (default: TOP_P_PROTO)
        }
    """

    @classmethod
    def validate_params(cls, params: SamplingParams):
        """Validate that forced_seed_config in extra_args has correct structure."""
        if not params.extra_args:
            return  # No config = disabled for this request
        
        config = params.extra_args.get("forced_seed_config")
        if config is None:
            return  # No config = disabled for this request
            
        if not isinstance(config, dict):
            raise ValueError(
                f"forced_seed_config must be a dict, got {type(config)}"
            )
        
        required_fields = {
            "randomness": str,
            "prompt_idx": int,
            "checkpoint_hash": str,
            "rollout_index": int,
            "base_offset": int,
        }
        
        for field, expected_type in required_fields.items():
            if field not in config:
                raise ValueError(
                    f"forced_seed_config missing required field: {field}"
                )
            if not isinstance(config[field], expected_type):
                raise ValueError(
                    f"forced_seed_config[{field}] must be {expected_type.__name__}, "
                    f"got {type(config[field]).__name__}"
                )
        
        # Optional fields with defaults
        if "temperature" in config and not isinstance(config["temperature"], (int, float)):
            raise ValueError("forced_seed_config['temperature'] must be numeric")
        if "top_k" in config and not isinstance(config["top_k"], int):
            raise ValueError("forced_seed_config['top_k'] must be int")
        if "top_p" in config and not isinstance(config["top_p"], (int, float)):
            raise ValueError("forced_seed_config['top_p'] must be numeric")

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        """Initialize the forced seed processor.
        
        Args:
            vllm_config: vLLM engine configuration
            device: Device for tensor operations
            is_pin_memory: Whether pinned memory is available
        """
        self.device = device
        
        # Sparse representation: only store state for requests with forced seed enabled
        # Maps batch_index -> request configuration
        self.req_configs: dict[int, dict] = {}
        
        # Track current step count per request (increments each call to apply)
        self.req_step_counts: dict[int, int] = {}

    def is_argmax_invariant(self) -> bool:
        """Return False because forced sampling changes token selection."""
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        """Update internal state based on batch composition changes.
        
        Processes add/remove/move operations to keep req_configs and req_step_counts
        aligned with the current batch indices.
        """
        if not batch_update:
            return
        
        # Process removed requests first
        for index in batch_update.removed:
            self.req_configs.pop(index, None)
            self.req_step_counts.pop(index, None)
        
        # Process added requests
        for index, params, _, _ in batch_update.added:
            if params is None:
                continue
                
            self.validate_params(params)
            
            # Extract configuration if present
            if params.extra_args:
                config = params.extra_args.get("forced_seed_config")
                if config is not None:
                    # Store config with defaults for optional fields
                    self.req_configs[index] = {
                        "randomness": config["randomness"],
                        "prompt_idx": int(config["prompt_idx"]),
                        "checkpoint_hash": config["checkpoint_hash"],
                        "rollout_index": int(config["rollout_index"]),
                        "base_offset": int(config["base_offset"]),
                        "temperature": float(config.get("temperature", T_PROTO)),
                        "top_k": int(config.get("top_k", TOP_K_PROTO)),
                        "top_p": float(config.get("top_p", TOP_P_PROTO)),
                    }
                    self.req_step_counts[index] = 0
                else:
                    # No config = disabled for this request
                    self.req_configs.pop(index, None)
                    self.req_step_counts.pop(index, None)
            else:
                # No extra_args = disabled
                self.req_configs.pop(index, None)
                self.req_step_counts.pop(index, None)
        
        # Process moves only if we have any configured requests
        if self.req_configs:
            # Process move operations
            for adx, bdx, direction in batch_update.moved:
                a_config = self.req_configs.pop(adx, None)
                a_step = self.req_step_counts.pop(adx, None)
                b_config = self.req_configs.pop(bdx, None)
                b_step = self.req_step_counts.pop(bdx, None)
                
                # Move a to b
                if a_config is not None:
                    self.req_configs[bdx] = a_config
                    self.req_step_counts[bdx] = a_step
                
                # For swap moves, move b to a
                if direction == MoveDirectionality.SWAP and b_config is not None:
                    self.req_configs[adx] = b_config
                    self.req_step_counts[adx] = b_step

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply forced seed sampling to the batch logits tensor.
        
        For each request with forced seed configuration, this:
        1. Computes the deterministic uniform value u for this position
        2. Applies protocol warp (temperature, top_k, top_p) to get probabilities
        3. Picks the token via inverse-CDF using u
        4. Masks logits to force selection of that token
        
        Args:
            logits: Batch logits tensor of shape (num_requests, vocab_size)
            
        Returns:
            Modified logits tensor with forced tokens having highest probability
        """
        if not self.req_configs:
            # No requests have forced seed enabled
            return logits
        
        # DEBUG: Log batch shape and indices
        import sys
        print(f"\n[DEBUG] apply batch_size={logits.shape[0]} indices={sorted(self.req_configs.keys())}", file=sys.stderr, flush=True)
        
        # Get indices of requests with forced seed
        indices = list(self.req_configs.keys())
        
        # CRITICAL FIX: Sort indices to ensure deterministic processing order
        indices.sort()
        
        # Process each configured request
        for batch_idx in indices:
            if batch_idx >= logits.shape[0]:
                # Should not happen with correct update_state, but be defensive
                continue
            
            config = self.req_configs[batch_idx]
            step = self.req_step_counts[batch_idx]
            
            # Compute completion position: base_offset + current_step
            t = config["base_offset"] + step
            
            # Compute deterministic uniform value for this position
            u = u_at(
                config["randomness"],
                config["prompt_idx"],
                config["checkpoint_hash"],
                config["rollout_index"],
                t,
            )
            
            # CRITICAL FIX: Clone the logits for this batch index to avoid
            # any interference from in-place modifications during warp/pick
            logits_copy = logits[batch_idx].clone()
            
            # DEBUG: Print logits hash to detect if model outputs differ
            import sys
            logits_hash = hash(tuple(logits_copy.cpu().numpy()[:100]))  # Hash first 100 values
            print(f"[DEBUG] rollout={config['rollout_index']} step={step} logits_hash={logits_hash}", file=sys.stderr, flush=True)
            
            # Apply protocol warp to get probabilities
            probs = warp(
                logits_copy,
                t=config["temperature"],
                top_k=config["top_k"],
                top_p=config["top_p"],
            )
            
            # DEBUG: Print top 3 tokens before picking
            top3_probs, top3_ids = torch.topk(probs, 3)
            print(f"[DEBUG] rollout={config['rollout_index']} step={step} u={u:.6f} top3={[(int(tid), float(tp)) for tid, tp in zip(top3_ids, top3_probs)]}", file=sys.stderr, flush=True)
            
            # Pick token via inverse-CDF
            token_id = pick(probs, u)
            
            # Mask logits to force selection of this token
            # Set all to -inf except the selected token
            logits[batch_idx, :] = float("-inf")
            logits[batch_idx, token_id] = 0.0
            
            # Increment step count for next position
            self.req_step_counts[batch_idx] = step + 1
        
        return logits


def forced_seed_vllm_sampling_params(
    *,
    randomness: str,
    prompt_idx: int,
    checkpoint_hash: str,
    rollout_index: int,
    base_offset: int = 0,
    temperature: float = T_PROTO,
    top_k: int = TOP_K_PROTO,
    top_p: float = TOP_P_PROTO,
    max_tokens: int = 512,
    **kwargs,
) -> SamplingParams:
    """Helper to create SamplingParams with forced seed configuration.
    
    Args:
        randomness: Window randomness seed from validator /state
        prompt_idx: Prompt index in environment
        checkpoint_hash: HF checkpoint revision hash
        rollout_index: Rollout index (0-7 for M_ROLLOUTS=8)
        base_offset: Starting completion offset (0 for phase-1, varies for BFT phase-2)
        temperature: Protocol temperature (default: T_PROTO=1.0)
        top_k: Protocol top_k (default: TOP_K_PROTO=50)
        top_p: Protocol top_p (default: TOP_P_PROTO=1.0)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional SamplingParams arguments (will override defaults)
        
    Returns:
        SamplingParams configured for forced seed generation
    """
    # Default settings for forced seed mode
    defaults = {
        "temperature": 1.0,  # Will be ignored by processor, but vLLM requires valid value
        "max_tokens": max_tokens,
        "extra_args": {
            "forced_seed_config": {
                "randomness": randomness,
                "prompt_idx": prompt_idx,
                "checkpoint_hash": checkpoint_hash,
                "rollout_index": rollout_index,
                "base_offset": base_offset,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
        },
    }
    
    # Merge with user overrides
    defaults.update(kwargs)
    
    return SamplingParams(**defaults)
