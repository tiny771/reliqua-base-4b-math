"""Debug version of forced seed processor with extensive logging."""
import torch
from vllm.v1.sample.logits_processor import LogitsProcessor
from typing import Any

from reliquary.environment.forced_sampling import u_at, warp, pick
from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO


class DebugForcedSeedVllmLogitsProcessor(LogitsProcessor):
    """Debug version with logging."""
    
    def __init__(self, vllm_config: Any, device: torch.device, is_pin_memory: bool):
        super().__init__()
        self.vllm_config = vllm_config
        self.device = device
        self.is_pin_memory = is_pin_memory
        self.req_configs: dict[int, dict] = {}
        self.req_step_counts: dict[int, int] = {}
        self.call_count = 0
        print("[DEBUG] Processor initialized")
    
    def is_argmax_invariant(self) -> bool:
        return False
    
    @classmethod
    def validate_params(cls, params):
        """Validate sampling params."""
        if params is None:
            return
        if not hasattr(params, "extra_args") or not params.extra_args:
            return
        config = params.extra_args.get("forced_seed_config")
        if config is None:
            return
        required_fields = ["randomness", "prompt_idx", "checkpoint_hash", "rollout_index", "base_offset"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"forced_seed_config missing required field: {field}")
    
    def update_state(self, batch_update):
        """Update state based on batch changes."""
        if not batch_update:
            return
        
        print(f"[DEBUG] update_state called:")
        print(f"  removed: {batch_update.removed}")
        print(f"  added: {[(idx, params.extra_args.get('forced_seed_config', {}).get('rollout_index', 'N/A') if params and hasattr(params, 'extra_args') and params.extra_args else 'N/A') for idx, params, _, _ in batch_update.added]}")
        
        # Process removed requests
        for index in batch_update.removed:
            self.req_configs.pop(index, None)
            self.req_step_counts.pop(index, None)
        
        # Process added requests
        for index, params, _, _ in batch_update.added:
            if params is None:
                continue
            
            self.validate_params(params)
            
            if params.extra_args:
                config = params.extra_args.get("forced_seed_config")
                if config is not None:
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
                    print(f"  batch_idx={index} -> rollout={config['rollout_index']}")
                else:
                    self.req_configs.pop(index, None)
                    self.req_step_counts.pop(index, None)
            else:
                self.req_configs.pop(index, None)
                self.req_step_counts.pop(index, None)
        
        # Process moves
        if self.req_configs:
            for adx, bdx, direction in batch_update.moved:
                a_config = self.req_configs.pop(adx, None)
                a_step = self.req_step_counts.pop(adx, None)
                b_config = self.req_configs.pop(bdx, None)
                b_step = self.req_step_counts.pop(bdx, None)
                
                if a_config is not None:
                    self.req_configs[bdx] = a_config
                    self.req_step_counts[bdx] = a_step
                
                if direction == 1 and b_config is not None:  # SWAP
                    self.req_configs[adx] = b_config
                    self.req_step_counts[adx] = b_step
        
        print(f"  active configs after update: {list(self.req_configs.keys())}")
    
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply forced seed sampling with debug logging."""
        self.call_count += 1
        
        if not self.req_configs:
            return logits
        
        print(f"\n[DEBUG] apply() call #{self.call_count}:")
        print(f"  logits shape: {logits.shape}")
        print(f"  active configs: {list(self.req_configs.keys())}")
        
        indices = list(self.req_configs.keys())
        
        for batch_idx in indices:
            if batch_idx >= logits.shape[0]:
                print(f"  WARNING: batch_idx={batch_idx} >= logits.shape[0]={logits.shape[0]}")
                continue
            
            config = self.req_configs[batch_idx]
            step = self.req_step_counts[batch_idx]
            t = config["base_offset"] + step
            
            # Compute u
            u = u_at(
                config["randomness"],
                config["prompt_idx"],
                config["checkpoint_hash"],
                config["rollout_index"],
                t,
            )
            
            # Clone logits to avoid interference
            logits_copy = logits[batch_idx].clone()
            
            # Apply warp
            probs = warp(
                logits_copy,
                t=config["temperature"],
                top_k=config["top_k"],
                top_p=config["top_p"],
            )
            
            # Pick token
            token_id = pick(probs, u)
            
            print(f"  batch_idx={batch_idx}, rollout={config['rollout_index']}, step={step}, t={t}, u={u:.6f}, token={token_id}")
            
            # Force token
            logits[batch_idx, :] = float("-inf")
            logits[batch_idx, token_id] = 0.0
            
            # Increment step
            self.req_step_counts[batch_idx] = step + 1
        
        return logits
