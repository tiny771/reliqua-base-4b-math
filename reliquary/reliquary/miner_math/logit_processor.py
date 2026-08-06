from __future__ import annotations

from typing import Any, Optional

import torch

from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import LogitsProcessor, BatchUpdate
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

from reliquary.environment.forced_sampling import u_at, _warp_batch

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO, FORCED_SEED_DOMAIN


class ForcedSeedLogitsProcessor(LogitsProcessor):
    """vLLM v1 batch-level logits processor for protocol-forced sampling."""

    @classmethod
    def validate_params(cls, params: SamplingParams):
        """Validate that required protocol arguments are present in extra_args."""
        if not params.extra_args:
            return

        required_keys = {"randomness", "prompt_idx", "checkpoint_hash",
                         "rollout_index", "base_offset"}
        missing = required_keys - set(params.extra_args.keys())
        if missing:
            raise ValueError(f"ForcedSeedLogitsProcessor requires extra_args: {missing}")

        if not isinstance(params.extra_args.get("rollout_index"), int):
            raise ValueError("rollout_index must be an int")
        if not isinstance(params.extra_args.get("base_offset"), int):
            raise ValueError("base_offset must be an int")

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        self.device = device
        # Sparse state: maps vLLM internal batch index -> protocol state
        self.req_state: dict[int, dict[str, Any]] = {}

        # Protocol constants (fallback to model config if needed, but protocol should dominate)
        self.temperature = T_PROTO
        self.top_k = TOP_K_PROTO
        self.top_p = TOP_P_PROTO

    def is_argmax_invariant(self) -> bool:
        # We actively change the argmax to force the protocol's pick
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        """Track batch reshuffling and retain reference to live output_tok_ids."""

        def extract_protocol_state(
            params: SamplingParams,
            prompt_tok_ids: list[int] | None,
            output_tok_ids: list[int]
        ) -> Optional[dict]:
            if not params.extra_args or "randomness" not in params.extra_args:
                return None  # Disable processor for this request
            # Debug: emit extra_args and current output token list when enabled
            # print(f"[ForcedSeedLogitsProcessor] enabled extra_args={params.extra_args} output_tok_len={len(output_tok_ids) if output_tok_ids is not None else None}")

            return {
                "randomness": params.extra_args["randomness"],
                "prompt_idx": int(params.extra_args["prompt_idx"]),
                "checkpoint_hash": params.extra_args["checkpoint_hash"],
                "rollout_index": int(params.extra_args["rollout_index"]),
                "base_offset": int(params.extra_args["base_offset"]),
                "output_tok_ids": output_tok_ids,
            }

        # process_dict_updates handles Adds, Removes, and Moves automatically.
        # It calls extract_protocol_state for new requests, and safely remaps
        # existing dictionary keys for moved requests.
        process_dict_updates(self.req_state, batch_update, extract_protocol_state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply forced sampling to active requests at batch granularity."""
        if not self.req_state:
            return logits

        rows = []
        u_values = []

        for row_idx, state in self.req_state.items():
            rows.append(row_idx)

            # s is the ground-truth number of tokens generated so far for this request
            s = len(state["output_tok_ids"])
            t = state["base_offset"] + s

            u = u_at(
                state["randomness"],
                state["prompt_idx"],
                state["checkpoint_hash"],
                state["rollout_index"],
                t
            )
            u_values.append(u)

            # if row_idx == rows[0]:
            #     print(
            #         f"[ForcedSeedLogitsProcessor] apply row={row_idx} s={s} base_offset={state['base_offset']} t={t} u={u}"
            #     )

        rows_tensor = torch.tensor(rows, dtype=torch.long, device=logits.device)

        # 1. Extract logits for active rows and cast to float32 for stable numeric
        # warp / softmax / cdf computation (matches validator's float casting).
        active_logits = logits[rows_tensor].float()

        # 2. Warp probabilities (vectorized) — operates in float32
        probs = _warp_batch(active_logits, t=self.temperature, top_k=self.top_k, top_p=self.top_p)

        # 3. Vectorized inverse-CDF pick (operate in float32 explicitly)
        cdf = torch.cumsum(probs, dim=-1)
        u_tensor = torch.tensor(u_values, dtype=torch.float32, device=cdf.device).unsqueeze(-1)

        picked_indices = torch.searchsorted(cdf, u_tensor, right=True)
        picked_indices = torch.clamp(picked_indices, max=probs.shape[-1] - 1).squeeze(-1)

        # if rows_tensor.numel() > 0:
        #     first_picked = int(picked_indices[0].item())
        #     print(
        #         f"[ForcedSeedLogitsProcessor] picked row={rows_tensor[0].item()} token_id={first_picked}"
        #     )

        # 4. Construct forced logits tensor (out-of-place to avoid vLLM caching issues).
        # Preserve original logits dtype when writing forced values to avoid
        # unexpected promotion/rounding differences inside vLLM.
        out_logits = logits.clone()
        neg_inf = torch.tensor(float("-inf"), dtype=logits.dtype, device=logits.device)
        zero_val = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
        out_logits[rows_tensor] = neg_inf
        out_logits[rows_tensor, picked_indices] = zero_val  # Force selection

        return out_logits