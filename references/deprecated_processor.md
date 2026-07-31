import json

import torch
from typing import Optional, Dict, List
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

# Import forced-seed primitives from Reliquary
try:
    from reliquary.environment.forced_sampling import u_at
    from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
    FORCED_SEED_AVAILABLE = True
except ImportError:
    FORCED_SEED_AVAILABLE = False


def _prepare_logits(logits: torch.Tensor, *, t: float, top_k: int, top_p: float) -> torch.Tensor:
    """Apply the same warp as the reference sampler and return a probability vector."""
    with torch.inference_mode():
        lg = logits.float() / float(t)
        if top_k and top_k > 0:
            k = min(int(top_k), lg.numel())
            if k < lg.numel():
                kth = torch.topk(lg, k).values[-1]
                lg = torch.where(lg < kth, torch.full_like(lg, float("-inf")), lg)
        probs = torch.softmax(lg, dim=-1)
        if top_p and top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, dim=-1)
            keep_sorted = (cum - sp) < top_p
            keep_sorted[..., -1] = True
            sp = torch.where(keep_sorted, sp, torch.zeros_like(sp))
            probs = torch.zeros_like(probs).scatter(-1, si, sp)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return probs


def _pick_from_logits(logits: torch.Tensor, *, u: float, t: float, top_k: int, top_p: float) -> int:
    """Select the forced token by applying the reference warp and inverse-CDF pick."""
    probs = _prepare_logits(logits, t=t, top_k=top_k, top_p=top_p)
    cdf = torch.cumsum(probs, dim=-1)
    u_tensor = torch.tensor(float(u), device=cdf.device, dtype=cdf.dtype)
    idx = int(torch.searchsorted(cdf, u_tensor, right=True))
    idx = min(idx, probs.numel() - 1)
    return int(idx)


class ForcedSeedLogitsProcessorV2(LogitsProcessor):
    @classmethod
    def validate_params(cls, params: SamplingParams):
        if not FORCED_SEED_AVAILABLE:
            raise ValueError("Forced-seed processor requires reliquary environment")

        extra = params.extra_args or {}
        required = {"start_len", "u_list"}
        missing = required - set(extra.keys())
        if missing:
            raise ValueError(f"Forced-seed processor missing required args: {missing}")

        if not isinstance(extra["start_len"], int):
            raise ValueError("start_len must be int")
        if not isinstance(extra["u_list"], list) and not isinstance(extra["u_list"], str):
            raise ValueError("u_list must be list")

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        # super().__init__(vllm_config, device, is_pin_memory)
        self.req_info: Dict[int, Dict] = {}      # req_idx -> config
        self.req_step: Dict[int, int] = {}       # req_idx -> current step count
        self.device = device
        self.temperature = T_PROTO
        self.top_k = TOP_K_PROTO
        self.top_p = TOP_P_PROTO

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        """Handle batch changes: add/remove/move."""
        if not batch_update:
            return

        # Add new requests
        for index, params, _, _ in batch_update.added:
            assert params is not None
            self.validate_params(params)
            extra = params.extra_args

            start_len = int(extra["start_len"])
            u_list = json.loads(extra["u_list"]) if isinstance(extra["u_list"], str) else extra["u_list"]

            # Flatten nested u_list if needed
            if isinstance(u_list, list) and len(u_list) > 0 and isinstance(u_list[0], list):
                u_list = u_list[0]

            self.req_info[index] = {
                "start_len": start_len,
                "u_list": u_list,
            }
            self.req_step[index] = 0

        # Remove finished requests
        for index in batch_update.removed:
            self.req_info.pop(index, None)
            self.req_step.pop(index, None)

        # Handle moves (batch reordering)
        for adx, bdx, direct in batch_update.moved:
            a_info = self.req_info.pop(adx, None)
            a_step = self.req_step.pop(adx, None)
            b_info = self.req_info.pop(bdx, None)
            b_step = self.req_step.pop(bdx, None)

            if a_info is not None:
                self.req_info[bdx] = a_info
                self.req_step[bdx] = a_step
            if direct == MoveDirectionality.SWAP and b_info is not None:
                self.req_info[adx] = b_info
                self.req_step[adx] = b_step

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_info or not FORCED_SEED_AVAILABLE:
            return logits
        return _apply_forced_tokens(
            logits,
            self.req_info,
            self.req_step,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )


def _pick_from_logits_batch(
    logits: torch.Tensor, 
    u_tensor: torch.Tensor, 
    t: float, 
    top_k: int, 
    top_p: float
) -> torch.Tensor:
    """Fully vectorized batched version of the reference warp-and-pick logic."""
    vocab_size = logits.shape[1]

    with torch.inference_mode():
        lg = logits.float() / float(t)
        if top_k and top_k > 0:
            k = min(int(top_k), vocab_size)
            if k < vocab_size:
                kth = torch.topk(lg, k, dim=-1).values[..., -1:]
                lg = torch.where(lg < kth, torch.full_like(lg, float("-inf")), lg)

        probs = torch.softmax(lg, dim=-1)
        if top_p and top_p < 1.0:
            sp, si = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sp, dim=-1)
            keep_sorted = (cum - sp) < top_p
            keep_sorted[..., -1] = True
            sp = torch.where(keep_sorted, sp, torch.zeros_like(sp))
            probs = torch.zeros_like(probs).scatter(-1, si, sp)

        total = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(total > 0, probs / total, torch.zeros_like(probs))

        cdf = torch.cumsum(probs, dim=-1)
        u_expanded = u_tensor.unsqueeze(-1)
        idx = torch.searchsorted(cdf, u_expanded, right=True)
        idx = torch.clamp(idx, max=vocab_size - 1)
        return idx.squeeze(-1)


def _apply_forced_tokens(
    logits: torch.Tensor,
    req_info: Dict[int, Dict],
    req_step: Dict[int, int],
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if not req_info:
        return logits

    active_reqs: list[int] = []
    active_u: list[float] = []

    for req_idx, cfg in req_info.items():
        step = req_step.get(req_idx, 0)
        t = cfg["start_len"] + step
        u_list = cfg["u_list"]
        if t < len(u_list):
            active_reqs.append(req_idx)
            active_u.append(float(u_list[t]))

    if not active_reqs:
        return logits

    active_indices = torch.tensor(active_reqs, device=logits.device, dtype=torch.long)
    active_u_tensor = torch.tensor(active_u, device=logits.device, dtype=torch.float32)
    active_logits = logits[active_indices].clone()

    picked_tokens = _pick_from_logits_batch(
        active_logits,
        u_tensor=active_u_tensor,
        t=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    active_logits.fill_(float("-inf"))
    active_logits[torch.arange(active_logits.shape[0], device=logits.device), picked_tokens] = 0.0
    logits[active_indices] = active_logits

    for req_idx in active_reqs:
        req_step[req_idx] = req_step.get(req_idx, 0) + 1

    return logits


class ForcedSeedLogitsProcessorV3(LogitsProcessor):
    @classmethod
    def validate_params(cls, params: SamplingParams):
        if not FORCED_SEED_AVAILABLE:
            raise ValueError("Forced-seed processor requires reliquary environment")

        extra = params.extra_args or {}
        required = {"start_len", "u_list"}
        missing = required - set(extra.keys())
        if missing:
            raise ValueError(f"Forced-seed processor missing required args: {missing}")

        if not isinstance(extra["start_len"], int):
            raise ValueError("start_len must be int")
        if not isinstance(extra["u_list"], list) and not isinstance(extra["u_list"], str):
            raise ValueError("u_list must be list")

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        self.req_info: Dict[int, Dict] = {}      # req_idx -> config
        self.req_step: Dict[int, int] = {}       # req_idx -> current step count
        self.device = device
        self.temperature = T_PROTO
        self.top_k = TOP_K_PROTO
        self.top_p = TOP_P_PROTO

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        """Handle batch changes: add/remove/move."""
        if not batch_update:
            return

        # Add new requests
        for index, params, _, _ in batch_update.added:
            assert params is not None
            self.validate_params(params)
            extra = params.extra_args

            start_len = int(extra["start_len"])
            u_list = json.loads(extra["u_list"]) if isinstance(extra["u_list"], str) else extra["u_list"]

            # Flatten nested u_list if needed
            if isinstance(u_list, list) and len(u_list) > 0 and isinstance(u_list[0], list):
                u_list = u_list[0]

            self.req_info[index] = {
                "start_len": start_len,
                "u_list": u_list,
            }
            self.req_step[index] = 0

        # Remove finished requests
        for index in batch_update.removed:
            self.req_info.pop(index, None)
            self.req_step.pop(index, None)

        # Handle moves (batch reordering)
        for adx, bdx, direct in batch_update.moved:
            a_info = self.req_info.pop(adx, None)
            a_step = self.req_step.pop(adx, None)
            b_info = self.req_info.pop(bdx, None)
            b_step = self.req_step.pop(bdx, None)

            if a_info is not None:
                self.req_info[bdx] = a_info
                self.req_step[bdx] = a_step
            if direct == MoveDirectionality.SWAP and b_info is not None:
                self.req_info[adx] = b_info
                self.req_step[adx] = b_step

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_info or not FORCED_SEED_AVAILABLE:
            return logits
        return _apply_forced_tokens(
            logits,
            self.req_info,
            self.req_step,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )