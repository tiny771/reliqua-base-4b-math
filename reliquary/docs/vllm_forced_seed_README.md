# vLLM Forced Seed Sampler for Reliquary

This directory contains the vLLM implementation of Reliquary's protocol-v2 forced seed sampling mechanism, designed for high-throughput mining with validator-verifiable generation.

## Overview

The forced seed sampler ensures that every token is deterministically sampled from a public, protocol-defined uniform distribution. This allows the validator to verify that generations came from the correct model and checkpoint without miners being able to fabricate or cherry-pick outputs.

## Key Differences: Transformers vs vLLM

| Aspect | Transformers (old) | vLLM (new) |
|--------|-------------------|------------|
| **API** | `LogitsProcessor` (HuggingFace) | `LogitsProcessor` (vLLM v1 API) |
| **Granularity** | Per-token callback in generate loop | Batch-level apply per forward pass |
| **State management** | Instance per generate call | Persistent batch state with update_state |
| **Performance** | Slow (Python loop over rollouts) | Fast (batched GPU operations) |
| **Integration** | `logits_processor=` arg | `logits_processors=` init arg |

## File Structure

```
reliquary/miner/
├── forced_seed_sampler.py          # Original transformers implementation
├── vllm_forced_seed_sampler.py     # New vLLM implementation
reliquary/examples/
└── vllm_forced_seed_example.py     # Usage examples
```

## Installation

Ensure you have vLLM installed:

```bash
pip install vllm>=0.6.0
```

## Basic Usage

### Initialize vLLM with Forced Seed Processor

```python
from vllm import LLM
from reliquary.miner.vllm_forced_seed_sampler import forced_seed_vllm_sampling_params

# Method 1: Pass fully-qualified class name (recommended)
llm = LLM(
    model="Qwen/Qwen3.5-2B",
    logits_processors=[
        "reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"
    ],
    max_model_len=2048,
)

# Method 2: Pass class object (offline only)
from reliquary.miner.vllm_forced_seed_sampler import ForcedSeedVllmLogitsProcessor
llm = LLM(
    model="Qwen/Qwen3.5-2B",
    logits_processors=[ForcedSeedVllmLogitsProcessor],
)
```

### Generate with Forced Seed

```python
# Get parameters from validator /state
window_randomness = state["randomness"]
checkpoint_hash = state["checkpoint_revision"]
prompt_idx = 42  # Your chosen prompt index

# Use helper function to create sampling params
sampling_params = forced_seed_vllm_sampling_params(
    randomness=window_randomness,
    prompt_idx=prompt_idx,
    checkpoint_hash=checkpoint_hash,
    rollout_index=0,  # 0-7 for M_ROLLOUTS=8
    base_offset=0,
    max_tokens=512,
)

output = llm.generate("What is 2+2?", sampling_params)
```

### Generate 8 Rollouts (Protocol Requirement)

```python
outputs = []
for rollout_idx in range(8):  # M_ROLLOUTS = 8
    params = forced_seed_vllm_sampling_params(
        randomness=window_randomness,
        prompt_idx=prompt_idx,
        checkpoint_hash=checkpoint_hash,
        rollout_index=rollout_idx,
        max_tokens=512,
    )
    output = llm.generate(prompt, params)
    outputs.append(output)
```

## Configuration Reference

### SamplingParams.extra_args["forced_seed_config"]

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `randomness` | `str` | Yes | Window randomness from validator `/state.randomness` |
| `prompt_idx` | `int` | Yes | Prompt index in environment (must not be in cooldown) |
| `checkpoint_hash` | `str` | Yes | HF revision from `/state.checkpoint_revision` |
| `rollout_index` | `int` | Yes | Rollout index (0-7 for M_ROLLOUTS=8) |
| `base_offset` | `int` | Yes | Starting completion offset (0 for phase-1, varies for BFT phase-2) |
| `temperature` | `float` | No | Protocol temperature (default: `T_PROTO=1.0`) |
| `top_k` | `int` | No | Protocol top_k (default: `TOP_K_PROTO=50`) |
| `top_p` | `float` | No | Protocol top_p (default: `TOP_P_PROTO=1.0`) |

### Phase-2 BFT base_offset Calculation

For BFT phase-2 generation (continuing from primed sequences):

```python
base_offset = max(0, primed_length - prompt_length)
```

Where:
- `primed_length`: Length of the primed sequence from phase-1
- `prompt_length`: Length of the original prompt tokens

## Implementation Details

### How It Works

1. **Per-request configuration**: Each request can have its own forced seed config passed via `SamplingParams.extra_args`
2. **Batch state management**: The processor maintains sparse state (only configured requests stored)
3. **Deterministic sampling**: For each position `t`:
   ```python
   u = u_at(randomness, prompt_idx, checkpoint_hash, rollout_index, t)
   probs = warp(logits, temperature, top_k, top_p)
   token = pick(probs, u)  # Inverse-CDF
   ```
4. **Logit masking**: All logits set to `-inf` except selected token (set to `0.0`)

### State Lifecycle

```
Request Added → Config extracted from extra_args → Store in req_configs
   ↓
Apply called → For each configured request:
   - Compute u_at for current step
   - Warp logits with protocol params
   - Pick token via inverse-CDF
   - Mask logits to force selection
   - Increment step counter
   ↓
Request Finished → Remove from req_configs
```

### Batch Update Handling

The processor correctly handles vLLM's batch state changes:

1. **Removed**: Clean up finished requests
2. **Added**: Extract and validate forced seed config
3. **Moved**: Track index changes for unidirectional and swap moves

## Mixing Forced and Regular Sampling

The processor gracefully handles mixed batches:

```python
# Some requests use forced seed
forced_params = forced_seed_vllm_sampling_params(...)

# Others use regular sampling (no forced_seed_config)
regular_params = SamplingParams(temperature=0.7, top_p=0.9)

# Both work in same batch
outputs = llm.generate(
    prompts=["Forced prompt", "Regular prompt"],
    sampling_params=[forced_params, regular_params],
)
```

## Performance Considerations

### vLLM Advantages

1. **Batching**: Multiple rollouts processed in parallel
2. **GPU efficiency**: All operations on GPU, minimal CPU overhead
3. **Memory**: Shared KV cache across requests where possible
4. **Throughput**: ~10-50x faster than HuggingFace transformers sequential generation

### Optimization Tips

1. **Batch rollouts**: Generate all 8 rollouts in one batch call if possible
2. **GPU memory**: Adjust `gpu_memory_utilization` based on model size and batch size
3. **Max model length**: Set `max_model_len` to reasonable value to save memory
4. **Tensor parallelism**: Use `tensor_parallel_size` for large models

Example optimized batch generation:

```python
# Generate all 8 rollouts in parallel
prompts = [prompt] * 8
sampling_params = [
    forced_seed_vllm_sampling_params(
        randomness=window_randomness,
        prompt_idx=prompt_idx,
        checkpoint_hash=checkpoint_hash,
        rollout_index=i,
    )
    for i in range(8)
]

outputs = llm.generate(prompts, sampling_params)  # Single batched call
```

## Validation & Testing

### Verify Forced Seed Correctness

The validator will check:
1. **Seed consistency**: Token matches forced draw at stochastic positions (≥80% for group)
2. **GRAIL proof**: Hidden state activations match forward pass
3. **Checkpoint hash**: Submission uses current checkpoint

### Local Testing

```python
# Generate same rollout twice - should be identical
params = forced_seed_vllm_sampling_params(
    randomness="test-seed-123",
    prompt_idx=0,
    checkpoint_hash="abc",
    rollout_index=0,
)

output1 = llm.generate("Test prompt", params)
output2 = llm.generate("Test prompt", params)

assert output1[0].outputs[0].text == output2[0].outputs[0].text
```

## Common Issues

### Issue: "forced_seed_config missing required field"

**Cause**: Incomplete configuration dict

**Fix**: Ensure all required fields present:
```python
config = {
    "randomness": "...",
    "prompt_idx": 42,
    "checkpoint_hash": "...",
    "rollout_index": 0,
    "base_offset": 0,
}
```

### Issue: Validator rejects with SEED_MISMATCH

**Possible causes**:
1. Using wrong `checkpoint_hash` (must match validator's current revision)
2. Using wrong `randomness` (must match current window)
3. Incorrect `base_offset` for BFT phase-2

**Fix**: Always fetch fresh values from `/state` before generation

### Issue: Low throughput compared to expectations

**Possible causes**:
1. Generating rollouts sequentially instead of batching
2. GPU memory constrained (causing swapping)
3. Not using tensor parallelism for large models

**Fix**: See Performance Considerations section above

## Migration from Transformers

If migrating from the old `forced_seed_sampler.py`:

### Before (Transformers)
```python
from reliquary.miner.forced_seed_sampler import ForcedSeedLogitsProcessor

processor = ForcedSeedLogitsProcessor(
    randomness=randomness,
    hotkey=hotkey,
    prompt_idx=prompt_idx,
    checkpoint_hash=checkpoint_hash,
    rollout_indices=[0],
    base_offsets=[0],
    start_len=len(input_ids[0]),
)

output = model.generate(
    input_ids,
    do_sample=False,
    logits_processor=LogitsProcessorList([processor]),
    max_new_tokens=512,
)
```

### After (vLLM)
```python
from vllm import LLM
from reliquary.miner.vllm_forced_seed_sampler import forced_seed_vllm_sampling_params

llm = LLM(
    model="Qwen/Qwen3.5-2B",
    logits_processors=[
        "reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"
    ],
)

params = forced_seed_vllm_sampling_params(
    randomness=randomness,
    prompt_idx=prompt_idx,
    checkpoint_hash=checkpoint_hash,
    rollout_index=0,
    max_tokens=512,
)

output = llm.generate(prompt_text, params)
```

### Key Differences

1. No `hotkey` parameter (protocol-v2 is hotkey-free)
2. No `start_len` (vLLM handles internally)
3. Single rollout per generate call (batch multiple calls for M rollouts)
4. Processor initialized once at LLM creation, not per-generate

## References

- [vLLM Custom Logits Processors Docs](https://docs.vllm.ai/en/stable/features/custom_logitsprocs/)
- [Reliquary Concepts: Forced Seed v2](../docs/concepts.md#forced-seed-v2)
- [Difficulty Auction v2 Design](../docs/superpowers/specs/2026-07-15-difficulty-auction-v2-design.md)
- [Protocol Constants](../reliquary/constants.py)

## License

See [LICENSE.md](../LICENSE.md)
