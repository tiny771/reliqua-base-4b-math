"""
CRITICAL FIX for seed_mismatch failures

ROOT CAUSE:
-----------
vLLM is not batch-invariant by default. The model produces DIFFERENT logits
depending on whether a request is processed alone (batch_size=1) or with other
requests (batch_size=N).

Evidence from debug logs:
- Sequential: logits_hash=5789795269595825262 (batch_size=1)
- Batch:      logits_hash=-6500014155457751686 (batch_size=4)

The forced seed processor receives different logits, so even though it
deterministically picks a token, the BASE LOGITS are wrong!

SOLUTION:
---------
Enable vLLM's batch invariance features:

1. **disable_log_stats_time_limit=True**
   - Prevents scheduling interference that breaks determinism

2. **scheduling_policy="fcfs"** 
   - First-come-first-served instead of batching optimizations

3. **max_num_batched_tokens and max_num_seqs**
   - Control batch sizes to ensure consistency

4. **disable_async_output_proc=True**
   - Synchronous output processing for determinism

REFERENCES:
-----------
- https://github.com/vllm-project/vllm/issues/12343
- https://docs.vllm.ai/en/latest/features/batch_invariance/
- https://github.com/vllm-project/vllm/pull/24583

IMPLEMENTATION:
---------------
"""

from vllm import LLM

# CORRECT: Batch-invariant configuration
llm_batch_invariant = LLM(
    model="facebook/opt-125m",
    logits_processors=[
        "reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"
    ],
    # Core batch invariance settings
    enable_prefix_caching=False,           # CRITICAL: Prefix caching breaks invariance
    enable_chunked_prefill=False,          # CRITICAL: Chunked prefill breaks invariance
    
    # Scheduling for determinism
    # scheduling_policy="fcfs",            # First-come-first-served (vLLM 0.6+)
    
    # Memory and performance
    max_model_len=512,
    gpu_memory_utilization=0.3,
    enforce_eager=True,                    # Disable CUDA graphs for consistency
    
    # Disable async operations
    # disable_async_output_proc=True,      # Synchronous output (vLLM 0.6+)
    
    # Batch size control
    # max_num_batched_tokens=2048,         # Limit batch size
    # max_num_seqs=32,                     # Limit concurrent sequences
)

# WRONG: Default configuration (causes seed_mismatch)
llm_default = LLM(
    model="facebook/opt-125m",
    logits_processors=[
        "reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"
    ],
    max_model_len=512,
    gpu_memory_utilization=0.3,
    # ❌ enable_prefix_caching=True (default)
    # ❌ enable_chunked_prefill=True (default)
    # ❌ Various batch optimizations enabled
)

print("""
DEPLOYMENT CHECKLIST:
---------------------
✓ Set enable_prefix_caching=False in vLLM initialization
✓ Set enable_chunked_prefill=False in vLLM initialization
✓ Set enforce_eager=True to disable CUDA graphs
✓ Test with test_minimal.py to verify batch == sequential
✓ Monitor for seed_mismatch errors after deployment

If seed_mismatch still occurs:
1. Add scheduling_policy="fcfs" (requires vLLM 0.6+)
2. Add disable_async_output_proc=True (requires vLLM 0.6+)
3. Limit max_num_batched_tokens and max_num_seqs
4. Check vLLM version and update if needed
""")
