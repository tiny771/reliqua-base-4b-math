#!/usr/bin/env python3
"""Fresh test with separate LLM instances to rule out state leakage."""

import sys
import torch
from vllm import LLM

# Add reliquary to path
sys.path.insert(0, "/root/reliquary-miner/reliquary")

from reliquary.miner.vllm_forced_seed_sampler import forced_seed_vllm_sampling_params


def test_with_fresh_llms():
    """Create fresh LLM instance for each mode."""
    model_name = "facebook/opt-125m"
    randomness = "test-window-randomness-12345"
    prompt_idx = 42
    checkpoint_hash = "test-checkpoint-abc123"
    prompt_text = "The answer is"
    max_tokens = 20
    
    print("=" * 80)
    print("FRESH LLM TEST - Sequential and Batch with separate LLM instances")
    print("=" * 80)
    
    # Sequential generation with its own LLM
    print("\n1. SEQUENTIAL GENERATION (fresh LLM)")
    print("-" * 80)
    llm_seq = LLM(
        model=model_name,
        logits_processors=[
            "reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"
        ],
        max_model_len=512,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_batched_tokens=512,
        max_num_seqs=4,
    )
    
    sequential_texts = []
    for rollout_idx in range(4):
        params = forced_seed_vllm_sampling_params(
            randomness=randomness,
            prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash,
            rollout_index=rollout_idx,
            max_tokens=max_tokens,
        )
        output = llm_seq.generate(prompt_text, params)
        text = output[0].outputs[0].text
        sequential_texts.append(text)
        print(f"Rollout {rollout_idx}: '{text}'")
    
    # Clean up
    del llm_seq
    torch.cuda.empty_cache()
    
    # Batch generation with its own LLM
    print("\n2. BATCH GENERATION (fresh LLM)")
    print("-" * 80)
    llm_batch = LLM(
        model=model_name,
        logits_processors=[
            "reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"
        ],
        max_model_len=512,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_batched_tokens=512,
        max_num_seqs=4,
    )
    
    prompts = [prompt_text] * 4
    params_list = [
        forced_seed_vllm_sampling_params(
            randomness=randomness,
            prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash,
            rollout_index=i,
            max_tokens=max_tokens,
        )
        for i in range(4)
    ]
    batch_outputs = llm_batch.generate(prompts, params_list)
    batch_texts = [output.outputs[0].text for output in batch_outputs]
    for i, text in enumerate(batch_texts):
        print(f"Rollout {i}: '{text}'")
    
    # Clean up
    del llm_batch
    torch.cuda.empty_cache()
    
    # Compare
    print("\n3. COMPARISON")
    print("-" * 80)
    all_match = True
    for i in range(4):
        match = sequential_texts[i] == batch_texts[i]
        status = "✓ MATCH" if match else "✗ MISMATCH"
        print(f"Rollout {i}: {status}")
        
        if not match:
            all_match = False
            print(f"  Sequential: '{sequential_texts[i]}'")
            print(f"  Batch:      '{batch_texts[i]}'")
            
            # Find where they diverge
            seq_tokens = sequential_texts[i].split()
            batch_tokens = batch_texts[i].split()
            min_len = min(len(seq_tokens), len(batch_tokens))
            
            diverge_idx = -1
            for j in range(min_len):
                if seq_tokens[j] != batch_tokens[j]:
                    diverge_idx = j
                    break
            
            if diverge_idx >= 0:
                print(f"  Diverges at token {diverge_idx}: '{seq_tokens[diverge_idx]}' vs '{batch_tokens[diverge_idx]}'")
    
    print("\n" + "=" * 80)
    if all_match:
        print("✅ SUCCESS: All rollouts match!")
    else:
        print("❌ FAILURE: Mismatch detected")
    print("=" * 80)
    
    return all_match


if __name__ == "__main__":
    success = test_with_fresh_llms()
    sys.exit(0 if success else 1)
