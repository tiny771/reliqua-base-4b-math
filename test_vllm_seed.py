#!/usr/bin/env python3
"""Test if vLLM with same seed produces same logits in sequential vs batch mode."""

import sys
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

def test():
    model_name = "facebook/opt-125m"
    prompt_text = "The answer is"
    seed = 12345  # Same seed for all
    
    print("="*80)
    print("Testing if vLLM with SAME SEED produces identical outputs")
    print("="*80)
    
    # Sequential mode
    print("\n1. SEQUENTIAL MODE with seed=12345")
    print("-"*80)
    llm_seq = LLM(
        model=model_name,
        max_model_len=512,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        seed=seed,  # Set seed
    )
    
    params_seq = SamplingParams(max_tokens=20, temperature=1.0)
    output_seq = llm_seq.generate(prompt_text, params_seq)
    seq_text = output_seq[0].outputs[0].text
    print(f"Sequential output: '{seq_text}'")
    
    del llm_seq
    torch.cuda.empty_cache()
    
    # Batch mode - all 4 requests with same seed
    print("\n2. BATCH MODE with seed=12345 (4 identical requests)")
    print("-"*80)
    llm_batch = LLM(
        model=model_name,
        max_model_len=512,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        seed=seed,  # Same seed
    )
    
    # Generate 4 IDENTICAL requests
    prompts = [prompt_text] * 4
    params_list = [SamplingParams(max_tokens=20, temperature=1.0) for _ in range(4)]
    outputs_batch = llm_batch.generate(prompts, params_list)
    
    batch_texts = [output.outputs[0].text for output in outputs_batch]
    for i, text in enumerate(batch_texts):
        print(f"Batch rollout {i}: '{text}'")
    
    # Compare all batch outputs to sequential
    print("\n3. COMPARISON")
    print("-"*80)
    print(f"Sequential:     '{seq_text}'")
    print()
    
    all_match = True
    for i, batch_text in enumerate(batch_texts):
        if batch_text == seq_text:
            print(f"Batch rollout {i}: ✓ MATCHES sequential")
        else:
            print(f"Batch rollout {i}: ✗ DIFFERS from sequential")
            print(f"  Expected: '{seq_text}'")
            print(f"  Got:      '{batch_text}'")
            all_match = False
    
    print("\n" + "="*80)
    if all_match:
        print("✅ vLLM IS BATCH-INVARIANT with same seed")
    else:
        print("❌ vLLM is NOT batch-invariant even with same seed!")
    print("="*80)
    
    return all_match


if __name__ == "__main__":
    success = test()
    sys.exit(0 if success else 1)
