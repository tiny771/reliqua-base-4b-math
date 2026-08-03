#!/usr/bin/env python3
"""Analyze raw logits from vLLM without any logit processor."""

import sys
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams


def test_without_processor():
    """Test vLLM without any logit processor to see raw logits."""
    model_name = "facebook/opt-125m"
    prompt_text = "The answer is"
    
    print("="*80)
    print("ANALYZING RAW LOGITS FROM vLLM (NO PROCESSOR)")
    print("="*80)
    
    # Sequential mode
    print("\n1. SEQUENTIAL MODE (single request, greedy decoding)")
    print("-"*80)
    
    llm_seq = LLM(
        model=model_name,
        max_model_len=512,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        seed=12345,
    )
    
    # Use greedy decoding (temperature=0) and request logprobs
    params_seq = SamplingParams(
        max_tokens=10, 
        temperature=0.0,  # Greedy = deterministic
        logprobs=5,  # Get top 5 logprobs at each position
    )
    output_seq = llm_seq.generate(prompt_text, params_seq)
    seq_text = output_seq[0].outputs[0].text
    seq_tokens = output_seq[0].outputs[0].token_ids
    seq_logprobs = output_seq[0].outputs[0].logprobs
    
    print(f"Output: '{seq_text}'")
    print(f"Tokens: {seq_tokens}")
    print(f"\nToken-by-token logprobs:")
    for i, (token_id, logprob_dict) in enumerate(zip(seq_tokens, seq_logprobs)):
        if logprob_dict:
            # Get the selected token's logprob
            selected_logprob = logprob_dict.get(token_id)
            if selected_logprob:
                print(f"  Step {i}: token={token_id} logprob={selected_logprob.logprob:.4f}")
    
    del llm_seq
    torch.cuda.empty_cache()
    
    # Batch mode
    print("\n2. BATCH MODE (4 identical requests, greedy decoding)")
    print("-"*80)
    
    llm_batch = LLM(
        model=model_name,
        max_model_len=512,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        seed=12345,
    )
    
    # 4 identical requests
    prompts = [prompt_text] * 4
    params_list = [SamplingParams(max_tokens=10, temperature=0.0, logprobs=5) for _ in range(4)]
    outputs_batch = llm_batch.generate(prompts, params_list)
    
    batch_texts = [output.outputs[0].text for output in outputs_batch]
    batch_tokens_list = [output.outputs[0].token_ids for output in outputs_batch]
    
    print(f"Outputs:")
    for i, (text, tokens) in enumerate(zip(batch_texts, batch_tokens_list)):
        print(f"  Rollout {i}: '{text}'")
        print(f"    Tokens: {tokens}")
    
    # Analysis
    print("\n3. COMPARISON")
    print("-"*80)
    
    print(f"\nSequential tokens: {seq_tokens}")
    print(f"Sequential output: '{seq_text}'")
    print()
    
    all_match_seq = True
    all_match_each_other = True
    
    for i, (batch_text, batch_tokens) in enumerate(zip(batch_texts, batch_tokens_list)):
        if batch_tokens == seq_tokens:
            print(f"Rollout {i}: ✓ MATCHES sequential")
        else:
            print(f"Rollout {i}: ✗ DIFFERS from sequential")
            print(f"  Expected tokens: {seq_tokens}")
            print(f"  Got tokens:      {batch_tokens}")
            all_match_seq = False
    
    # Check if all batch outputs match each other
    print("\n4. BATCH INTERNAL CONSISTENCY")
    print("-"*80)
    first_tokens = batch_tokens_list[0]
    for i, tokens in enumerate(batch_tokens_list[1:], 1):
        if tokens == first_tokens:
            print(f"Rollout {i} vs Rollout 0: ✓ MATCH")
        else:
            print(f"Rollout {i} vs Rollout 0: ✗ DIFFER")
            all_match_each_other = False
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    
    if all_match_seq:
        print("✅ ALL batch rollouts match sequential (vLLM is batch-invariant with greedy)")
    else:
        print("⚠️  Batch rollouts differ from sequential")
        if batch_tokens_list[0] == seq_tokens:
            print("   - Rollout 0 matches (expected)")
            print("   - Other rollouts differ (they use different RNG streams)")
        else:
            print("   - Even rollout 0 differs (UNEXPECTED!)")
    
    if all_match_each_other:
        print("✅ All batch rollouts are identical to each other")
    else:
        print("⚠️  Batch rollouts differ from each other (different RNG streams)")
    
    print("\nNOTE: With temperature=0 (greedy), all outputs should be identical.")
    print("If they differ, vLLM's model evaluation itself may be non-deterministic.")
    print("="*80)
    
    del llm_batch


if __name__ == "__main__":
    test_without_processor()
