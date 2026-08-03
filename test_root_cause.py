"""Deep diagnostic: Check if vLLM produces identical logits in batch vs sequential.

This tests whether the issue is in vLLM's logits generation or in our processor.
"""
import torch
from vllm import LLM, SamplingParams
import numpy as np


def test_raw_logits_comparison():
    """Compare raw logits (before forced seed) between batch and sequential."""
    print("\n" + "="*80)
    print("RAW LOGITS COMPARISON TEST")
    print("="*80)
    
    model_name = "facebook/opt-125m"
    
    print("\nInitializing vLLM WITHOUT forced seed processor:")
    llm = LLM(
        model=model_name,
        max_model_len=512,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_batched_tokens=512,
        max_num_seqs=4,
    )
    print("✓ vLLM initialized")
    
    prompt_text = "The answer is"
    
    # Use greedy decoding to get deterministic results
    params = SamplingParams(
        temperature=0.0,  # Greedy = argmax
        max_tokens=5,
        logprobs=10,  # Get log probabilities
    )
    
    print(f"\nPrompt: '{prompt_text}'")
    print("Generating with greedy decoding (temperature=0)...")
    
    # Sequential generation
    print("\n" + "-"*80)
    print("SEQUENTIAL GENERATION (run 4 times)")
    print("-"*80)
    sequential_outputs = []
    for i in range(4):
        output = llm.generate(prompt_text, params)[0]
        text = output.outputs[0].text
        sequential_outputs.append((text, output))
        print(f"Run {i}: '{text}'")
    
    # Batch generation
    print("\n" + "-"*80)
    print("BATCH GENERATION (4 requests)")
    print("-"*80)
    prompts = [prompt_text] * 4
    params_list = [params] * 4
    batch_outputs = llm.generate(prompts, params_list)
    for i, output in enumerate(batch_outputs):
        text = output.outputs[0].text
        print(f"Run {i}: '{text}'")
    
    # Compare
    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80)
    all_match = True
    for i in range(4):
        seq_text = sequential_outputs[i][0]
        batch_text = batch_outputs[i].outputs[0].text
        match = seq_text == batch_text
        
        if not match:
            all_match = False
            print(f"Run {i}: ✗ MISMATCH")
            print(f"  Sequential: '{seq_text}'")
            print(f"  Batch:      '{batch_text}'")
        else:
            print(f"Run {i}: ✓ MATCH - '{seq_text}'")
    
    print("\n" + "="*80)
    if all_match:
        print("✅ vLLM produces IDENTICAL outputs in batch vs sequential")
        print("   Issue must be in forced seed processor logic")
    else:
        print("❌ vLLM produces DIFFERENT outputs in batch vs sequential")
        print("   This is a vLLM batch-invariance bug, not our processor")
    print("="*80)
    
    return all_match


def test_context_length_effect():
    """Test if the issue is related to different context lengths in batch."""
    print("\n" + "="*80)
    print("CONTEXT LENGTH EFFECT TEST")
    print("="*80)
    
    model_name = "facebook/opt-125m"
    
    # Test with same prompt padded to same length
    print("\nInitializing vLLM...")
    llm = LLM(
        model=model_name,
        max_model_len=512,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    
    # Use different prompt lengths
    prompts = [
        "The answer is",
        "The answer is",  # Same
        "The answer is",  # Same
        "The answer is",  # Same
    ]
    
    params = SamplingParams(
        temperature=0.0,
        max_tokens=5,
    )
    
    print(f"\nAll prompts identical: {len(set(prompts)) == 1}")
    print("Generating in batch...")
    
    batch_outputs = llm.generate(prompts, [params] * 4)
    
    print("\nBatch outputs:")
    texts = []
    for i, output in enumerate(batch_outputs):
        text = output.outputs[0].text
        texts.append(text)
        print(f"  {i}: '{text}'")
    
    # Check if all identical
    all_same = len(set(texts)) == 1
    print(f"\n✓ All outputs identical: {all_same}")
    
    if not all_same:
        print("❌ Even with identical prompts, batch produces different outputs!")
        print("   This is definitely a vLLM issue")
    
    return all_same


def test_processor_state_bug():
    """Check if the processor state management has a bug."""
    print("\n" + "="*80)
    print("PROCESSOR STATE MANAGEMENT TEST")
    print("="*80)
    
    from reliquary.miner.vllm_forced_seed_sampler import forced_seed_vllm_sampling_params
    
    model_name = "facebook/opt-125m"
    
    print("\nInitializing vLLM with forced seed processor...")
    llm = LLM(
        model=model_name,
        logits_processors=[
            "reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"
        ],
        max_model_len=512,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    
    randomness = "test-randomness"
    prompt_idx = 42
    checkpoint_hash = "abc123"
    prompt_text = "The answer is"
    
    # Generate same rollout TWICE in batch
    print("\nTest: Generate same rollout index TWICE in same batch")
    print("If processor state is buggy, they might differ")
    
    prompts = [prompt_text, prompt_text]
    params_list = [
        forced_seed_vllm_sampling_params(
            randomness=randomness,
            prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash,
            rollout_index=1,  # Same rollout
            max_tokens=10,
        ),
        forced_seed_vllm_sampling_params(
            randomness=randomness,
            prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash,
            rollout_index=1,  # Same rollout
            max_tokens=10,
        ),
    ]
    
    batch_outputs = llm.generate(prompts, params_list)
    
    text0 = batch_outputs[0].outputs[0].text
    text1 = batch_outputs[1].outputs[0].text
    
    print(f"\nBatch index 0: '{text0}'")
    print(f"Batch index 1: '{text1}'")
    
    if text0 == text1:
        print("\n✓ Same rollout produces same output in same batch")
        print("  Processor state management seems OK")
        return True
    else:
        print("\n✗ Same rollout produces DIFFERENT output in same batch!")
        print("  This indicates a processor state management bug")
        return False


def main():
    """Run diagnostics to find root cause."""
    print("\n" + "="*80)
    print("DEEP DIAGNOSTIC: ROOT CAUSE ANALYSIS")
    print("="*80)
    
    print("\n" + "="*80)
    print("TEST 1: Raw vLLM logits (without forced seed)")
    print("="*80)
    test_raw_logits_comparison()
    
    print("\n\n" + "="*80)
    print("TEST 2: Context length effects")
    print("="*80)
    test_context_length_effect()
    
    print("\n\n" + "="*80)
    print("TEST 3: Processor state management")
    print("="*80)
    test_processor_state_bug()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
