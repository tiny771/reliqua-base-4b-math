"""Comprehensive test to diagnose batch vs sequential mismatch.

Tests with all batch-variance-breaking features disabled as per:
- https://github.com/vllm-project/vllm/issues/12343
- https://docs.vllm.ai/en/latest/features/batch_invariance/
- https://github.com/vllm-project/vllm/pull/24583
"""
import torch
from vllm import LLM

from reliquary.miner.vllm_forced_seed_sampler import forced_seed_vllm_sampling_params
from reliquary.environment.forced_sampling import u_at, warp, pick


def test_batch_invariance_strict():
    """Test batch invariance with ALL variance-breaking features disabled."""
    print("\n" + "="*80)
    print("STRICT BATCH INVARIANCE TEST")
    print("="*80)
    
    model_name = "facebook/opt-125m"
    
    print("\nInitializing vLLM with batch-invariance-safe settings:")
    print("  - enable_prefix_caching=False (no KV cache reuse)")
    print("  - enable_chunked_prefill=False (no chunking)")
    print("  - enforce_eager=True (no CUDA graphs)")
    print("  - max_num_batched_tokens=512 (fixed batch size)")
    print("  - max_num_seqs=4 (fixed sequence count)")
    
    try:
        llm = LLM(
            model=model_name,
            logits_processors=[
                "reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"
            ],
            max_model_len=512,
            gpu_memory_utilization=0.3,
            enforce_eager=True,
            enable_prefix_caching=False,  # CRITICAL: Disable prefix caching
            enable_chunked_prefill=False,  # CRITICAL: Disable chunked prefill
            max_num_batched_tokens=512,  # Fixed batch size
            max_num_seqs=4,  # Fixed sequence count
        )
        print("✓ vLLM initialized with strict batch-invariance settings\n")
    except Exception as e:
        print(f"✗ Failed to initialize vLLM: {e}")
        return False
    
    # Test configuration
    randomness = "test-window-randomness-12345"
    prompt_idx = 42
    checkpoint_hash = "test-checkpoint-abc123"
    prompt_text = "The answer is"
    max_tokens = 20  # Full sequence
    
    print(f"Test configuration:")
    print(f"  Prompt: '{prompt_text}'")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Rollouts: 4")
    print(f"  Window randomness: {randomness}")
    
    # Sequential generation
    print("\n" + "-"*80)
    print("1. SEQUENTIAL GENERATION")
    print("-"*80)
    sequential_outputs = []
    sequential_texts = []
    for rollout_idx in range(4):
        params = forced_seed_vllm_sampling_params(
            randomness=randomness,
            prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash,
            rollout_index=rollout_idx,
            max_tokens=max_tokens,
        )
        output = llm.generate(prompt_text, params)
        text = output[0].outputs[0].text
        sequential_outputs.append(output[0])
        sequential_texts.append(text)
        print(f"Rollout {rollout_idx}: '{text}'")
    
    # Batch generation
    print("\n" + "-"*80)
    print("2. BATCH GENERATION")
    print("-"*80)
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
    batch_outputs = llm.generate(prompts, params_list)
    batch_texts = [output.outputs[0].text for output in batch_outputs]
    for i, text in enumerate(batch_texts):
        print(f"Rollout {i}: '{text}'")
    
    # Compare
    print("\n" + "-"*80)
    print("3. COMPARISON")
    print("-"*80)
    all_match = True
    for i in range(4):
        match = sequential_texts[i] == batch_texts[i]
        status = "✓ MATCH" if match else "✗ MISMATCH"
        print(f"\nRollout {i}: {status}")
        
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
            else:
                print(f"  Lengths differ: {len(seq_tokens)} vs {len(batch_tokens)}")
    
    print("\n" + "="*80)
    if all_match:
        print("✅ SUCCESS: All rollouts match with strict batch-invariance settings!")
        print("="*80)
        return True
    else:
        print("❌ FAILURE: Still seeing batch vs sequential mismatch")
        print("="*80)
        return False


def test_token_by_token_comparison():
    """Generate token-by-token to see exactly where divergence happens."""
    print("\n" + "="*80)
    print("TOKEN-BY-TOKEN DIAGNOSTIC TEST")
    print("="*80)
    
    model_name = "facebook/opt-125m"
    
    try:
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
            max_num_batched_tokens=512,
            max_num_seqs=4,
        )
    except Exception as e:
        print(f"Failed to init vLLM: {e}")
        return False
    
    randomness = "test-window-randomness-12345"
    prompt_idx = 42
    checkpoint_hash = "test-checkpoint-abc123"
    prompt_text = "The answer is"
    rollout_idx = 1  # Test rollout 1 since it showed mismatch
    
    print(f"\nTesting Rollout {rollout_idx} token-by-token:")
    print(f"Prompt: '{prompt_text}'")
    
    # Generate tokens one at a time
    print("\n" + "-"*80)
    print("SEQUENTIAL TOKEN-BY-TOKEN")
    print("-"*80)
    for num_tokens in range(1, 6):
        params = forced_seed_vllm_sampling_params(
            randomness=randomness,
            prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash,
            rollout_index=rollout_idx,
            max_tokens=num_tokens,
        )
        output = llm.generate(prompt_text, params)
        text = output[0].outputs[0].text
        print(f"Tokens 1-{num_tokens}: '{text}'")
    
    # Now batch generation with increasing lengths
    print("\n" + "-"*80)
    print("BATCH TOKEN-BY-TOKEN (4 rollouts with 5 tokens each)")
    print("-"*80)
    prompts = [prompt_text] * 4
    params_list = [
        forced_seed_vllm_sampling_params(
            randomness=randomness,
            prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash,
            rollout_index=i,
            max_tokens=5,
        )
        for i in range(4)
    ]
    batch_outputs = llm.generate(prompts, params_list)
    for i, output in enumerate(batch_outputs):
        text = output.outputs[0].text
        print(f"Rollout {i}: '{text}'")
    
    return True


def test_forced_token_probabilities():
    """Check the probabilities of forced tokens to see if low-prob tokens cause issues."""
    print("\n" + "="*80)
    print("FORCED TOKEN PROBABILITY ANALYSIS")
    print("="*80)
    
    model_name = "facebook/opt-125m"
    
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=model_name,
            max_model_len=512,
            gpu_memory_utilization=0.3,
            enforce_eager=True,
        )
        
        # Get tokenizer
        tokenizer = llm.get_tokenizer()
        
        print("\nAnalyzing forced token selections:")
        print("-"*80)
        
        randomness = "test-window-randomness-12345"
        prompt_idx = 42
        checkpoint_hash = "test-checkpoint-abc123"
        prompt_text = "The answer is"
        
        # Tokenize prompt
        prompt_tokens = tokenizer.encode(prompt_text)
        print(f"Prompt: '{prompt_text}'")
        print(f"Prompt tokens: {prompt_tokens}")
        
        # Get logits for the first token position
        # We need to generate with logprobs to see what tokens were likely
        params = SamplingParams(
            temperature=1.0,
            top_k=50,
            top_p=1.0,
            max_tokens=1,
            logprobs=50,  # Get top 50 log probabilities
        )
        
        output = llm.generate(prompt_text, params)[0]
        
        if output.outputs[0].logprobs:
            print(f"\nTop 10 most likely tokens at position 0:")
            first_logprobs = output.outputs[0].logprobs[0]
            
            # Sort by logprob
            sorted_tokens = sorted(first_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)
            
            for rank, (token_id, logprob_data) in enumerate(sorted_tokens[:10]):
                token_text = tokenizer.decode([token_id])
                prob = torch.exp(torch.tensor(logprob_data.logprob)).item()
                print(f"  {rank+1}. Token {token_id} ('{token_text}'): prob={prob:.6f}, logprob={logprob_data.logprob:.4f}")
            
            # Now check what token forced seed would select
            print(f"\n" + "-"*80)
            print("Checking forced seed selections for rollouts 0-3:")
            print("-"*80)
            
            # Get the logits (we need raw logits, not just logprobs)
            # We'll simulate with the probabilities we have
            for rollout_idx in range(4):
                # Compute what u_at would give us
                u = u_at(randomness, prompt_idx, checkpoint_hash, rollout_idx, 0)
                
                # For now, just generate with forced seed and see what we get
                params_forced = forced_seed_vllm_sampling_params(
                    randomness=randomness,
                    prompt_idx=prompt_idx,
                    checkpoint_hash=checkpoint_hash,
                    rollout_index=rollout_idx,
                    max_tokens=1,
                )
                
                output_forced = llm.generate(prompt_text, params_forced)[0]
                token_text = output_forced.outputs[0].text
                
                # Try to find this token in our top tokens
                token_ids = tokenizer.encode(token_text)
                if token_ids:
                    forced_token_id = token_ids[0]
                    
                    # Check if this token is in our logprobs
                    if forced_token_id in first_logprobs:
                        forced_prob = torch.exp(torch.tensor(first_logprobs[forced_token_id].logprob)).item()
                        
                        # Find rank
                        rank = next((i for i, (tid, _) in enumerate(sorted_tokens) if tid == forced_token_id), -1) + 1
                        
                        print(f"Rollout {rollout_idx}: u={u:.6f}, token '{token_text}', prob={forced_prob:.6f}, rank={rank}")
                    else:
                        print(f"Rollout {rollout_idx}: u={u:.6f}, token '{token_text}', prob=<very low, not in top 50>")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all diagnostic tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE BATCH INVARIANCE DIAGNOSTIC")
    print("Testing vLLM forced seed batch vs sequential mismatch")
    print("="*80)
    
    tests = [
        ("Strict Batch Invariance", test_batch_invariance_strict),
        ("Token-by-Token Diagnostic", test_token_by_token_comparison),
        ("Forced Token Probabilities", test_forced_token_probabilities),
    ]
    
    for name, test_func in tests:
        try:
            print(f"\n\n{'='*80}")
            print(f"RUNNING: {name}")
            print(f"{'='*80}")
            success = test_func()
            if not success and name == "Strict Batch Invariance":
                print("\n⚠️  CRITICAL: Batch invariance still failing with all features disabled!")
                print("This indicates a deeper issue with vLLM or the processor implementation.")
                break
        except Exception as e:
            print(f"\n❌ Test '{name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
