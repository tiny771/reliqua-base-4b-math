"""Simplified test to debug the batch vs sequential discrepancy."""
import torch
from vllm import LLM

from reliquary.miner.vllm_forced_seed_sampler import forced_seed_vllm_sampling_params


def test_first_token_only():
    """Test that first token is identical in batch vs sequential."""
    print("\n" + "="*80)
    print("SIMPLIFIED TEST: First Token Only")
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
        )
    except Exception as e:
        print(f"Failed to init vLLM: {e}")
        return False
    
    randomness = "test-window-randomness-12345"
    prompt_idx = 42
    checkpoint_hash = "test-checkpoint-abc123"
    prompt_text = "The answer is"
    max_tokens = 1  # Generate just ONE token
    
    print(f"\nPrompt: '{prompt_text}'")
    print(f"Max tokens: {max_tokens}")
    print("="*80)
    
    # Sequential generation
    print("\n1. Sequential Generation (4 separate requests):")
    sequential_tokens = []
    for rollout_idx in range(4):
        params = forced_seed_vllm_sampling_params(
            randomness=randomness,
            prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash,
            rollout_index=rollout_idx,
            max_tokens=max_tokens,
        )
        output = llm.generate(prompt_text, params)
        token_text = output[0].outputs[0].text
        sequential_tokens.append(token_text)
        print(f"  Rollout {rollout_idx}: '{token_text}'")
    
    # Batch generation
    print("\n2. Batch Generation (4 requests in one batch):")
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
    batch_tokens = [output.outputs[0].text for output in batch_outputs]
    for i, token_text in enumerate(batch_tokens):
        print(f"  Rollout {i}: '{token_text}'")
    
    # Compare
    print("\n3. Comparison:")
    all_match = True
    for i in range(4):
        match = sequential_tokens[i] == batch_tokens[i]
        print(f"  Rollout {i}: {'✓ MATCH' if match else '✗ MISMATCH'}")
        if not match:
            print(f"    Sequential: '{sequential_tokens[i]}'")
            print(f"    Batch:      '{batch_tokens[i]}'")
            all_match = False
    
    print("\n" + "="*80)
    if all_match:
        print("✅ SUCCESS: All first tokens match!")
        print("="*80)
        return True
    else:
        print("❌ FAILURE: First tokens don't match between batch and sequential!")
        print("="*80)
        return False


if __name__ == "__main__":
    success = test_first_token_only()
    exit(0 if success else 1)
