"""Simple test with debug processor to see what's happening."""
from vllm import LLM
from reliquary.miner.vllm_forced_seed_sampler import forced_seed_vllm_sampling_params


def test_debug():
    """Test with debug logging."""
    print("="*80)
    print("DEBUG TEST - BATCH VS SEQUENTIAL")
    print("="*80)
    
    model_name = "facebook/opt-125m"
    
    print("\nInitializing with DEBUG processor...")
    llm = LLM(
        model=model_name,
        logits_processors=["debug_processor:DebugForcedSeedVllmLogitsProcessor"],
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
    max_tokens = 5
    
    print("\n" + "-"*80)
    print("SEQUENTIAL - Rollout 1 only")
    print("-"*80)
    params = forced_seed_vllm_sampling_params(
        randomness=randomness,
        prompt_idx=prompt_idx,
        checkpoint_hash=checkpoint_hash,
        rollout_index=1,
        max_tokens=max_tokens,
    )
    seq_output = llm.generate(prompt_text, params)
    seq_text = seq_output[0].outputs[0].text
    print(f"Result: '{seq_text}'")
    
    print("\n" + "-"*80)
    print("BATCH - Rollouts 0,1,2,3")
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
    for i, output in enumerate(batch_outputs):
        batch_text = output.outputs[0].text
        print(f"Rollout {i}: '{batch_text}'")
        if i == 1:
            match = batch_text == seq_text
            print(f"  Matches sequential: {match}")
            if not match:
                print(f"  Sequential: '{seq_text}'")


if __name__ == "__main__":
    test_debug()
