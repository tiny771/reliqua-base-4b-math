"""Quick seed mismatch test - tests single rollout determinism."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "reliquary"))

from reliquary.miner.vllm_forced_seed_sampler import forced_seed_vllm_sampling_params
from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO


async def test_single_rollout_determinism(enable_prefix_caching=True):
    """Test if same rollout generates same output twice."""
    from vllm import LLM
    
    print(f"\n{'='*80}")
    print(f"Testing with prefix_caching={enable_prefix_caching}")
    print(f"{'='*80}\n")
    
    print("Initializing vLLM...")
    llm = LLM(
        model="facebook/opt-125m",
        logits_processors=[
            "reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"
        ],
        max_model_len=512,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=enable_prefix_caching,
    )
    print("✓ vLLM initialized\n")
    
    # Test parameters
    prompt = "What is 2+2? The answer is"
    randomness = "test-randomness"
    prompt_idx = 42
    checkpoint_hash = "test-hash"
    rollout_idx = 0
    max_tokens = 30
    
    print(f"Prompt: '{prompt}'")
    print(f"Rollout: {rollout_idx}, Max tokens: {max_tokens}\n")
    
    # Generate same rollout 3 times
    results = []
    for i in range(3):
        print(f"🔄 Generation {i+1}/3...")
        
        params = forced_seed_vllm_sampling_params(
            randomness=randomness,
            prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash,
            rollout_index=rollout_idx,
            max_tokens=max_tokens,
        )
        
        def _gen():
            return llm.generate(prompt, params)[0]
        
        output = await asyncio.to_thread(_gen)
        tokens = output.outputs[0].token_ids
        text = output.outputs[0].text
        
        results.append({
            "tokens": tokens,
            "text": text,
        })
        
        print(f"  Generated {len(tokens)} tokens")
        print(f"  Text: '{text[:60]}...'\n")
    
    # Compare all three
    print(f"{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}\n")
    
    ref_tokens = results[0]["tokens"]
    all_match = True
    
    for i in range(1, 3):
        if results[i]["tokens"] == ref_tokens:
            print(f"Generation {i+1}: ✓ MATCH")
        else:
            all_match = False
            print(f"Generation {i+1}: ✗ MISMATCH")
            
            # Find divergence
            for j in range(min(len(ref_tokens), len(results[i]["tokens"]))):
                if ref_tokens[j] != results[i]["tokens"][j]:
                    print(f"  Divergence at token {j}")
                    print(f"  Gen 1: token_id={ref_tokens[j]}")
                    print(f"  Gen {i+1}: token_id={results[i]['tokens'][j]}")
                    break
    
    print(f"\n{'='*80}")
    if all_match:
        print(f"✅ PASSED: All generations identical!")
        print(f"   prefix_caching={enable_prefix_caching}")
        print(f"{'='*80}\n")
        return True
    else:
        print(f"❌ FAILED: Generations NOT identical!")
        print(f"   prefix_caching={enable_prefix_caching}")
        print(f"   THIS IS THE SEED MISMATCH BUG!")
        print(f"{'='*80}\n")
        return False


async def main():
    print("\n" + "="*80)
    print("QUICK SEED MISMATCH TEST")
    print("="*80)
    
    # Test with prefix caching enabled
    try:
        result_with = await test_single_rollout_determinism(enable_prefix_caching=True)
    except Exception as e:
        print(f"❌ Test with caching failed: {e}")
        import traceback
        traceback.print_exc()
        result_with = False
    
    # Test with prefix caching disabled
    try:
        result_without = await test_single_rollout_determinism(enable_prefix_caching=False)
    except Exception as e:
        print(f"❌ Test without caching failed: {e}")
        import traceback
        traceback.print_exc()
        result_without = False
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"With prefix caching:    {'✅ PASS' if result_with else '❌ FAIL'}")
    print(f"Without prefix caching: {'✅ PASS' if result_without else '❌ FAIL'}")
    print("="*80)
    
    if not result_with and result_without:
        print("\n🚨 ROOT CAUSE: Prefix caching breaks determinism!")
        print("\nSOLUTION: Disable prefix caching in vLLM server")
        return 1
    elif not result_with and not result_without:
        print("\n🚨 ROOT CAUSE: Not just prefix caching!")
        print("\nNeed to investigate other vLLM options")
        return 1
    else:
        print("\n✅ No seed mismatch detected")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
