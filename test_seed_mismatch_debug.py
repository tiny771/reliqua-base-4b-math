"""Critical test to debug seed mismatch failures in production.

This test mimics the exact engine behavior:
1. Concurrent generation (not batching)
2. Each rollout is a separate vLLM call
3. Tests determinism across multiple runs
4. Tests with/without prefix caching and other options
"""
import asyncio
import torch
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np

# Add reliquary to path
sys.path.insert(0, str(Path(__file__).parent / "reliquary"))

from reliquary.environment.forced_sampling import u_at, warp, pick
from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO


class TestConfig:
    """Test configuration matching production."""
    model_name = "facebook/opt-125m"
    randomness = "test-window-randomness-12345"
    prompt_idx = 42
    checkpoint_hash = "test-checkpoint-abc123"
    prompt_text = "What is 2+2? The answer is"
    max_tokens = 50
    M_ROLLOUTS = 8


def initialize_vllm(enable_prefix_caching=True):
    """Initialize vLLM with specified configuration."""
    from vllm import LLM
    
    print(f"\n{'='*80}")
    print(f"Initializing vLLM with prefix_caching={enable_prefix_caching}")
    print(f"{'='*80}")
    
    llm = LLM(
        model=TestConfig.model_name,
        logits_processors=[
            "reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"
        ],
        max_model_len=512,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=enable_prefix_caching,  # KEY OPTION
        # disable_sliding_window=False,  # Default
        # trust_remote_code=False,
    )
    return llm


async def generate_single_rollout_async(
    llm,
    prompt: str,
    rollout_idx: int,
    randomness: str,
    prompt_idx: int,
    checkpoint_hash: str,
    max_tokens: int,
) -> dict:
    """Generate a single rollout - mimics engine's _generate_single_rollout."""
    from reliquary.miner.vllm_forced_seed_sampler import forced_seed_vllm_sampling_params
    
    # Create sampling params with forced seed config
    params = forced_seed_vllm_sampling_params(
        randomness=randomness,
        prompt_idx=prompt_idx,
        checkpoint_hash=checkpoint_hash,
        rollout_index=rollout_idx,
        base_offset=0,
        temperature=T_PROTO,
        top_k=TOP_K_PROTO,
        top_p=TOP_P_PROTO,
        max_tokens=max_tokens,
    )
    
    # Run generation in thread pool (vLLM generate is blocking)
    def _generate():
        outputs = llm.generate(prompt, params)
        return outputs[0]
    
    output = await asyncio.to_thread(_generate)
    
    # Extract result
    result = {
        "rollout_idx": rollout_idx,
        "text": output.outputs[0].text,
        "tokens": output.outputs[0].token_ids,
        "finish_reason": output.outputs[0].finish_reason,
    }
    
    return result


async def generate_rollouts_concurrent(
    llm,
    prompt: str,
    randomness: str,
    prompt_idx: int,
    checkpoint_hash: str,
    max_tokens: int,
    n_rollouts: int = 8,
) -> List[dict]:
    """Generate multiple rollouts concurrently - mimics engine's _generate_rollouts."""
    
    # Launch all rollouts concurrently (like engine does)
    tasks = []
    for rollout_idx in range(n_rollouts):
        task = asyncio.create_task(
            generate_single_rollout_async(
                llm,
                prompt,
                rollout_idx,
                randomness,
                prompt_idx,
                checkpoint_hash,
                max_tokens,
            )
        )
        tasks.append(task)
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ Rollout {i} failed: {result}")
        else:
            valid_results.append(result)
    
    # Sort by rollout_idx (like engine does)
    valid_results.sort(key=lambda r: r["rollout_idx"])
    
    return valid_results


def compare_rollouts(results1: List[dict], results2: List[dict]) -> dict:
    """Compare two sets of rollouts for determinism."""
    comparison = {
        "total_rollouts": len(results1),
        "matches": 0,
        "mismatches": 0,
        "mismatch_details": [],
    }
    
    for i in range(min(len(results1), len(results2))):
        r1 = results1[i]
        r2 = results2[i]
        
        rollout_idx = r1["rollout_idx"]
        tokens1 = r1["tokens"]
        tokens2 = r2["tokens"]
        text1 = r1["text"]
        text2 = r2["text"]
        
        if tokens1 == tokens2:
            comparison["matches"] += 1
            print(f"  Rollout {rollout_idx}: ✓ MATCH ({len(tokens1)} tokens)")
        else:
            comparison["mismatches"] += 1
            
            # Find first divergence point
            divergence_idx = None
            for j in range(min(len(tokens1), len(tokens2))):
                if tokens1[j] != tokens2[j]:
                    divergence_idx = j
                    break
            
            detail = {
                "rollout_idx": rollout_idx,
                "divergence_token": divergence_idx,
                "run1_len": len(tokens1),
                "run2_len": len(tokens2),
                "run1_text": text1[:100],
                "run2_text": text2[:100],
            }
            comparison["mismatch_details"].append(detail)
            
            print(f"  Rollout {rollout_idx}: ✗ MISMATCH")
            print(f"    Divergence at token: {divergence_idx}")
            print(f"    Run 1: {len(tokens1)} tokens - '{text1[:80]}...'")
            print(f"    Run 2: {len(tokens2)} tokens - '{text2[:80]}...'")
    
    return comparison


async def test_determinism_with_config(enable_prefix_caching=True) -> bool:
    """Test determinism with specific vLLM configuration."""
    print(f"\n{'='*80}")
    print(f"TEST: Determinism with prefix_caching={enable_prefix_caching}")
    print(f"{'='*80}")
    
    llm = initialize_vllm(enable_prefix_caching=enable_prefix_caching)
    
    # Generate rollouts twice with SAME parameters
    print("\n🔄 Run 1: Generating 8 rollouts concurrently...")
    results1 = await generate_rollouts_concurrent(
        llm,
        TestConfig.prompt_text,
        TestConfig.randomness,
        TestConfig.prompt_idx,
        TestConfig.checkpoint_hash,
        TestConfig.max_tokens,
        TestConfig.M_ROLLOUTS,
    )
    
    print(f"✓ Run 1 completed: {len(results1)}/{TestConfig.M_ROLLOUTS} rollouts")
    
    print("\n🔄 Run 2: Generating same 8 rollouts again...")
    results2 = await generate_rollouts_concurrent(
        llm,
        TestConfig.prompt_text,
        TestConfig.randomness,
        TestConfig.prompt_idx,
        TestConfig.checkpoint_hash,
        TestConfig.max_tokens,
        TestConfig.M_ROLLOUTS,
    )
    
    print(f"✓ Run 2 completed: {len(results2)}/{TestConfig.M_ROLLOUTS} rollouts")
    
    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON: Run 1 vs Run 2")
    print(f"{'='*80}")
    
    comparison = compare_rollouts(results1, results2)
    
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"{'='*80}")
    print(f"Total rollouts: {comparison['total_rollouts']}")
    print(f"✓ Matches: {comparison['matches']}")
    print(f"✗ Mismatches: {comparison['mismatches']}")
    
    if comparison["mismatches"] > 0:
        print(f"\n❌ FAILED: {comparison['mismatches']} rollout(s) not deterministic!")
        print(f"\nThis is the CRITICAL BUG causing seed_mismatch failures!")
        return False
    else:
        print(f"\n✅ PASSED: All rollouts are deterministic!")
        return True


async def test_single_rollout_repeated(enable_prefix_caching=True, n_repeats=3) -> bool:
    """Test single rollout determinism by generating it multiple times."""
    print(f"\n{'='*80}")
    print(f"TEST: Single rollout repeated {n_repeats} times")
    print(f"      prefix_caching={enable_prefix_caching}")
    print(f"{'='*80}")
    
    llm = initialize_vllm(enable_prefix_caching=enable_prefix_caching)
    
    # Generate same rollout multiple times
    rollout_idx = 0
    results = []
    
    for i in range(n_repeats):
        print(f"\n🔄 Repetition {i+1}/{n_repeats}...")
        result = await generate_single_rollout_async(
            llm,
            TestConfig.prompt_text,
            rollout_idx,
            TestConfig.randomness,
            TestConfig.prompt_idx,
            TestConfig.checkpoint_hash,
            TestConfig.max_tokens,
        )
        results.append(result)
        print(f"✓ Generated {len(result['tokens'])} tokens")
    
    # Compare all results
    print(f"\n{'='*80}")
    print("COMPARISON: All repetitions")
    print(f"{'='*80}")
    
    reference = results[0]
    all_match = True
    
    for i in range(1, n_repeats):
        if results[i]["tokens"] == reference["tokens"]:
            print(f"  Repetition {i+1}: ✓ MATCH")
        else:
            all_match = False
            print(f"  Repetition {i+1}: ✗ MISMATCH")
            
            # Find divergence
            divergence_idx = None
            for j in range(min(len(reference["tokens"]), len(results[i]["tokens"]))):
                if reference["tokens"][j] != results[i]["tokens"][j]:
                    divergence_idx = j
                    break
            print(f"    Divergence at token: {divergence_idx}")
    
    if all_match:
        print(f"\n✅ PASSED: All {n_repeats} repetitions are identical!")
        return True
    else:
        print(f"\n❌ FAILED: Repetitions are NOT identical!")
        print(f"\nThis confirms the seed_mismatch bug!")
        return False


async def test_low_probability_tokens(enable_prefix_caching=True) -> bool:
    """Test if mismatch happens when forcing low-probability tokens."""
    print(f"\n{'='*80}")
    print(f"TEST: Low-probability token forcing")
    print(f"      prefix_caching={enable_prefix_caching}")
    print(f"{'='*80}")
    
    llm = initialize_vllm(enable_prefix_caching=enable_prefix_caching)
    
    # Generate and check token probabilities
    print("\n🔍 Analyzing forced token probabilities...")
    
    rollout_idx = 0
    result = await generate_single_rollout_async(
        llm,
        TestConfig.prompt_text,
        rollout_idx,
        TestConfig.randomness,
        TestConfig.prompt_idx,
        TestConfig.checkpoint_hash,
        20,  # Just 20 tokens for analysis
    )
    
    # Note: We can't easily get the probability of forced tokens without modifying vLLM
    # But we can check if generation is deterministic
    
    print(f"\n🔄 Generating same rollout again...")
    result2 = await generate_single_rollout_async(
        llm,
        TestConfig.prompt_text,
        rollout_idx,
        TestConfig.randomness,
        TestConfig.prompt_idx,
        TestConfig.checkpoint_hash,
        20,
    )
    
    if result["tokens"] == result2["tokens"]:
        print(f"\n✅ PASSED: Low-probability forcing is deterministic!")
        return True
    else:
        print(f"\n❌ FAILED: Low-probability forcing NOT deterministic!")
        return False


async def main():
    """Run all critical tests."""
    print("\n" + "="*80)
    print("CRITICAL SEED MISMATCH DEBUG TEST SUITE")
    print("="*80)
    print(f"Model: {TestConfig.model_name}")
    print(f"Prompt: '{TestConfig.prompt_text}'")
    print(f"Max tokens: {TestConfig.max_tokens}")
    print(f"Rollouts: {TestConfig.M_ROLLOUTS}")
    print("="*80)
    
    results = {}
    
    # Test 1: Determinism with prefix caching ENABLED (production default)
    try:
        results["determinism_with_caching"] = await test_determinism_with_config(
            enable_prefix_caching=True
        )
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results["determinism_with_caching"] = False
    
    # Test 2: Determinism with prefix caching DISABLED
    try:
        results["determinism_without_caching"] = await test_determinism_with_config(
            enable_prefix_caching=False
        )
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results["determinism_without_caching"] = False
    
    # Test 3: Single rollout repeated (with caching)
    try:
        results["single_rollout_with_caching"] = await test_single_rollout_repeated(
            enable_prefix_caching=True,
            n_repeats=5
        )
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results["single_rollout_with_caching"] = False
    
    # Test 4: Single rollout repeated (without caching)
    try:
        results["single_rollout_without_caching"] = await test_single_rollout_repeated(
            enable_prefix_caching=False,
            n_repeats=5
        )
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results["single_rollout_without_caching"] = False
    
    # Test 5: Low-probability tokens (with caching)
    try:
        results["low_prob_with_caching"] = await test_low_probability_tokens(
            enable_prefix_caching=True
        )
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results["low_prob_with_caching"] = False
    
    # Test 6: Low-probability tokens (without caching)
    try:
        results["low_prob_without_caching"] = await test_low_probability_tokens(
            enable_prefix_caching=False
        )
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results["low_prob_without_caching"] = False
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for p in results.values() if p)
    failed_tests = total_tests - passed_tests
    
    print(f"Total: {total_tests} tests")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print("="*80)
    
    if failed_tests > 0:
        print("\n🚨 CRITICAL: Seed mismatch bug confirmed!")
        print("\nRecommendations:")
        
        if not results.get("determinism_with_caching") and results.get("determinism_without_caching"):
            print("  1. ❗ Disable prefix caching in production vLLM server")
            print("  2. Update vLLM launch config: enable_prefix_caching=False")
        
        if not results.get("determinism_without_caching"):
            print("  1. ❗ Issue is NOT just prefix caching")
            print("  2. Check vLLM version and other config options")
            print("  3. Review: https://docs.vllm.ai/en/latest/features/batch_invariance/")
        
        print("\n" + "="*80)
        return 1
    else:
        print("\n✅ SUCCESS: No seed mismatch detected!")
        print("\nAll rollouts are deterministic across multiple runs.")
        print("="*80)
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
