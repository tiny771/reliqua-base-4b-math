"""Test that mimics engine's concurrent generation pattern.

The engine uses asyncio.create_task() to launch multiple HTTP requests
concurrently. These arrive at vLLM server and may be batched together.

This is DIFFERENT from calling llm.generate() in a loop (which is always batch_size=1).
"""
import asyncio
import httpx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "reliquary"))

from reliquary.miner.vllm_forced_seed_sampler import forced_seed_vllm_sampling_params
from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO


# Test configuration
VLLM_URL = "http://localhost:8000"
MODEL_NAME = "reliquary"
PROMPT = "The answer is"
RANDOMNESS = "test-window-randomness-12345"
PROMPT_IDX = 42
CHECKPOINT_HASH = "test-checkpoint-abc123"
MAX_TOKENS = 20
M_ROLLOUTS = 8


async def generate_single_rollout_http(
    client: httpx.AsyncClient,
    rollout_idx: int,
) -> dict:
    """Generate a single rollout via HTTP request - mimics engine's _generate_single_rollout."""
    
    # Build sampling params
    params = forced_seed_vllm_sampling_params(
        randomness=RANDOMNESS,
        prompt_idx=PROMPT_IDX,
        checkpoint_hash=CHECKPOINT_HASH,
        rollout_index=rollout_idx,
        max_tokens=MAX_TOKENS,
    )
    
    # Convert to HTTP payload
    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "temperature": params.temperature,
        "max_tokens": params.max_tokens,
        "top_k": params.top_k,
        "top_p": params.top_p,
        "logprobs": 1,
        "skip_special_tokens": params.skip_special_tokens,
        "ignore_eos": params.ignore_eos,
        "return_token_ids": True,
    }
    
    # Add forced seed config
    if params.extra_args:
        payload["vllm_xargs"] = params.extra_args
    
    # Make HTTP request
    response = await client.post(
        f"{VLLM_URL}/v1/completions",
        json=payload,
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()
    
    # Extract result
    choice = data["choices"][0]
    return {
        "rollout_idx": rollout_idx,
        "text": choice["text"],
        "tokens": choice.get("token_ids", []),
    }


async def generate_rollouts_concurrent(n_rollouts: int = 8) -> list:
    """Generate rollouts concurrently - EXACTLY like engine does."""
    
    async with httpx.AsyncClient() as client:
        # Launch all rollouts concurrently (like engine does with asyncio.create_task)
        tasks = []
        for rollout_idx in range(n_rollouts):
            task = asyncio.create_task(
                generate_single_rollout_http(client, rollout_idx)
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
        
        # Sort by rollout_idx
        valid_results.sort(key=lambda r: r["rollout_idx"])
        return valid_results


async def test_engine_pattern():
    """Test the actual engine pattern: concurrent HTTP requests."""
    
    print("\n" + "="*80)
    print("ENGINE CONCURRENT PATTERN TEST")
    print("="*80)
    print(f"vLLM URL: {VLLM_URL}")
    print(f"Prompt: '{PROMPT}'")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Rollouts: {M_ROLLOUTS}")
    print("="*80)
    
    # Run 1: Generate rollouts concurrently
    print("\n🔄 Run 1: Generating 8 rollouts concurrently (via HTTP)...")
    results1 = await generate_rollouts_concurrent(M_ROLLOUTS)
    print(f"✓ Run 1 completed: {len(results1)}/{M_ROLLOUTS} rollouts")
    
    for r in results1:
        print(f"  Rollout {r['rollout_idx']}: {len(r['tokens'])} tokens - '{r['text'][:50]}...'")
    
    # Run 2: Generate same rollouts again
    print("\n🔄 Run 2: Generating same 8 rollouts again...")
    results2 = await generate_rollouts_concurrent(M_ROLLOUTS)
    print(f"✓ Run 2 completed: {len(results2)}/{M_ROLLOUTS} rollouts")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON: Run 1 vs Run 2")
    print("="*80)
    
    matches = 0
    mismatches = 0
    
    for i in range(min(len(results1), len(results2))):
        r1 = results1[i]
        r2 = results2[i]
        
        if r1["tokens"] == r2["tokens"]:
            matches += 1
            print(f"  Rollout {i}: ✓ MATCH")
        else:
            mismatches += 1
            print(f"  Rollout {i}: ✗ MISMATCH")
            
            # Find divergence
            divergence_idx = None
            for j in range(min(len(r1["tokens"]), len(r2["tokens"]))):
                if r1["tokens"][j] != r2["tokens"][j]:
                    divergence_idx = j
                    break
            
            print(f"    Divergence at token: {divergence_idx}")
            print(f"    Run 1: '{r1['text'][:60]}...'")
            print(f"    Run 2: '{r2['text'][:60]}...'")
    
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    print(f"Total rollouts: {M_ROLLOUTS}")
    print(f"✓ Matches: {matches}")
    print(f"✗ Mismatches: {mismatches}")
    
    if mismatches > 0:
        print("\n❌ FAILED: Engine pattern has seed_mismatch!")
        print("\nThis is the bug causing production failures!")
        print("\nSOLUTION: Update vLLM server configuration:")
        print("  - enable_prefix_caching=False")
        print("  - enable_chunked_prefill=False")
        print("  - enforce_eager=True")
        return 1
    else:
        print("\n✅ PASSED: Engine pattern is deterministic!")
        return 0


if __name__ == "__main__":
    """
    USAGE:
    ------
    1. Start vLLM server with batch-invariant settings:
       
       python -m vllm.entrypoints.openai.api_server \\
         --model facebook/opt-125m \\
         --port 8000 \\
         --logits-processor reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor \\
         --enable-prefix-caching=False \\
         --enable-chunked-prefill=False \\
         --enforce-eager
    
    2. Run this test:
       python test_engine_concurrent_pattern.py
    
    This will show if concurrent requests (like engine uses) are deterministic.
    """
    
    try:
        exit_code = asyncio.run(test_engine_pattern())
        exit(exit_code)
    except httpx.ConnectError:
        print("\n❌ ERROR: Cannot connect to vLLM server")
        print(f"\nMake sure vLLM is running at {VLLM_URL}")
        print("\nStart server with:")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print("    --model facebook/opt-125m \\")
        print("    --port 8000 \\")
        print("    --logits-processor reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor \\")
        print("    --enable-prefix-caching=False \\")
        print("    --enable-chunked-prefill=False \\")
        print("    --enforce-eager")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
