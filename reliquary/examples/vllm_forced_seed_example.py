"""Example usage of vLLM forced seed sampler for Reliquary mining.

This demonstrates how to initialize vLLM with the forced seed processor and
generate completions that are validator-verifiable.
"""
from vllm import LLM, SamplingParams

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.miner.vllm_forced_seed_sampler import (
    ForcedSeedVllmLogitsProcessor,
    forced_seed_vllm_sampling_params,
)


def example_basic_usage():
    """Basic example: initialize vLLM and generate with forced seed."""
    
    # Initialize vLLM with the forced seed processor
    # Method 1: Pass fully-qualified class name
    llm = LLM(
        model="Qwen/Qwen3.5-2B",
        logits_processors=[
            "reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"
        ],
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )
    
    # Method 2: Pass class object (offline only)
    # llm = LLM(
    #     model="Qwen/Qwen3.5-2B",
    #     logits_processors=[ForcedSeedVllmLogitsProcessor],
    #     max_model_len=2048,
    # )
    
    # Example configuration from validator /state
    window_randomness = "drand-quicknet-round-12345678"
    prompt_idx = 42
    checkpoint_hash = "abc123def456"  # HF revision from /state.checkpoint_revision
    
    prompt = "What is the square root of 144?"
    
    # Generate 8 rollouts with forced seed (protocol M_ROLLOUTS=8)
    outputs = []
    for rollout_idx in range(8):
        # Use helper to create sampling params
        sampling_params = forced_seed_vllm_sampling_params(
            randomness=window_randomness,
            prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash,
            rollout_index=rollout_idx,
            base_offset=0,  # Phase-1 generation starts at offset 0
            temperature=T_PROTO,
            top_k=TOP_K_PROTO,
            top_p=TOP_P_PROTO,
            max_tokens=512,
        )
        
        output = llm.generate(prompt, sampling_params)
        outputs.append(output)
        
        print(f"Rollout {rollout_idx}:")
        print(f"  Generated: {output[0].outputs[0].text[:100]}...")
        print()
    
    return outputs


def example_batch_generation():
    """Example: batch generation with multiple prompts."""
    
    llm = LLM(
        model="Qwen/Qwen3.5-2B",
        logits_processors=[
            "reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"
        ],
        max_model_len=2048,
    )
    
    window_randomness = "drand-quicknet-round-12345678"
    checkpoint_hash = "abc123def456"
    
    # Generate multiple prompts in batch (each with its own configuration)
    prompts = [
        "What is 2 + 2?",
        "Solve x^2 = 16",
        "What is the capital of France?",
    ]
    
    # Create sampling params for each prompt
    sampling_params_list = []
    for idx, prompt in enumerate(prompts):
        params = forced_seed_vllm_sampling_params(
            randomness=window_randomness,
            prompt_idx=idx,
            checkpoint_hash=checkpoint_hash,
            rollout_index=0,
            max_tokens=256,
        )
        sampling_params_list.append(params)
    
    # Batch generate
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params_list,
    )
    
    for idx, output in enumerate(outputs):
        print(f"Prompt {idx}: {prompts[idx]}")
        print(f"Output: {output.outputs[0].text}")
        print()
    
    return outputs


def example_manual_sampling_params():
    """Example: manually construct SamplingParams with forced seed config."""
    
    llm = LLM(
        model="Qwen/Qwen3.5-2B",
        logits_processors=[
            "reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"
        ],
    )
    
    # Manually construct sampling params (equivalent to using helper)
    sampling_params = SamplingParams(
        temperature=1.0,  # Required by vLLM but will be ignored by processor
        max_tokens=512,
        extra_args={
            "forced_seed_config": {
                "randomness": "drand-quicknet-round-12345678",
                "prompt_idx": 42,
                "checkpoint_hash": "abc123def456",
                "rollout_index": 0,
                "base_offset": 0,
                "temperature": T_PROTO,
                "top_k": TOP_K_PROTO,
                "top_p": TOP_P_PROTO,
            }
        },
    )
    
    output = llm.generate("What is 5 * 7?", sampling_params)
    print(f"Output: {output[0].outputs[0].text}")
    
    return output


def example_without_forced_seed():
    """Example: mix forced and non-forced requests in same batch."""
    
    llm = LLM(
        model="Qwen/Qwen3.5-2B",
        logits_processors=[
            "reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"
        ],
    )
    
    prompts = ["Forced seed prompt", "Regular sampling prompt"]
    
    # First prompt uses forced seed
    forced_params = forced_seed_vllm_sampling_params(
        randomness="drand-quicknet-round-12345678",
        prompt_idx=0,
        checkpoint_hash="abc123",
        rollout_index=0,
        max_tokens=100,
    )
    
    # Second prompt uses regular sampling (no forced_seed_config)
    regular_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
    )
    
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=[forced_params, regular_params],
    )
    
    print("Forced seed output:", outputs[0].outputs[0].text)
    print("Regular sampling output:", outputs[1].outputs[0].text)
    
    return outputs


def example_phase2_bft():
    """Example: Phase-2 BFT generation with non-zero base_offset."""
    
    llm = LLM(
        model="Qwen/Qwen3.5-2B",
        logits_processors=[
            "reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"
        ],
    )
    
    # In BFT phase-2, we continue from primed sequences
    # base_offset = primed_length - prompt_length (clamped at 0)
    primed_length = 150
    prompt_length = 50
    base_offset = max(0, primed_length - prompt_length)
    
    sampling_params = forced_seed_vllm_sampling_params(
        randomness="drand-quicknet-round-12345678",
        prompt_idx=42,
        checkpoint_hash="abc123",
        rollout_index=0,
        base_offset=base_offset,  # Continue from primed position
        max_tokens=256,
    )
    
    output = llm.generate("Prompt text...", sampling_params)
    print(f"Phase-2 output: {output[0].outputs[0].text}")
    
    return output


if __name__ == "__main__":
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)
    example_basic_usage()
    
    print("\n" + "=" * 80)
    print("Example 2: Batch Generation")
    print("=" * 80)
    example_batch_generation()
    
    print("\n" + "=" * 80)
    print("Example 3: Manual SamplingParams")
    print("=" * 80)
    example_manual_sampling_params()
    
    print("\n" + "=" * 80)
    print("Example 4: Mixed Forced/Regular Sampling")
    print("=" * 80)
    example_without_forced_seed()
    
    print("\n" + "=" * 80)
    print("Example 5: Phase-2 BFT with base_offset")
    print("=" * 80)
    example_phase2_bft()
