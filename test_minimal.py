"""Minimal reproduction test."""
import os
os.environ["DEBUG_FORCED_SEED"] = "1"

from vllm import LLM
from reliquary.miner.vllm_forced_seed_sampler import forced_seed_vllm_sampling_params


model_name = "facebook/opt-125m"
llm = LLM(
    model=model_name,
    logits_processors=["reliquary.miner.vllm_forced_seed_sampler:ForcedSeedVllmLogitsProcessor"],
    max_model_len=512,
    gpu_memory_utilization=0.3,
    enforce_eager=True,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
)

randomness = "test-window-randomness-12345"
prompt_idx = 42
checkpoint_hash = "test-checkpoint-abc123"
prompt_text = "The answer is"

print("\n" + "="*80)
print("SEQUENTIAL - Rollout 1")
print("="*80)
params1 = forced_seed_vllm_sampling_params(
    randomness=randomness,
    prompt_idx=prompt_idx,
    checkpoint_hash=checkpoint_hash,
    rollout_index=1,
    max_tokens=20,  # Increased to reproduce mismatch
)
output1 = llm.generate(prompt_text, params1)
text1 = output1[0].outputs[0].text
print(f"Result: '{text1}'")

print("\n" + "="*80)
print("BATCH - Rollouts [0,1,2,3]")
print("="*80)
prompts = [prompt_text] * 4
params_list = [
    forced_seed_vllm_sampling_params(
        randomness=randomness,
        prompt_idx=prompt_idx,
        checkpoint_hash=checkpoint_hash,
        rollout_index=i,
        max_tokens=20,  # Increased to reproduce mismatch
    )
    for i in range(4)
]
outputs = llm.generate(prompts, params_list)
for i, output in enumerate(outputs):
    text = output.outputs[0].text
    print(f"Rollout {i}: '{text}'")
    if i == 1:
        if text == text1:
            print(f"  ✓ Matches sequential")
        else:
            print(f"  ✗ MISMATCH with sequential!")
            print(f"    Sequential: '{text1}'")
            print(f"    Batch:      '{text}'")



print("\n" + "="*80)
print("BATCH - Rollouts [0,1,2,3]")
print("="*80)
prompts = [prompt_text] * 4
params_list = [
    forced_seed_vllm_sampling_params(
        randomness=randomness,
        prompt_idx=prompt_idx,
        checkpoint_hash=checkpoint_hash,
        rollout_index=i,
        max_tokens=20,  # Increased to reproduce mismatch
    )
    for i in range(4)
]
# outputs = llm.generate(prompts, params_list)
outputs = []
for i in range(4):
  output = llm.generate(prompts[i], params_list[i])
  outputs.append(output)

for i, output in enumerate(outputs):
    text = output[0].outputs[0].text
    print(f"Rollout {i}: '{text}'")
    if i == 1:
        if text == text1:
            print(f"  ✓ Matches sequential")
        else:
            print(f"  ✗ MISMATCH with sequential!")
            print(f"    Sequential: '{text1}'")
            print(f"    Batch:      '{text}'")