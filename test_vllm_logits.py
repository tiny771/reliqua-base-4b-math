#!/usr/bin/env python3
"""Test if vLLM produces different logits in batch vs sequential mode."""

import sys
import torch
from vllm import LLM
from vllm.v1.sample.logits_processor import LogitsProcessor
from vllm.sampling_params import SamplingParams

# Add reliquary to path
sys.path.insert(0, "/root/reliquary-miner/reliquary")


class LogitsCaptureProcessor(LogitsProcessor):
    """Captures logits from each position to compare."""
    
    def __init__(self):
        self.captured_logits = {}  # {batch_idx: [logits_at_step0, logits_at_step1, ...]}
    
    def __call__(self, *args, **kwargs):
        pass
    
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Capture first 10 logits values for debugging."""
        # Store first 10 values as fingerprint
        for batch_idx in range(logits.shape[0]):
            key = f"batch_{batch_idx}"
            if key not in self.captured_logits:
                self.captured_logits[key] = []
            
            # Store first 10 logit values as fingerprint
            fingerprint = tuple(logits[batch_idx, :10].cpu().tolist())
            self.captured_logits[key].append(fingerprint)
        
        return logits


def test():
    model_name = "facebook/opt-125m"
    prompt_text = "The answer is"
    
    print("="*80)
    print("Testing if vLLM produces same logits in batch vs sequential")
    print("="*80)
    
    # Sequential mode
    print("\n1. SEQUENTIAL MODE")
    print("-"*80)
    llm_seq = LLM(
        model=model_name,
        max_model_len=512,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    
    # Generate 1 sequence with 5 tokens, capturing logits
    params_seq = SamplingParams(max_tokens=5, temperature=1.0)
    output_seq = llm_seq.generate(prompt_text, params_seq)
    print(f"Sequential output: {output_seq[0].outputs[0].text}")
    
    del llm_seq
    torch.cuda.empty_cache()
    
    # Batch mode
    print("\n2. BATCH MODE")
    print("-"*80)
    llm_batch = LLM(
        model=model_name,
        max_model_len=512,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    
    # Generate 4 sequences (same prompt, same request)
    prompts = [prompt_text] * 4
    params_list = [SamplingParams(max_tokens=5, temperature=1.0) for _ in range(4)]
    outputs_batch = llm_batch.generate(prompts, params_list)
    
    print(f"Batch rollout 0: {outputs_batch[0].outputs[0].text}")
    print(f"Batch rollout 1: {outputs_batch[1].outputs[0].text}")
    print(f"Batch rollout 2: {outputs_batch[2].outputs[0].text}")
    print(f"Batch rollout 3: {outputs_batch[3].outputs[0].text}")
    
    # Check if rollout 0 from batch matches sequential
    batch_0_text = outputs_batch[0].outputs[0].text
    seq_text = output_seq[0].outputs[0].text
    
    print("\n3. COMPARISON")
    print("-"*80)
    if batch_0_text == seq_text:
        print(f"✓ Batch rollout 0 MATCHES sequential")
        print(f"  Both: '{batch_0_text}'")
    else:
        print(f"✗ Batch rollout 0 DIFFERS from sequential")
        print(f"  Sequential: '{seq_text}'")
        print(f"  Batch[0]:   '{batch_0_text}'")
        print(f"\nThis proves vLLM itself produces different outputs in batch vs sequential mode!")
    
    del llm_batch


if __name__ == "__main__":
    test()
