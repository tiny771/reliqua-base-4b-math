#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import torch
    import math
except Exception:
    torch = None
    import math

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# The repository contains a nested `reliquary/` package directory. Add it
# to sys.path so imports like `reliquary.shared` resolve correctly.
PKG_ROOT = REPO_ROOT / "reliquary"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from reliquary.environment.forced_sampling import u_at
from reliquary.miner_math.utils import VLLMGenerator
from reliquary.shared.modeling import load_text_generation_model, load_tokenizer
from reliquary.shared.forward import forward_single_layer


def compare(prompt: str, vllm_url: str, vllm_model: str, hf_model: str, device: str, max_tokens: int, logprobs: int, output_json: str | None = None):
    if torch is None:
        print('torch not available: falling back to vLLM-only generation')
        device = None
    else:
        device = torch.device(device)
    print('Using vLLM server at', vllm_url, 'model', vllm_model)
    vgen = VLLMGenerator(base_url=vllm_url, model_name=vllm_model)

    print('Requesting rollout from vLLM...')
    extra_body = {
        'randomness': 'debug-randomness',
        'prompt_idx': 0,
        'checkpoint_hash': 'debug-checkpoint',
        'rollout_index': 0,
        'base_offset': 0,
    }
    res = vgen.generate_rollout(
        prompt=prompt,
        temperature=0.6,
        max_tokens=max_tokens,
        top_k=20,
        top_p=0.95,
        logprobs=logprobs,
        extra_body=extra_body,
    )
    if not res:
        print('vLLM generation failed')
        return 1

    v_tokens = res.tokens
    v_token_logprobs = res.token_logprobs
    v_top_logprobs = res.top_logprobs
    print('vLLM returned', len(v_tokens), 'tokens')

    seed_u_values = [
        u_at(
            extra_body['randomness'],
            extra_body['prompt_idx'],
            extra_body['checkpoint_hash'],
            extra_body['rollout_index'],
            t,
        )
        for t in range(len(v_tokens))
    ]

    payload: dict[str, Any] = {
        'prompt': prompt,
        'vllm_model': vllm_model,
        'hf_model': hf_model,
        'device': str(device) if device is not None else None,
        'max_tokens': max_tokens,
        'logprobs': logprobs,
        'tokens': v_tokens,
        'token_logprobs': v_token_logprobs,
        'top_logprobs': v_top_logprobs,
        'seed_u_values': seed_u_values,
        'hf_replay': None,
    }

    if torch is None:
        # Print vLLM-only response details
        print('vLLM tokens:', v_tokens)
        print('vLLM token_logprobs (len):', len(v_token_logprobs))
        if v_top_logprobs:
            print('Sample vLLM top_logprobs for first positions:')
            for i, tl in enumerate(v_top_logprobs[:10]):
                print(f' pos={i} topk={tl}')
        if output_json:
            out_path = Path(output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
            print('Saved rollout payload to', out_path)
        return 0

    # Load HF model and tokenizer
    print('Loading HF model', hf_model)
    hf_model_obj = load_text_generation_model(hf_model, torch_dtype=torch.bfloat16)
    hf_model_obj = hf_model_obj.to(device).eval()
    tokenizer = load_tokenizer(hf_model)

    # Re-run HF forward on the exact token ids returned by vllm
    input_ids = torch.tensor([v_tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    print('Running HF forward_single_layer...')
    with torch.no_grad():
        _, logits = forward_single_layer(hf_model_obj, input_ids, attention_mask, -1)
    logits = logits[0].detach().float().cpu()  # [seq_len, vocab]

    # Compute HF per-token logprobs for the same tokens
    import torch.nn.functional as F
    seq_len = logits.shape[0]
    hf_token_logprobs = []
    for i in range(seq_len):
        lp = F.log_softmax(logits[i], dim=-1)
        tid = int(v_tokens[i])
        hf_token_logprobs.append(float(lp[tid].item()))

    payload['hf_replay'] = {
        'hf_token_logprobs': hf_token_logprobs,
        'diffs': [abs(a - b) for a, b in zip(hf_token_logprobs, v_token_logprobs)],
    }

    # Compare stats
    diffs = payload['hf_replay']['diffs']
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    print(f'Per-token logprob diffs | max={max_diff:.6g} mean={mean_diff:.6g}')

    # Print sample positions with largest diffs
    top_n = min(10, len(diffs))
    idxs = sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[:top_n]
    for i in idxs:
        print(f'pos={i} token={v_tokens[i]} vllm_lp={v_token_logprobs[i]:.6g} hf_lp={hf_token_logprobs[i]:.6g} diff={diffs[i]:.6g}')
        # compare topk
        if v_top_logprobs and i < len(v_top_logprobs):
            print('  vllm toplogprobs:', v_top_logprobs[i])
            # HF topk
            topk = 10
            hf_top = torch.topk(logits[i], topk)
            print('  hf topk:', list(zip(hf_top.indices.tolist(), [float(x) for x in hf_top.values.tolist()])))

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print('Saved rollout payload to', out_path)

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=False, default='Hello world')
    parser.add_argument('--vllm-url', type=str, default='http://0.0.0.0:8000')
    parser.add_argument('--vllm-model', type=str, default='reliquary')
    parser.add_argument('--hf-model', type=str, default='ReliquaryForge/qwen3.5-4b-reliquary-v4')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--logprobs', type=int, default=1)
    parser.add_argument('--output-json', type=str, default=None, help='Path to save the rollout payload JSON')
    args = parser.parse_args()
    return compare(args.prompt, args.vllm_url, args.vllm_model, args.hf_model, args.device, args.max_tokens, args.logprobs, args.output_json)


if __name__ == '__main__':
    raise SystemExit(main())
