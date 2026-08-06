#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reliquary.shared.modeling import load_text_generation_model, load_tokenizer
from reliquary.shared.forward import forward_single_layer


def compare_logits(hf_model_name: str, tokenizer_name: str | None, prompt: str, device: str):
    device = torch.device(device)
    print('Loading tokenizer...')
    tokenizer = load_tokenizer(tokenizer_name or hf_model_name)

    # Default tiny model for safe run if requested model missing
    print('Loading HF model (float32)...')
    try:
        hf_fp32 = load_text_generation_model(hf_model_name, torch_dtype=torch.float32)
        hf_fp32 = hf_fp32.to(device).eval()
    except Exception as exc:
        print('Failed loading HF float32 model:', exc)
        return 1

    # Try to load bf16 variant if possible
    hf_bf16 = None
    try:
        print('Loading HF model (bfloat16)...')
        hf_bf16 = load_text_generation_model(hf_model_name, torch_dtype=torch.bfloat16)
        hf_bf16 = hf_bf16.to(device).eval()
    except Exception as exc:
        print('Could not load bfloat16 variant (continuing without):', exc)

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], device=device)
    attention_mask = torch.ones_like(input_tensor)

    print('Running forward_single_layer on float32 model...')
    with torch.no_grad():
        _, logits_fp32 = forward_single_layer(hf_fp32, input_tensor, attention_mask, -1)
    logits_fp32 = logits_fp32.detach().float()

    if hf_bf16 is not None:
        print('Running forward_single_layer on bfloat16 model...')
        with torch.no_grad():
            _, logits_bf16 = forward_single_layer(hf_bf16, input_tensor, attention_mask, -1)
        logits_bf16 = logits_bf16.detach().float()

        # Compare last token logits
        last_fp32 = logits_fp32[0, -1].cpu()
        last_bf16 = logits_bf16[0, -1].cpu()
        diff = (last_fp32 - last_bf16).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        print(f'Last-token logits | max_abs_diff={max_abs:.6g} mean_abs_diff={mean_abs:.6g}')

        # Top-10 tokens in each
        topk = 10
        fp32_top = torch.topk(last_fp32, topk)
        bf16_top = torch.topk(last_bf16, topk)
        print('Top tokens FP32:', list(zip(fp32_top.indices.tolist(), fp32_top.values.tolist())))
        print('Top tokens BF16:', list(zip(bf16_top.indices.tolist(), bf16_top.values.tolist())))

        # Show any rank differences in topk
        fp32_ranks = {int(t): i for i, t in enumerate(fp32_top.indices.tolist())}
        rank_changes = []
        for i, tid in enumerate(bf16_top.indices.tolist()):
            if int(tid) not in fp32_ranks:
                rank_changes.append((int(tid), 'new_in_bf16'))
            else:
                rank_changes.append((int(tid), fp32_ranks[int(tid)] - i))
        print('Rank changes (bf16 topk vs fp32):', rank_changes)
    else:
        print('bfloat16 variant not available; only FP32 logits computed. You can rerun with a model that supports bf16.')

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sshleifer/tiny-gpt2')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--prompt', type=str, default='Hello world')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    return compare_logits(args.model, args.tokenizer, args.prompt, args.device)


if __name__ == '__main__':
    raise SystemExit(main())
