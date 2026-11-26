#!/usr/bin/env python3
"""kv_int8.py: what an int8 KV cache costs the reference. Development only.

Step 14 stores the KV cache in int8: groups of g values of a key or value
row (one token, one KV head) with an f32 scale each, s = max|x| / 127,
q = rint(x / s), read back as q * s (src/ops/kv.c). This script runs the
reference (transformers 5.24.0, eager attention) in bf16 with its keys and
values replaced by what such a cache reads back, just before the attention,
and reports the mean KL against the fp32 reference over moe.py's prompts:
the same measure as test_forward's gate. It chose the format (a scale per
32 values rather than per row of 256) and gives the bound of the int8 runs
in forward_golden (KV_INT8_KL): the bf16 reference with athanor's cache.

    tests/golden/.venv/bin/python tests/golden/kv_int8.py models/Qwen3.5-0.8B

Measured 2026-11-26 on Qwen3.5-0.8B, 4 prompts, 234 positions (mean KL; the
bf16 reference alone 5.262e-4):

    a scale per row (256), keys and values   1.278e-3   (2.43 times)
                           keys only         5.287e-4
                           values only       1.243e-3
    groups of 64, keys and values            7.817e-4
    groups of 32, keys and values            6.394e-4   (1.21 times)
    groups of 16, values; keys per row       5.776e-4
"""

import argparse
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as ref

from moe import PROMPTS, kl_per_position

PINNED_TRANSFORMERS = "5.24.0"


def int8_groups(x, g):
    """What athanor's int8 cache reads back for x [..., d], a scale per g values."""
    sh = x.shape
    xf = x.float().reshape(*sh[:-1], sh[-1] // g, g)
    s = xf.abs().amax(-1, keepdim=True) / 127
    q = torch.where(s > 0, torch.round(xf / torch.where(s > 0, s, 1)).clamp(-127, 127), 0)
    return (q * s).reshape(sh).to(x.dtype)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model_dir")
    args = p.parse_args()
    assert transformers.__version__ == PINNED_TRANSFORMERS, transformers.__version__
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    ids = [torch.tensor([tok.encode(text, add_special_tokens=False)]) for _, text in PROMPTS]
    m32 = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32, attn_implementation="eager")
    logits32 = [m32(x).logits[0] for x in ids]
    del m32
    m16 = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16, attn_implementation="eager")

    # keys after RoPE and values, as the cache holds them: the attention's inputs
    store = {}
    eager = ref.eager_attention_forward

    def attention(module, query, key, value, *a, **kw):
        if store.get("k"):
            key = store["k"](key)
        if store.get("v"):
            value = store["v"](value)
        return eager(module, query, key, value, *a, **kw)

    ref.eager_attention_forward = attention
    ref.ALL_ATTENTION_FUNCTIONS["eager"] = attention

    def run(name, k=None, v=None):
        store.clear()
        store.update(k=k, v=v)
        kl, n = 0.0, 0
        for x, lg32 in zip(ids, logits32):
            kl += kl_per_position(lg32, m16(x).logits[0]).sum().item()
            n += x.shape[1]
        return kl / n

    base = run("bf16")
    print(f"kv_int8.py: bf16 reference, the cache in bf16: mean KL {base:.3e} over {sum(x.shape[1] for x in ids)} "
          "positions")
    d = m16.config.get_text_config().head_dim
    rows = [("a scale per row, keys and values", d, d), ("a scale per row, keys only", d, None),
            ("a scale per row, values only", None, d), ("groups of 64, keys and values", 64, 64),
            ("groups of 32, keys and values", 32, 32), ("groups of 16, values; keys per row", d, 16)]
    for name, gk, gv in rows:
        kl = run(name, (lambda t, g=gk: int8_groups(t, g)) if gk else None,
                 (lambda t, g=gv: int8_groups(t, g)) if gv else None)
        print(f"kv_int8.py: {name:36s} mean KL {kl:.3e} ({kl / base:.2f} times)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
