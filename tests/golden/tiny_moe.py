#!/usr/bin/env python3
"""tiny_moe.py: a tiny Qwen3.5-MoE checkpoint and its forward records. Development only.

Step 12 runs the MoE model of the family from a GGUF file (moe.py checks it).
This covers the other way in, a safetensors checkpoint with Hugging Face's
names (the fused gate_up_proj bound as two views per expert): a text-only
Qwen3_5MoeForCausalLM of transformers 5.24.0 with small dimensions and random
weights (seeded), saved with save_pretrained, and the forward records of
dump.py's layout for a few prompts of random token ids, each followed by
greedy decode steps of the bf16 reference, in bf16 and in fp32.

The dimensions are small but keep what matters: more value heads than key
heads in the Gated DeltaNet (two per key head, the grouped order), grouped
query attention, 8 experts with 2 per token, a shared expert, partial RoPE.

    tests/golden/.venv/bin/python tests/golden/tiny_moe.py

writes tests/golden/data/tiny-moe/{model/,golden.safetensors,manifest.json};
tests/test_forward.c replays the records with ATH_MODEL_DIR=.../tiny-moe/model
and ATH_GOLDEN_DIR=.../tiny-moe.
"""

import json
import os
import platform
import sys

import torch
import transformers
from safetensors.torch import save_file
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as ref
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

sys.path.insert(0, os.path.dirname(__file__))
from moe import MODELING_SHA256, PINNED_TRANSFORMERS, kl_per_position  # noqa: E402

SEED = 12
DECODE = 4
PROMPTS = (("a", 23), ("b", 70), ("c", 9))   # name, prompt tokens (70: two chunks of the chunked rule)


def config():
    return Qwen3_5MoeTextConfig(
        vocab_size=512, hidden_size=128, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        full_attention_interval=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64,
        shared_expert_intermediate_size=64, rms_norm_eps=1e-6, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    )


def run(model, ids, steps, decode_ids=None):
    """Prefill, then `steps` decode steps; the residual stream entering every
    layer and the final norm, and the logits, as dump.py records them."""
    lm = model.model
    seen = {}

    def capture(i):
        def hook(mod, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            seen[i] = x[0].detach().clone()
        return hook

    n = len(lm.layers)
    hooks = [layer.register_forward_pre_hook(capture(i), with_kwargs=True) for i, layer in enumerate(lm.layers)]
    hooks.append(lm.norm.register_forward_pre_hook(capture(n), with_kwargs=True))
    fed, res, logits = list(ids), [], []
    out = model(input_ids=torch.tensor([ids]), use_cache=True)
    cache = out.past_key_values
    res.append(torch.stack([seen[i] for i in range(n + 1)]))
    logits.append(out.logits[0])
    for s in range(steps):
        nxt = decode_ids[s] if decode_ids is not None else int(out.logits[0, -1].argmax())
        fed.append(nxt)
        out = model(input_ids=torch.tensor([[nxt]]), cache=cache, use_cache=True)
        res.append(torch.stack([seen[i] for i in range(n + 1)]))
        logits.append(out.logits[0])
    for h in hooks:
        h.remove()
    return fed, torch.cat(res, 1), torch.cat(logits, 0)


def main():
    assert transformers.__version__ == PINNED_TRANSFORMERS
    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    out = os.path.join(os.path.dirname(__file__), "data", "tiny-moe")
    mdir = os.path.join(out, "model")
    cfg = config()
    cfg._attn_implementation = "eager"
    model = ref.Qwen3_5MoeForCausalLM(cfg).eval()
    for name, p in model.named_parameters():   # weights of a trained model's scale, not zero norms
        if name.endswith("norm.weight"):
            p.normal_(0.0, 0.1)
        elif p.dim() >= 2:
            p.normal_(0.0, 0.06)
        else:
            p.uniform_(-0.5, 0.5)
    model = model.to(torch.bfloat16)
    # the published layout (fused experts: mlp.experts.gate_up_proj [E, 2I, H],
    # docs/contract/qwen35-35b-a3b.md), as the model holds its weights;
    # save_pretrained would write one tensor per expert instead
    os.makedirs(mdir, exist_ok=True)
    cfg_dict = cfg.to_dict()
    cfg_dict.pop("_attn_implementation", None)
    cfg_dict["architectures"] = ["Qwen3_5MoeForCausalLM"]
    with open(os.path.join(mdir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=1)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(mdir, "model.safetensors"),
              metadata={"format": "pt"})
    model32 = ref.Qwen3_5MoeForCausalLM(cfg).eval()
    model32.load_state_dict({k: v.float() for k, v in model.state_dict().items()})

    g = torch.Generator().manual_seed(SEED)
    tensors, recs = {}, []
    for name, n in PROMPTS:
        ids = torch.randint(0, cfg.vocab_size, (n,), generator=g).tolist()
        fed, res, lg = run(model, ids, DECODE)
        fed32, res32, lg32 = run(model32, ids, DECODE, decode_ids=fed[n:])
        assert fed == fed32
        kl = kl_per_position(lg32, lg)
        rid = f"fwd.{name}"
        tensors[f"{rid}.tokens"] = torch.tensor(fed, dtype=torch.int32)
        tensors[f"{rid}.residual"] = res.contiguous()
        tensors[f"{rid}.residual32"] = res32.contiguous()
        tensors[f"{rid}.logits"] = lg.contiguous()
        tensors[f"{rid}.logits32"] = lg32.contiguous()
        recs.append({"id": rid, "name": name, "prompt": "", "n_prompt": n, "n_decode": DECODE, "tokens": fed,
                     "kl_mean": kl.mean().item(), "kl_max": kl.max().item(),
                     "argmax_differs": int((lg.float().argmax(-1) != lg32.argmax(-1)).sum())})
        print(f"tiny_moe.py: '{name}': {n} + {DECODE} tokens, bf16 vs fp32 KL mean {kl.mean().item():.3e}")
    data = os.path.join(out, "golden.safetensors")
    save_file(tensors, data)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({"format": 1, "about": "athanor: tiny Qwen3.5-MoE forward records (tests/golden/tiny_moe.py)",
                   "reference": {"transformers": transformers.__version__,
                                 "modeling_qwen3_5_moe.py sha256": MODELING_SHA256, "torch": torch.__version__,
                                 "python": platform.python_version()},
                   "seed": SEED, "records": [], "forward": recs}, f, indent=1)
        f.write("\n")
    print(f"tiny_moe.py: model in {mdir}, records in {out}")


if __name__ == "__main__":
    sys.exit(main())
