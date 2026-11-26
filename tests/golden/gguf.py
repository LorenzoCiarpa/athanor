#!/usr/bin/env python3
"""gguf.py: forward records of a dense Qwen3.5 model from its GGUF. Development only.

moe.py's method for a dense checkpoint (step 14: Qwen3.5-9B in Q8_0, whose
bf16 weights do not fit the GPU and whose fp32 weights, 36 GB, would fill the
machine's memory twice over with transformers' copies). The reference
implementation (transformers 5.24.0, Qwen3_5*, the classes of
docs/contract/qwen35-0.8b.md) runs one decoder layer at a time:

  - every tensor is read from the GGUF and dequantized exactly (moe.py's
    reader: a Q8_0 weight is d * q, exact in fp32);
  - the converter's transforms are undone: 1 subtracted from the zero-centered
    norm weights, A_log = log(-ssm_a), the Gated DeltaNet's value heads back
    from tiled to grouped order, the convolution kernel's middle dimension;
  - one Qwen3_5DecoderLayer at a time is built with those weights, run on the
    residual stream of every prompt, and dropped; then the final norm and the
    output head.

Each prompt runs in fp32 (the exact model on these weights) and in bf16, as
moe.py; the records have dump.py's layout, prompt positions only, and
tests/test_forward.c (forward_golden) replays them with ATH_MODEL_DIR and
ATH_GOLDEN_DIR pointing at them.

    tests/golden/.venv/bin/python tests/golden/gguf.py models/Qwen3.5-9B

writes tests/golden/data/qwen35-9b/{golden.safetensors,manifest.json}.
"""

import argparse
import hashlib
import json
import os
import platform
import sys

import torch
import transformers
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5 import modeling_qwen3_5 as ref
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from moe import PROMPTS, Gguf, kl_per_position, sha256_file, untile

PINNED_TRANSFORMERS = "5.24.0"
MODELING_SHA256 = "c4d847b7e809ed2a49611e5afcea8bd0db39de597b67b57ce9753fb65f550de2"   # as dump.py
MODELS = {
    "Qwen3.5-9B-Q8_0.gguf": ("809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4", "qwen35-9b"),
}


def layer_state(g, l, cfg):
    nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    per = nv // nk
    b = f"blk.{l}."
    sd = {
        "input_layernorm.weight": g.get(b + "attn_norm.weight") - 1,
        "post_attention_layernorm.weight": g.get(b + "post_attention_norm.weight") - 1,
        "mlp.gate_proj.weight": g.get(b + "ffn_gate.weight"),
        "mlp.up_proj.weight": g.get(b + "ffn_up.weight"),
        "mlp.down_proj.weight": g.get(b + "ffn_down.weight"),
    }
    if cfg.layer_types[l] == "full_attention":
        sd.update({
            "self_attn.q_proj.weight": g.get(b + "attn_q.weight"),
            "self_attn.k_proj.weight": g.get(b + "attn_k.weight"),
            "self_attn.v_proj.weight": g.get(b + "attn_v.weight"),
            "self_attn.o_proj.weight": g.get(b + "attn_output.weight"),
            "self_attn.q_norm.weight": g.get(b + "attn_q_norm.weight") - 1,
            "self_attn.k_norm.weight": g.get(b + "attn_k_norm.weight") - 1,
        })
        return sd
    kd = nk * dk
    qkv = g.get(b + "attn_qkv.weight")
    qkv = torch.cat([qkv[:2 * kd], untile(qkv[2 * kd:], 0, nk, per, dv)], 0)
    conv = g.get(b + "ssm_conv1d.weight")
    conv = torch.cat([conv[:2 * kd], untile(conv[2 * kd:], 0, nk, per, dv)], 0)[:, None, :]
    a = -g.get(b + "ssm_a").double()
    sd.update({
        "linear_attn.in_proj_qkv.weight": qkv,
        "linear_attn.in_proj_z.weight": untile(g.get(b + "attn_gate.weight"), 0, nk, per, dv),
        "linear_attn.in_proj_b.weight": untile(g.get(b + "ssm_beta.weight"), 0, nk, per, 1),
        "linear_attn.in_proj_a.weight": untile(g.get(b + "ssm_alpha.weight"), 0, nk, per, 1),
        "linear_attn.A_log": untile(torch.log(a).float(), 0, nk, per, 1),
        "linear_attn.dt_bias": untile(g.get(b + "ssm_dt.bias"), 0, nk, per, 1),
        "linear_attn.conv1d.weight": conv,
        "linear_attn.norm.weight": g.get(b + "ssm_norm.weight"),
        "linear_attn.out_proj.weight": untile(g.get(b + "ssm_out.weight"), 1, nk, per, dv),
    })
    return sd


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model_dir")
    p.add_argument("--out", help="default: tests/golden/data/<the model's folder>")
    p.add_argument("--no-hash", action="store_true", help="do not check the GGUF's sha256")
    args = p.parse_args()
    assert transformers.__version__ == PINNED_TRANSFORMERS, transformers.__version__
    with open(ref.__file__, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == MODELING_SHA256, "the reference file changed"
    names = [n for n in os.listdir(args.model_dir) if n in MODELS]
    assert len(names) == 1, f"{args.model_dir}: none of {list(MODELS)}"
    gname = names[0]
    gsha, outname = MODELS[gname]
    gpath = os.path.join(args.model_dir, gname)
    if not args.no_hash:
        assert sha256_file(gpath) == gsha, "another GGUF file"
    out = args.out or os.path.join(os.path.dirname(__file__), "data", outname)
    torch.set_grad_enabled(False)

    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg_json = json.load(f)
    cfg = Qwen3_5TextConfig(**cfg_json["text_config"])
    cfg._attn_implementation = "eager"
    g = Gguf(gpath)
    tok = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    ids = [tok.encode(text, add_special_tokens=False).ids for _, text in PROMPTS]

    embed = g.get("token_embd.weight")
    rot = ref.Qwen3_5TextRotaryEmbedding(cfg)
    runs = {}
    for dtype in (torch.float32, torch.bfloat16):
        hs = [embed[torch.tensor(x)].to(dtype)[None] for x in ids]
        res = [[h[0].clone()] for h in hs]
        aux = []
        for x, h in zip(ids, hs):
            T = len(x)
            pos = torch.arange(T)[None].expand(4, 1, T)
            text_pos, mpos = pos[0], pos[1:]
            mask = create_causal_mask(config=cfg, inputs_embeds=h, attention_mask=None, cache=None,
                                      position_ids=text_pos)
            aux.append((rot(h, mpos), mask, text_pos))
        for l in range(cfg.num_hidden_layers):
            layer = ref.Qwen3_5DecoderLayer(cfg, l)
            missing, unexpected = layer.load_state_dict(layer_state(g, l, cfg), strict=False)
            assert not unexpected and not [k for k in missing if not k.endswith("rotary_emb.inv_freq")], (missing,
                                                                                                          unexpected)
            layer = layer.to(dtype).eval()
            for i, (pe, mask, text_pos) in enumerate(aux):
                pe = tuple(t.to(dtype) for t in pe)
                hs[i] = layer(hs[i], position_embeddings=pe,
                              attention_mask=mask if cfg.layer_types[l] == "full_attention" else None,
                              position_ids=text_pos, cache=None)
                res[i].append(hs[i][0].clone())
            del layer
            print(f"gguf.py: {str(dtype)[6:]} layer {l + 1} of {cfg.num_hidden_layers}", flush=True)
        norm = ref.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        norm.weight.data = g.get("output_norm.weight") - 1
        norm = norm.to(dtype)
        head = g.get("output.weight" if "output.weight" in g.t else "token_embd.weight").to(dtype)
        runs[dtype] = (res, [norm(hs[i])[0] @ head.T for i in range(len(ids))])
        del head

    tensors, recs = {}, []
    for i, (name, text) in enumerate(PROMPTS):
        res16, lg16 = runs[torch.bfloat16][0][i], runs[torch.bfloat16][1][i]
        res32, lg32 = runs[torch.float32][0][i], runs[torch.float32][1][i]
        kl = kl_per_position(lg32, lg16)
        rid = f"fwd.{name}"
        tensors[f"{rid}.tokens"] = torch.tensor(ids[i], dtype=torch.int32)
        tensors[f"{rid}.residual"] = torch.stack(res16).contiguous()
        tensors[f"{rid}.residual32"] = torch.stack(res32).contiguous()
        tensors[f"{rid}.logits"] = lg16.contiguous()
        tensors[f"{rid}.logits32"] = lg32.contiguous()
        recs.append({"id": rid, "name": name, "prompt": text, "n_prompt": len(ids[i]), "n_decode": 0,
                     "tokens": ids[i], "kl_mean": kl.mean().item(), "kl_max": kl.max().item(),
                     "argmax_differs": int((lg16.float().argmax(-1) != lg32.argmax(-1)).sum())})
        print(f"gguf.py: '{name}': {len(ids[i])} tokens, bf16 reference vs fp32: KL mean {kl.mean().item():.3e}, "
              f"max {kl.max().item():.3e}")
    os.makedirs(out, exist_ok=True)
    data = os.path.join(out, "golden.safetensors")
    save_file(tensors, data)
    manifest = {
        "format": 1,
        "about": f"athanor forward records of {gname}: tests/golden/gguf.py, replayed by tests/test_forward.c",
        "reference": {"transformers": transformers.__version__, "modeling_qwen3_5.py sha256": MODELING_SHA256,
                      "torch": torch.__version__, "python": platform.python_version(),
                      "threads": torch.get_num_threads()},
        "model": {"gguf": gname, "gguf sha256": gsha},
        "data": {"file": "golden.safetensors", "sha256": sha256_file(data), "tensors": len(tensors)},
        "records": [],
        "forward": recs,
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(f"gguf.py: {len(recs)} prompts, {os.path.getsize(data) / 2**20:.1f} MiB in {out}")


if __name__ == "__main__":
    sys.exit(main())
