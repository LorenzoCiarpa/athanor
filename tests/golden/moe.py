#!/usr/bin/env python3
"""moe.py: forward records of Qwen3.6-35B-A3B from its Q8_0 GGUF. Development only.

The reference implementation (transformers 5.24.0, Qwen3_5Moe*, the classes of
docs/contract/qwen35-35b-a3b.md) cannot load the GGUF, and the model in fp32
(140 GB) does not fit in memory. So this script runs the reference's own
modules one decoder layer at a time:

  - every tensor is read from the GGUF and dequantized exactly (a Q8_0 weight
    is d * q, exact in fp32);
  - the converter's transforms are undone (docs/contract/qwen36-35b-a3b.md):
    1 is subtracted from the zero-centered norm weights, A_log = log(-ssm_a),
    the Gated DeltaNet's value heads go back from tiled to grouped order, the
    convolution kernel gets its middle dimension back, gate and up are
    concatenated into gate_up_proj;
  - one Qwen3_5MoeDecoderLayer at a time is built with those weights, run on
    the residual stream of every prompt, and dropped; then the final norm and
    the output head.

Each prompt runs twice: with the modules in fp32 (the exact model on these
weights) and in bf16 (the weights rounded to bf16, the activations in bf16:
transformers as one runs it in bf16). The records have the layout of
dump.py's forward records (tokens, residual, residual32, logits, logits32,
and the manifest's KL), with prompt positions only: a decode step would mean
streaming every layer again per token. tests/test_forward.c (forward_golden)
replays them with ATH_MODEL_DIR and ATH_GOLDEN_DIR pointing here.

    tests/golden/.venv/bin/python tests/golden/moe.py models/Qwen3.6-35B-A3B

writes tests/golden/data/qwen36-35b-a3b/{golden.safetensors,manifest.json}.
About 8 GB of memory at the peak (one layer's experts in fp32), some minutes.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import sys

import numpy as np
import torch
import transformers
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as ref
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

PINNED_TRANSFORMERS = "5.24.0"
MODELING_SHA256 = "aa3caec05ddf172964b6cfe9b19d333dde892ee99b75bc629c4c18169860f5d6"
GGUF_NAME = "Qwen3.6-35B-A3B-Q8_0.gguf"
GGUF_SHA256 = "c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131"

PROMPTS = (
    ("prose", "The athanor was the alchemist's furnace: a tower of brick or clay that kept a slow, "
              "even heat for weeks, so that a long work could go on without the fire ever going out. "
              "Its name comes from the Arabic al-tannur, an oven, and old treatises called it the "
              "slow furnace, because it never hurried and never went cold."),
    ("chat", "<|im_start|>user\nWhat does an alchemist's athanor do? Answer in one sentence.<|im_end|>\n"
             "<|im_start|>assistant\n<think>\n\n</think>\n\nAn athanor is a furnace that"),
    ("code", "def binary_search(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n"
             "        mid = (lo + hi) // 2\n        if xs[mid] == target:\n            return mid\n"
             "        if xs[mid] < target:\n            lo = mid + 1\n"),
    ("italiano", "La fornace degli alchimisti, l'atanor, teneva un calore lento e costante per settimane: "
                 "una torre di mattoni in cui il fuoco non si spegneva mai, perché l'opera richiedeva pazienza."),
)


# ---- GGUF ----------------------------------------------------------------------------

class Gguf:
    """The tensors of a GGUF file (version 2 or 3), mapped, by name."""

    SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def __init__(self, path):
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        buf = self.mm
        self.o = 0

        def rd(fmt):
            v = struct.unpack_from("<" + fmt, buf, self.o)
            self.o += struct.calcsize("<" + fmt)
            return v[0]

        def rstr():
            n = rd("Q")
            s = bytes(buf[self.o:self.o + n]).decode("utf-8", "replace")
            self.o += n
            return s

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                at, n = rd("I"), rd("Q")
                return [rval(at) for _ in range(n)]
            return rd(self.SCALAR[t])

        assert rd("I") == 0x46554747, "not a GGUF file"
        assert rd("I") in (2, 3)
        n_tensors, n_kv = rd("Q"), rd("Q")
        self.kv = {}
        for _ in range(n_kv):
            k = rstr()
            self.kv[k] = rval(rd("I"))
        infos = []
        for _ in range(n_tensors):
            name = rstr()
            nd = rd("I")
            ne = [rd("Q") for _ in range(nd)]
            infos.append((name, ne, rd("I"), rd("Q")))
        align = self.kv.get("general.alignment", 32)
        data = (self.o + align - 1) // align * align
        self.t = {name: (list(reversed(ne)), typ, data + off) for name, ne, typ, off in infos}

    def get(self, name):
        """The tensor as fp32 (exact for F32, F16, BF16 and Q8_0)."""
        shape, typ, off = self.t[name]
        n = math.prod(shape)
        if typ == 0:
            a = np.frombuffer(self.mm, np.float32, n, off)
        elif typ == 1:
            a = np.frombuffer(self.mm, np.float16, n, off).astype(np.float32)
        elif typ == 30:
            a = (np.frombuffer(self.mm, np.uint16, n, off).astype(np.uint32) << 16).view(np.float32)
        elif typ == 8:
            blocks = np.frombuffer(self.mm, np.uint8, n // 32 * 34, off).reshape(-1, 34)
            d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
            q = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
            a = (d * q).reshape(-1)
        else:
            raise ValueError(f"{name}: GGUF type {typ}")
        return torch.from_numpy(np.array(a, dtype=np.float32).reshape(shape))


# ---- the converter's transforms, undone ------------------------------------------------

def untile(x, dim, nk, per, hd):
    """Value heads from llama.cpp's tiled order (head r * nk + k reads key head
    k) back to the reference's grouped order (head k * per + r)."""
    shape = list(x.shape)
    x = x.reshape(shape[:dim] + [per, nk, hd] + shape[dim + 1:])
    x = x.transpose(dim, dim + 1)
    return x.reshape(shape)


def layer_state(g, l, cfg):
    nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    per = nv // nk
    b = f"blk.{l}."
    sd = {
        "input_layernorm.weight": g.get(b + "attn_norm.weight") - 1,
        "post_attention_layernorm.weight": g.get(b + "post_attention_norm.weight") - 1,
        "mlp.gate.weight": g.get(b + "ffn_gate_inp.weight"),
        "mlp.experts.gate_up_proj": torch.cat([g.get(b + "ffn_gate_exps.weight"), g.get(b + "ffn_up_exps.weight")], 1),
        "mlp.experts.down_proj": g.get(b + "ffn_down_exps.weight"),
        "mlp.shared_expert.gate_proj.weight": g.get(b + "ffn_gate_shexp.weight"),
        "mlp.shared_expert.up_proj.weight": g.get(b + "ffn_up_shexp.weight"),
        "mlp.shared_expert.down_proj.weight": g.get(b + "ffn_down_shexp.weight"),
        "mlp.shared_expert_gate.weight": g.get(b + "ffn_gate_inp_shexp.weight")[None, :],
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
    qkv = g.get(b + "attn_qkv.weight")
    kd = nk * dk
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


# ---- the run ----------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def kl_per_position(logits32, logits):
    lp32 = torch.log_softmax(logits32.double(), dim=-1)
    lp = torch.log_softmax(logits.double(), dim=-1)
    return (lp32.exp() * (lp32 - lp)).sum(dim=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model_dir")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "qwen36-35b-a3b"))
    p.add_argument("--no-hash", action="store_true", help="do not check the GGUF's sha256 (it takes a minute)")
    args = p.parse_args()
    assert transformers.__version__ == PINNED_TRANSFORMERS, transformers.__version__
    with open(ref.__file__, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == MODELING_SHA256, "the reference file changed"
    gpath = os.path.join(args.model_dir, GGUF_NAME)
    if not args.no_hash:
        assert sha256_file(gpath) == GGUF_SHA256, "another GGUF file"
    torch.set_grad_enabled(False)

    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg_json = json.load(f)
    cfg = Qwen3_5MoeTextConfig(**cfg_json["text_config"])
    cfg._attn_implementation = "eager"
    g = Gguf(gpath)
    tok = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    ids = [tok.encode(text, add_special_tokens=False).ids for _, text in PROMPTS]

    embed = g.get("token_embd.weight")
    rot = ref.Qwen3_5MoeTextRotaryEmbedding(cfg)
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
            layer = ref.Qwen3_5MoeDecoderLayer(cfg, l)
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
            print(f"moe.py: {str(dtype)[6:]} layer {l + 1} of {cfg.num_hidden_layers}", flush=True)
        norm = ref.Qwen3_5MoeRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        norm.weight.data = g.get("output_norm.weight") - 1
        norm = norm.to(dtype)
        head = g.get("output.weight").to(dtype)
        logits = []
        for i in range(len(ids)):
            logits.append(norm(hs[i])[0] @ head.T)
        runs[dtype] = (res, logits)
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
        print(f"moe.py: '{name}': {len(ids[i])} tokens, bf16 reference vs fp32: KL mean {kl.mean().item():.3e}, "
              f"max {kl.max().item():.3e}")
    os.makedirs(args.out, exist_ok=True)
    data = os.path.join(args.out, "golden.safetensors")
    save_file(tensors, data)
    manifest = {
        "format": 1,
        "about": "athanor forward records of Qwen3.6-35B-A3B Q8_0: tests/golden/moe.py, replayed by tests/test_forward.c",
        "reference": {"transformers": transformers.__version__, "modeling_qwen3_5_moe.py sha256": MODELING_SHA256,
                      "torch": torch.__version__, "python": platform.python_version(),
                      "threads": torch.get_num_threads()},
        "model": {"gguf": GGUF_NAME, "gguf sha256": GGUF_SHA256},
        "data": {"file": "golden.safetensors", "sha256": sha256_file(data), "tensors": len(tensors)},
        "records": [],
        "forward": recs,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(f"moe.py: {len(recs)} prompts, {os.path.getsize(data) / 2**20:.1f} MiB in {args.out}")


if __name__ == "__main__":
    sys.exit(main())
