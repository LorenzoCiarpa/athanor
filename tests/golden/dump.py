#!/usr/bin/env python3
"""dump.py: golden data for athanor's op tests. Development only.

Runs the pinned reference implementation (transformers 5.24.0, see
docs/contract/qwen35-0.8b.md) on the CPU over a fixed prompt, then one decode
step, and records every op the reference computes as one record:

  - the op's inputs, exactly as the reference had them;
  - `out`: the output in the reference's own dtype (bf16 for activations);
  - `out32`: the output of the same reference function recomputed in fp32 from
    the same inputs (weights upcast). It has no intermediate bf16 rounding, so
    it checks the math of athanor's _ref functions tightly.

tests/test_golden.c replays every record through the _ref functions in
src/ops/ and compares both outputs.

    tests/golden/.venv/bin/python tests/golden/dump.py models/Qwen3.5-0.8B

writes tests/golden/data/qwen35-0.8b/{golden.safetensors,manifest.json}.
The environment is pinned in requirements.txt (tests/golden/README.md).

The reference is run as transformers runs it, with two deliberate choices:

  - attn_implementation="eager", the readable path the contract was written
    from (the default, sdpa, fuses softmax and matmuls);
  - the 36 tensors stored as f32 in the checkpoint (linear_attn.A_log and
    linear_attn.norm.weight) are put back in f32 after loading:
    from_pretrained rounds them to bf16 (it has no _keep_in_fp32_modules for
    this model), while athanor uses them as stored. The forward code is
    unchanged; it already upcasts A_log and promotes the norm weight.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys

import torch
import torch.nn.functional as F
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as ref

PINNED_TRANSFORMERS = "5.24.0"
PINNED_TORCH = "2.15.0+cpu"
MODELING_SHA256 = "c4d847b7e809ed2a49611e5afcea8bd0db39de597b67b57ce9753fb65f550de2"

# Long enough for two chunks of the reference's chunked Gated DeltaNet (64
# tokens each), so the chunk boundary is part of the golden data.
PROMPT = (
    "The athanor was the alchemist's furnace: a tower of brick or clay that kept a slow, "
    "even heat for weeks, so that a long work could go on without the fire ever going out. "
    "Its name comes from the Arabic al-tannur, an oven, and old treatises called it the "
    "slow furnace, because it never hurried and never went cold."
)

CKPT_PREFIX = "model.language_model."


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def check_environment():
    """The golden data is only meaningful for the pinned reference."""
    if transformers.__version__ != PINNED_TRANSFORMERS:
        sys.exit(f"dump.py: transformers {transformers.__version__}, expected {PINNED_TRANSFORMERS}")
    if torch.__version__ != PINNED_TORCH:
        sys.exit(f"dump.py: torch {torch.__version__}, expected {PINNED_TORCH}")
    if sha256_file(ref.__file__) != MODELING_SHA256:
        sys.exit(f"dump.py: {ref.__file__} does not match the sha256 in docs/contract/qwen35-0.8b.md")
    # The Gated DeltaNet functions fall back to their torch versions only when
    # none of these packages is importable; otherwise a fused kernel would run.
    for pkg in ("fla", "causal_conv1d", "kernels"):
        if importlib.util.find_spec(pkg) is not None:
            sys.exit(f"dump.py: package '{pkg}' is installed; the reference must use the torch fallbacks")


def ckpt_name(qualname):
    """Module name in Qwen3_5ForCausalLM -> weight name in the checkpoint."""
    assert qualname.startswith("model."), qualname
    return CKPT_PREFIX + qualname[len("model."):] + ".weight"


def restore_f32(model, model_dir):
    """Put the checkpoint's f32 tensors back into the bf16-loaded model."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    restored = []
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(model_dir, shard), "pt") as st:
            for name in st.keys():
                if not name.startswith(CKPT_PREFIX):
                    continue
                if st.get_slice(name).get_dtype() != "F32":
                    continue
                param = model.get_parameter("model." + name[len(CKPT_PREFIX):])
                value = st.get_tensor(name)
                assert param.shape == value.shape and param.dtype == torch.bfloat16
                param.data = value
                restored.append(name)
    return sorted(restored)


class Recorder:
    """Collects records and their tensors; ids are <phase>.[L<layer>.]<name>."""

    def __init__(self):
        self.records = []
        self.tensors = {}
        self.ids = set()
        self.on = True
        self.phase = "p"
        self.layer = None
        self.buf = {}          # per-layer values needed by composite records
        self.positions = None  # int32 [T]
        self.pending_conv = {} # layer -> id of the prefill conv record, for its state

    @contextlib.contextmanager
    def paused(self):
        """No recording while the fp32 reruns call the wrapped functions."""
        was, self.on = self.on, False
        try:
            yield
        finally:
            self.on = was

    def rid(self, name):
        layer = "" if self.layer is None else f"L{self.layer}."
        return f"{self.phase}.{layer}{name}"

    def add(self, op, name, tensors, **params):
        rid = self.rid(name)
        assert rid not in self.ids, rid
        self.ids.add(rid)
        for slot, t in tensors.items():
            self.tensors[f"{rid}.{slot}"] = t.detach().contiguous().clone()
        self.records.append({"op": op, "id": rid, **params})
        return rid


R = Recorder()


def score(got, want):
    """The error measure of tests/test_golden.c: max |got - want| / (|want| + rms(want))."""
    got, want = got.double(), want.double()
    rms = want.pow(2).mean().sqrt()
    return ((got - want).abs() / (want.abs() + rms)).max().item()


def matmul_rows(n):
    """Rows of a matmul kept in the golden data. An output row depends only on
    the same input row, so a few rows check the whole weight; test_golden.c
    computes each kept row in full."""
    return [0] if n == 1 else [0, n - 1]


# ---- module hooks ----------------------------------------------------------


def hook_embedding(qualname):
    def hook(mod, args, out):
        if R.on:
            R.add("embed", "embed_tokens", {"ids": args[0][0].to(torch.int32), "out": out[0]},
                  weight=ckpt_name(qualname))
    return hook


def hook_linear(short, weight):
    def hook(mod, args, out):
        if not R.on:
            return
        x, y = args[0][0], out[0]               # drop the batch dimension: [T, in], [T, out]
        rows = matmul_rows(x.shape[0])
        with R.paused():
            y32 = F.linear(x[rows].float(), mod.weight.float())
        R.add("matmul", short, {"x": x[rows], "out": y[rows], "out32": y32}, weight=weight, rows=rows)
        R.buf[short] = y
        R.buf[short + ".x"] = x
    return hook


def hook_rmsnorm(qualname, short):
    def hook(mod, args, out):
        if not R.on:
            return
        x = args[0]
        with R.paused():
            y32 = mod(x.float())                 # Qwen3_5RMSNorm in fp32: no final rounding
        R.add("rmsnorm", short, {"x": x[0], "out": out[0], "out32": y32[0]},
              weight=ckpt_name(qualname), eps=mod.eps, offset=1.0)
        R.buf[short + ".x"] = x[0]
    return hook


def hook_rmsnorm_gated(qualname, short):
    def hook(mod, args, out):
        if not R.on:
            return
        x, z = args                              # [T * heads, head_dim] each
        with R.paused():
            y32 = mod(x.float(), z.float())
        R.add("rmsnorm_gated", short, {"x": x, "z": z, "out": out, "out32": y32},
              weight=ckpt_name(qualname), eps=mod.variance_epsilon, act=mod.activation)
    return hook


def hook_layer_pre(index):
    def hook(mod, args, kwargs):
        R.layer = index
        R.buf = {"x": (args[0] if args else kwargs["hidden_states"])[0]}
    return hook


def hook_layer_post(index, is_attention):
    """Records the elementwise ops of the decoder layer, from values the other
    hooks saved: residual adds, SwiGLU, attention output gate, GDN gates."""
    def hook(mod, args, out):
        if not R.on:
            R.layer = None
            return
        b = R.buf
        mixer = "self_attn.o_proj" if is_attention else "linear_attn.out_proj"
        h = b["post_attention_layernorm.x"]
        y = out[0]
        R.add("add", "residual_mixer", {"a": b["x"], "b": b[mixer], "out": h,
                                        "out32": b["x"].float() + b[mixer].float()})
        R.add("add", "residual_mlp", {"a": h, "b": b["mlp.down_proj"], "out": y,
                                      "out32": h.float() + b["mlp.down_proj"].float()})
        g, u = b["mlp.gate_proj"], b["mlp.up_proj"]
        R.add("swiglu", "mlp.act", {"gate": g, "up": u, "out": b["mlp.down_proj.x"],
                                    "out32": F.silu(g.float()) * u.float()})
        if is_attention:
            T = g.shape[0]
            head_dim = mod.self_attn.head_dim
            # q_proj holds query and gate interleaved per head (contract, Full-attention layer)
            gate = b["self_attn.q_proj"].view(T, -1, 2 * head_dim)[..., head_dim:].reshape(T, -1)
            x = b["attn_out"]
            R.add("sigmoid_mul", "self_attn.gate", {"x": x, "gate": gate, "out": b["self_attn.o_proj.x"],
                                                    "out32": x.float() * torch.sigmoid(gate.float())})
        else:
            la = mod.linear_attn
            a, bb = b["linear_attn.in_proj_a"], b["linear_attn.in_proj_b"]
            R.add("gdn_gates", "linear_attn.gates",
                  {"a": a, "b": bb, "g": b["g"], "beta": b["beta"], "beta32": torch.sigmoid(bb.float())},
                  a_log=CKPT_PREFIX + f"layers.{index}.linear_attn.A_log",
                  dt_bias=CKPT_PREFIX + f"layers.{index}.linear_attn.dt_bias")
            assert la.A_log.dtype == torch.float32
        R.layer = None
    return hook


# ---- wrapped reference functions ------------------------------------------
# The forward code calls these through module globals, so replacing the
# globals is enough; each wrapper calls the original.

orig_rope = ref.apply_rotary_pos_emb
orig_attn = ref.eager_attention_forward
orig_conv_fn = ref.causal_conv1d_fn
orig_conv_update = ref.causal_conv1d_update
orig_l2norm = ref.l2norm
orig_chunk = ref.torch_chunk_gated_delta_rule
orig_recurrent = ref.torch_recurrent_gated_delta_rule

CONFIG = {}


def rope_wrap(q, k, cos, sin, unsqueeze_dim=1):
    qe, ke = orig_rope(q, k, cos, sin, unsqueeze_dim)
    if R.on:
        # An fp32 run of the reference also has fp32 cos/sin tables: the
        # rotary module casts them to the activation dtype. Recompute them for
        # these positions and check that the bf16 tables are their rounding.
        cos32, sin32 = CONFIG["rotary"](q.float(), R.positions[None].long())
        assert torch.equal(cos32.to(cos.dtype), cos) and torch.equal(sin32.to(sin.dtype), sin)
        with R.paused():
            q32, k32 = orig_rope(q.float(), k.float(), cos32, sin32, unsqueeze_dim)
        tr = lambda t: t[0].transpose(0, 1)      # [heads, T, d] -> [T, heads, d]
        for name, x, y, y32 in (("q", q, qe, q32), ("k", k, ke, k32)):
            R.add("rope", f"self_attn.rope_{name}", {"x": tr(x), "pos": R.positions, "out": tr(y), "out32": tr(y32)},
                  theta=CONFIG["rope_theta"], rope_dim=cos.shape[-1])
    return qe, ke


def attn_wrap(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    out, w = orig_attn(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    if R.on:
        T, S = query.shape[2], key.shape[2]
        if attention_mask is not None:           # must be the plain causal mask
            m = attention_mask[0, 0]
            masked = m < -1e30 if m.dtype != torch.bool else ~m
            causal = torch.ones(T, S, dtype=torch.bool).triu(S - T + 1)
            assert torch.equal(masked, causal), "attention mask is not causal"
        mask32 = None if attention_mask is None or attention_mask.dtype == torch.bool else attention_mask.float()
        with R.paused():
            o32, _ = orig_attn(module, query.float(), key.float(), value.float(), mask32, scaling=scaling,
                               dropout=dropout, **kwargs)
        # The native tolerance of attn in tests/test_golden.c rests on this:
        # the eager path rounds q·kᵀ to bf16 before the softmax. Keep the
        # scores in fp32, round the rest as eager does, and the output is
        # within 2 bf16 ulps of the fp32 rerun.
        groups = module.num_key_value_groups
        scores = torch.matmul(query.float(), ref.repeat_kv(key, groups).float().transpose(2, 3)) * scaling
        scores = scores.masked_fill(torch.ones(T, S, dtype=torch.bool).triu(S - T + 1), float("-inf"))
        p = torch.softmax(scores, dim=-1).to(query.dtype)
        emulated = torch.matmul(p, ref.repeat_kv(value, groups)).transpose(1, 2)
        assert score(emulated, o32) <= 2 / 128, score(emulated, o32)
        tr = lambda t: t[0].transpose(0, 1)
        R.add("attn", "self_attn.attn", {"q": tr(query), "k": tr(key), "v": tr(value), "out": out[0],
                                         "out32": o32[0]}, scale=scaling)
        R.buf["attn_out"] = out[0].reshape(T, -1)
    return out, w


def conv_fn_wrap(hidden_states, weight, bias=None, activation=None, **kwargs):
    out = orig_conv_fn(hidden_states, weight, bias, activation, **kwargs)
    if R.on:
        assert bias is None
        with R.paused():
            o32 = orig_conv_fn(hidden_states.float(), weight.float(), None, activation)
        # [1, channels, T] -> [T, channels]; prefill from an empty cache: zero state
        rid = R.add("conv", "linear_attn.conv1d",
                    {"x": hidden_states[0].T, "out": out[0].T, "out32": o32[0].T},
                    weight=CKPT_PREFIX + f"layers.{R.layer}.linear_attn.conv1d.weight", act=activation)
        R.pending_conv[R.layer] = rid
    return out


def conv_update_wrap(hidden_states, conv_state, weight, bias=None, activation=None):
    state_in = conv_state.clone()
    out = orig_conv_update(hidden_states, conv_state, weight, bias, activation)
    if R.on:
        assert bias is None
        K = weight.shape[-1]
        assert state_in.shape[-1] == K
        with R.paused():
            o32 = orig_conv_update(hidden_states.float(), state_in.float().clone(), weight.float(), None, activation)
            # The cache keeps K inputs, but a step reads only the last K - 1:
            # athanor's conv state is those K - 1 (contract, Gated DeltaNet).
            probe = state_in.clone()
            probe[..., 0] = 0
            assert torch.equal(orig_conv_update(hidden_states, probe, weight, None, activation), out)
        R.add("conv", "linear_attn.conv1d",
              {"x": hidden_states[0].T, "state_in": state_in[0, :, 1:].T, "out": out[0].T, "out32": o32[0].T,
               "state_out": conv_state[0, :, 1:].T},
              weight=CKPT_PREFIX + f"layers.{R.layer}.linear_attn.conv1d.weight", act=activation)
    return out


def l2norm_wrap(x, dim=-1, eps=1e-6):
    y = orig_l2norm(x, dim=dim, eps=eps)
    if R.on:
        assert dim == -1
        name = "linear_attn.l2norm_q" if "linear_attn.l2norm_q" not in R.buf else "linear_attn.l2norm_k"
        R.buf[name] = True
        R.add("l2norm", name, {"x": x[0], "out": y[0]}, eps=eps)
    return y


def gdn_wrap(orig, form):
    def wrapped(query, key, value, g, beta, initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=False, **kwargs):
        state_in = None if initial_state is None else initial_state.clone()
        out, state = orig(query, key, value, g=g, beta=beta, initial_state=initial_state,
                          output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                          **kwargs)
        if R.on:
            assert use_qk_l2norm_in_kernel and output_final_state
            # Exact-math rerun: the recurrent form (athanor's reference
            # algorithm) in fp32 on the same inputs, also for the chunked prefill.
            with R.paused():
                o32, s32 = orig_recurrent(query.float(), key.float(), value.float(), g=g.float(), beta=beta.float(),
                                          initial_state=None if state_in is None else state_in.clone(),
                                          output_final_state=True, use_qk_l2norm_in_kernel=True)
            t = {"q": query[0], "k": key[0], "v": value[0], "g": g[0], "beta": beta[0],
                 "out": out[0], "out32": o32[0], "state_out": state[0], "state_out32": s32[0]}
            if state_in is not None:
                t["state_in"] = state_in[0]
            dk = query.shape[-1]
            # l2norm eps and the query scale are fixed inside the reference
            # functions: l2norm(..., eps=1e-6) and dk ** -0.5.
            R.add("gdn", "linear_attn.gdn", t, form=form, l2norm_eps=1e-6, q_scale=dk ** -0.5)
            R.buf["g"], R.buf["beta"] = g[0], beta[0]
        return out, state
    return wrapped


def install(model):
    ref.apply_rotary_pos_emb = rope_wrap
    ref.eager_attention_forward = attn_wrap
    ref.causal_conv1d_fn = conv_fn_wrap
    ref.causal_conv1d_update = conv_update_wrap
    ref.l2norm = l2norm_wrap
    ref.torch_chunk_gated_delta_rule = gdn_wrap(orig_chunk, "chunk")
    ref.torch_recurrent_gated_delta_rule = gdn_wrap(orig_recurrent, "recurrent")

    lm = model.model
    lm.embed_tokens.register_forward_hook(hook_embedding("model.embed_tokens"))
    lm.norm.register_forward_hook(hook_rmsnorm("model.norm", "norm"))
    model.lm_head.register_forward_hook(hook_linear("lm_head", CKPT_PREFIX + "embed_tokens.weight"))
    for i, layer in enumerate(lm.layers):
        is_attention = hasattr(layer, "self_attn")
        layer.register_forward_pre_hook(hook_layer_pre(i), with_kwargs=True)
        layer.register_forward_hook(hook_layer_post(i, is_attention))
        for short, mod in layer.named_modules():
            qual = f"model.layers.{i}.{short}"
            if isinstance(mod, torch.nn.Linear):
                mod.register_forward_hook(hook_linear(short, ckpt_name(qual)))
            elif isinstance(mod, ref.Qwen3_5RMSNorm):
                mod.register_forward_hook(hook_rmsnorm(qual, short))
            elif isinstance(mod, ref.Qwen3_5RMSNormGated):
                mod.register_forward_hook(hook_rmsnorm_gated(qual, short))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "qwen35-0.8b"))
    args = ap.parse_args()

    check_environment()
    torch.set_num_threads(1)                     # reproducible reductions
    torch.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()
    restored = restore_f32(model, args.model_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    CONFIG["rope_theta"] = model.config.rope_parameters["rope_theta"]
    CONFIG["rotary"] = model.model.rotary_emb
    install(model)

    ids = tok(PROMPT, return_tensors="pt")["input_ids"]
    T = ids.shape[1]
    assert T > 64, f"prompt has {T} tokens, the chunk boundary needs more than 64"

    with torch.no_grad():
        R.phase, R.positions = "p", torch.arange(T, dtype=torch.int32)
        out = model(input_ids=ids, use_cache=True)
        cache = out.past_key_values
        # Conv state after the prefill: the last K - 1 inputs of each GDN layer.
        for layer, rid in R.pending_conv.items():
            R.tensors[f"{rid}.state_out"] = cache.layers[layer].conv_states[0][0, :, 1:].T.contiguous().clone()
        next_id = int(out.logits[0, -1].argmax())
        R.phase, R.positions = "d", torch.tensor([T], dtype=torch.int32)
        model(input_ids=torch.tensor([[next_id]]), cache=cache, use_cache=True)

    os.makedirs(args.out, exist_ok=True)
    data_path = os.path.join(args.out, "golden.safetensors")
    save_file(R.tensors, data_path)
    manifest = {
        "format": 1,
        "about": "athanor golden data: tests/golden/dump.py, replayed by tests/test_golden.c",
        "reference": {
            "transformers": transformers.__version__,
            "modeling_qwen3_5.py sha256": MODELING_SHA256,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "threads": torch.get_num_threads(),
            "cpu": cpu_name(),
            "attn_implementation": model.config._attn_implementation,
            "dtype": "bfloat16",
            "restored_f32": restored,
        },
        "model": {
            "config.json sha256": sha256_file(os.path.join(args.model_dir, "config.json")),
        },
        "prompt": PROMPT,
        "tokens": ids[0].tolist(),
        "decode_token": next_id,
        "data": {"file": "golden.safetensors", "sha256": sha256_file(data_path), "tensors": len(R.tensors)},
        "records": R.records,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    ops = {}
    for r in R.records:
        ops[r["op"]] = ops.get(r["op"], 0) + 1
    size = os.path.getsize(data_path)
    print(f"dump.py: {T} prompt tokens + 1 decode token, {len(R.records)} records, {len(R.tensors)} tensors, "
          f"{size / 2**20:.1f} MiB in {args.out}")
    print("         " + ", ".join(f"{k} {v}" for k, v in sorted(ops.items())))


if __name__ == "__main__":
    main()
