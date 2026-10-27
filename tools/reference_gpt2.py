"""An independent GPT-2 forward pass in numpy, used only to check the engine.

It shares no code with src/: it reads the safetensors file itself and follows
the published architecture (pre-LayerNorm blocks, Conv1D weights stored
[in, out], tanh-approximated GELU, causal attention, tied unembedding).

    python3 tools/reference_gpt2.py weights.safetensors HEADS "1,2,3" out.json
"""

import json
import struct
import sys

import numpy as np


def load(path):
    with open(path, "rb") as handle:
        blob = handle.read()
    (length,) = struct.unpack("<Q", blob[:8])
    header = json.loads(blob[8 : 8 + length])
    base = 8 + length
    out = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        begin, end = entry["data_offsets"]
        assert entry["dtype"] == "F32", entry["dtype"]
        name = name[len("transformer.") :] if name.startswith("transformer.") else name
        out[name] = np.frombuffer(blob, dtype="<f4", count=(end - begin) // 4, offset=base + begin).reshape(entry["shape"]).astype(np.float64)
    return out


def layer_norm(x, gain, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gain + bias


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def forward(w, heads, ids):
    T = len(ids)
    d = w["wte.weight"].shape[1]
    x = w["wte.weight"][ids] + w["wpe.weight"][:T]
    trace = {"resid.0": x.copy()}
    causal = np.tril(np.ones((T, T), dtype=bool))
    layer = 0
    while f"h.{layer}.ln_1.weight" in w:
        p = f"h.{layer}"
        a = layer_norm(x, w[p + ".ln_1.weight"], w[p + ".ln_1.bias"])
        qkv = a @ w[p + ".attn.c_attn.weight"] + w[p + ".attn.c_attn.bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        split = lambda t: t.reshape(T, heads, d // heads).transpose(1, 0, 2)  # [H, T, hd]
        q, k, v = split(q), split(k), split(v)
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(d // heads)
        scores = np.where(causal, scores, -np.inf)
        scores = scores - scores.max(axis=-1, keepdims=True)
        pattern = np.exp(scores)
        pattern = pattern / pattern.sum(axis=-1, keepdims=True)  # [H, T, T]
        mixed = (pattern @ v).transpose(1, 0, 2).reshape(T, d)
        attn_out = mixed @ w[p + ".attn.c_proj.weight"] + w[p + ".attn.c_proj.bias"]
        x = x + attn_out
        hidden = gelu(layer_norm(x, w[p + ".ln_2.weight"], w[p + ".ln_2.bias"]) @ w[p + ".mlp.c_fc.weight"] + w[p + ".mlp.c_fc.bias"])
        mlp_out = hidden @ w[p + ".mlp.c_proj.weight"] + w[p + ".mlp.c_proj.bias"]
        x = x + mlp_out
        trace[f"block.{layer}.attn.pattern"] = pattern.transpose(1, 0, 2).copy()  # [T(query), H, T(key)]
        trace[f"block.{layer}.attn.out"] = attn_out
        trace[f"block.{layer}.mlp.post"] = hidden
        trace[f"block.{layer}.mlp.out"] = mlp_out
        trace[f"block.{layer}.resid"] = x.copy()
        layer += 1
    final = layer_norm(x, w["ln_f.weight"], w["ln_f.bias"])
    trace["final.resid"] = final
    trace["logits"] = final @ w["wte.weight"].T
    return trace


if __name__ == "__main__":
    path, heads, ids, out = sys.argv[1], int(sys.argv[2]), [int(t) for t in sys.argv[3].split(",")], sys.argv[4]
    trace = forward(load(path), heads, ids)
    with open(out, "w") as handle:
        json.dump({key: value.astype(np.float64).ravel().tolist() for key, value in trace.items()}, handle)
    print("reference:", ", ".join(f"{k}{list(v.shape)}" for k, v in list(trace.items())[:3]), "...")
