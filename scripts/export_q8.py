#!/usr/bin/env python3
"""Quantize a legacy (v0) llama2.c fp32 checkpoint to the Q8_0 v2 format.

The firmware currently loads karpathy's legacy fp32 checkpoint (a 28-byte
Config header followed by raw fp32 weights). This tool reads that file and
re-emits it in the "version 2" group-wise int8 (Q8_0) format that
karpathy's runq.c reads, so weights shrink ~2-4x (depending on group size)
while staying numerically faithful.

Format reference: https://github.com/karpathy/llama2.c/blob/master/export.py
                  (version2_export) and runq.c (read_checkpoint).

Why not torch: we already have an exported fp32 .bin, so we work directly on
the raw floats with numpy — no PyTorch, no model class needed.

Usage:
    python scripts/export_q8.py data/stories260K.bin data/stories260K_q8.bin
    python scripts/export_q8.py in.bin out.bin --group-size 64
"""
import argparse
import struct
import sys
from math import gcd

import numpy as np

MAGIC = 0x616B3432  # "ak42"
VERSION = 2
HEADER_SIZE = 256


def read_v0_checkpoint(path):
    """Parse a legacy fp32 checkpoint into (config dict, weights dict).

    Layout mirrors memory_map_weights() in main/llm.c. A positive vocab_size
    signals shared (tied) classifier/embedding weights.
    """
    with open(path, "rb") as f:
        blob = f.read()

    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack(
        "iiiiiii", blob[:28]
    )
    shared_weights = vocab_size > 0
    vocab_size = abs(vocab_size)
    head_size = dim // n_heads

    cfg = dict(
        dim=dim, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
        n_kv_heads=n_kv_heads, vocab_size=vocab_size, seq_len=seq_len,
        shared_weights=shared_weights,
    )

    floats = np.frombuffer(blob, dtype=np.float32, offset=28)
    off = 0

    def take(n):
        nonlocal off
        chunk = floats[off:off + n]
        if chunk.size != n:
            sys.exit(f"checkpoint truncated: wanted {n} floats at {off}, got {chunk.size}")
        off += n
        return chunk

    w = {}
    w["token_embedding_table"] = take(vocab_size * dim)
    w["rms_att_weight"] = take(n_layers * dim)
    w["wq"] = take(n_layers * dim * (n_heads * head_size))
    w["wk"] = take(n_layers * dim * (n_kv_heads * head_size))
    w["wv"] = take(n_layers * dim * (n_kv_heads * head_size))
    w["wo"] = take(n_layers * (n_heads * head_size) * dim)
    w["rms_ffn_weight"] = take(n_layers * dim)
    w["w1"] = take(n_layers * dim * hidden_dim)
    w["w2"] = take(n_layers * hidden_dim * dim)
    w["w3"] = take(n_layers * dim * hidden_dim)
    w["rms_final_weight"] = take(dim)
    take(seq_len * head_size // 2)  # freq_cis_real, unused (RoPE computed live)
    take(seq_len * head_size // 2)  # freq_cis_imag, unused
    if not shared_weights:
        w["wcls"] = take(vocab_size * dim)
    return cfg, w


def auto_group_size(cfg, requested):
    """Largest group size <= requested that divides every dimension the
    runtime quantizes or contracts over.

    The runtime calls quantize() on activations of length dim and hidden_dim,
    and matmul contracts over dim, kv_dim and hidden_dim. group_size must
    divide all of them (and every quantized weight's numel, which is implied
    since each is a product involving these dims).
    """
    dim = cfg["dim"]
    kv_dim = (cfg["dim"] * cfg["n_kv_heads"]) // cfg["n_heads"]
    g = gcd(gcd(dim, cfg["hidden_dim"]), kv_dim)
    gs = gcd(g, requested) if requested else g
    # prefer the requested size when it already divides everything
    if requested and g % requested == 0:
        gs = requested
    return gs


def quantize_q80(w, group_size):
    """Symmetric group-wise int8 quantization, range [-127, 127].

    Returns (int8 values flattened, fp32 scales flattened, max abs error).
    Matches quantize_q80() in karpathy's export.py.
    """
    assert w.size % group_size == 0
    g = w.reshape(-1, group_size).astype(np.float32)
    wmax = np.abs(g).max(axis=1)
    scale = wmax / 127.0
    # guard all-zero groups: scale 0 -> emit zeros, keep scale 0 (dequant gives 0)
    safe = np.where(scale == 0.0, 1.0, scale)
    q = np.round(g / safe[:, None]).astype(np.int8)
    q[np.broadcast_to((scale == 0.0)[:, None], q.shape)] = 0
    deq = q.astype(np.float32) * scale[:, None]
    err = float(np.abs(deq - g).max())
    return q.reshape(-1), scale.astype(np.float32), err


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="legacy fp32 checkpoint (e.g. data/stories260K.bin)")
    ap.add_argument("output", help="output Q8_0 v2 checkpoint")
    ap.add_argument("--group-size", type=int, default=64,
                    help="requested group size; auto-reduced to divide all dims (default 64)")
    args = ap.parse_args()

    cfg, w = read_v0_checkpoint(args.input)
    gs = auto_group_size(cfg, args.group_size)
    if gs != args.group_size:
        print(f"group size {args.group_size} does not divide all dims; using {gs}")
    if gs < 1:
        sys.exit("could not find a valid group size")

    # Build the flat list of individual tensors in the exact order (and
    # per-layer granularity) that runq.c's init_quantized_tensors reads them.
    # Each *layer* of a weight is its own quantized tensor (int8 block then
    # scales), so multi-layer weights must be split and interleaved per layer
    # rather than quantized as one blob.
    n_layers = cfg["n_layers"]
    q_tensors = [("token_embedding_table", w["token_embedding_table"])]
    for name in ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]:
        per_layer = w[name].reshape(n_layers, -1)
        for l in range(n_layers):
            q_tensors.append((f"{name}[{l}]", per_layer[l]))
    if not cfg["shared_weights"]:
        q_tensors.append(("wcls", w["wcls"]))

    with open(args.output, "wb") as f:
        f.write(struct.pack("I", MAGIC))
        f.write(struct.pack("i", VERSION))
        f.write(struct.pack("iiiiiii", cfg["dim"], cfg["hidden_dim"], cfg["n_layers"],
                            cfg["n_heads"], cfg["n_kv_heads"], cfg["vocab_size"],
                            cfg["seq_len"]))
        f.write(struct.pack("B", int(cfg["shared_weights"])))
        f.write(struct.pack("i", gs))
        pad = HEADER_SIZE - f.tell()
        assert pad >= 0, "header overflow"
        f.write(b"\0" * pad)

        # fp32 norms first: att (all layers), ffn (all layers), final
        f.write(w["rms_att_weight"].astype(np.float32).tobytes())
        f.write(w["rms_ffn_weight"].astype(np.float32).tobytes())
        f.write(w["rms_final_weight"].astype(np.float32).tobytes())

        # then the Q8_0 tensors: int8 block followed by fp32 scales, one
        # (int8, scales) pair per tensor in q_tensors order
        max_err = 0.0
        for name, tensor in q_tensors:
            q, s, err = quantize_q80(tensor, gs)
            f.write(q.tobytes())
            f.write(s.tobytes())
            max_err = max(max_err, err)
            print(f"quantized {name:22s} numel={tensor.size:>8d} max_err={err:.6f}")

    print(f"group_size={gs}  max quantization error={max_err:.6f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
