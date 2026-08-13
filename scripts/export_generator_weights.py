#!/usr/bin/env python3
"""Export a trained generator's weights to the flat blob tig-challenges embeds.

Every verifier must load byte-identical weights, so the blob is committed to git
rather than downloaded. The format is self-describing (per-layer dims in the
header) so a second generator architecture can be added without changing the
Rust parser.

Row-major `w[out_dim][in_dim]` matches PyTorch's `nn.Linear.weight` exactly, so
the CUDA kernel can index `weight + col * in_dim` with no transpose.
"""

import argparse
import hashlib
import struct
import sys
from pathlib import Path

import torch

MAGIC = b"TIGGAN01"


def export(checkpoint: Path, state_key: str, out: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state_key not in ckpt:
        raise SystemExit(
            f"{checkpoint} has no key {state_key!r}; found {sorted(ckpt)}"
        )
    state = ckpt[state_key]

    # nn.Sequential names them net.0, net.2, net.4, net.6 (odd indices are the
    # activations, which have no parameters). Sorting numerically keeps layer
    # order correct beyond net.9.
    indices = sorted(
        {int(k.split(".")[1]) for k in state if k.startswith("net.")}
    )
    if not indices:
        raise SystemExit(f"no 'net.N.*' parameters in {state_key}")

    blob = bytearray(MAGIC)
    blob += struct.pack("<I", len(indices))
    for i in indices:
        w = state[f"net.{i}.weight"]
        b = state[f"net.{i}.bias"]
        out_dim, in_dim = w.shape
        if b.shape[0] != out_dim:
            raise SystemExit(f"layer {i}: bias {b.shape} does not match {w.shape}")
        blob += struct.pack("<II", in_dim, out_dim)
        blob += w.to(torch.float32).contiguous().numpy().tobytes()
        blob += b.to(torch.float32).contiguous().numpy().tobytes()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)

    print(f"checkpoint : {checkpoint}")
    print(f"state key  : {state_key}  (step {ckpt.get('step')})")
    print(f"layers     : {[(state[f'net.{i}.weight'].shape[1], state[f'net.{i}.weight'].shape[0]) for i in indices]}")
    print(f"bytes      : {len(blob)}")
    print(f"sha256     : {hashlib.sha256(blob).hexdigest()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--state-key", default="generator_state_dict")
    args = p.parse_args()
    export(args.checkpoint, args.state_key, args.output)


if __name__ == "__main__":
    sys.exit(main())
