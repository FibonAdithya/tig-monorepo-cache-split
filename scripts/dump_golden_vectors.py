#!/usr/bin/env python3
"""Dump PyTorch generator outputs for fixed latents, as a port-correctness oracle.

Latents are a deterministic ramp rather than random draws: the point is to pin
the arithmetic of the forward pass, and a fixed pattern is reproducible without
agreeing on an RNG between Python and Rust.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/checkouts/wgan-synthetic")
from src.models.generator import Generator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--count", type=int, default=4)
    args = p.parse_args()

    model = Generator(latent_dim=128, output_dim=128,
                      hidden_dims=[512, 1024, 1024], negative_slope=0.2)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["generator_state_dict"])
    model.eval()

    # Values in [-1, 1); distinct per row and per column.
    latents = torch.stack([
        torch.arange(128, dtype=torch.float32) / 64.0 - 1.0 + i * 0.01
        for i in range(args.count)
    ])

    with torch.no_grad():
        out = model(latents)

    args.output.write_text(json.dumps({
        "latents": latents.tolist(),
        "outputs": out.tolist(),
    }, indent=2))
    print(f"wrote {args.count} golden vectors to {args.output}")


if __name__ == "__main__":
    main()
