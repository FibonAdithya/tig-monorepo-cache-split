#!/usr/bin/env python3
"""Dump PyTorch generator outputs for fixed inputs, as a port-correctness oracle.

Latents are a deterministic ramp, not random draws: the point is to pin the
arithmetic, and a fixed pattern needs no RNG agreement between Python and Rust.
For structured_gate the gate's logistic noise is pinned too, because a
stochastic gate cannot be compared otherwise.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_generator_weights import NORMALIZE_EPS, load_model  # noqa: E402


def ramp(count: int, dim: int, scale: float) -> torch.Tensor:
    """Distinct per row and per column, in roughly [-scale, scale)."""
    cols = torch.arange(dim, dtype=torch.float32) / (dim / 2.0) - 1.0
    return torch.stack([(cols + i * 0.01) * scale for i in range(count)])


def structured_gate_forward(g, z: torch.Tensor, gate_noise: torch.Tensor) -> torch.Tensor:
    """StructuredGateGenerator.forward with the logistic noise supplied.
    `hard = sigmoid((logits + n) / T) > 0.5` is `logits + n > 0` for T > 0."""
    h = g.trunk(z)
    magnitude = F.softplus(g.magnitude_head(h)).clamp(min=g.eps * 100.0)
    logits = g._gate_logits(h)
    hard = ((logits + g._smooth_noise(gate_noise)) > 0).to(logits.dtype)
    empty = hard.sum(dim=1, keepdim=True) == 0
    fallback = F.one_hot(logits.argmax(dim=1), logits.shape[1]).to(hard.dtype)
    x = torch.where(empty, fallback, hard) * magnitude
    return x / torch.clamp(torch.linalg.vector_norm(x, dim=1, keepdim=True), min=g.eps)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--arch", required=True, choices=["mlp", "structured_gate", "spherical"])
    p.add_argument("--run-config", required=True, type=Path)
    p.add_argument("--wgan-repo", required=True, type=Path)
    p.add_argument("--count", type=int, default=8)
    args = p.parse_args()

    model = load_model(args.checkpoint, args.run_config, args.wgan_repo)
    latent_dim = {"mlp": lambda m: m.net[0].in_features,
                  "structured_gate": lambda m: m.trunk[0].in_features,
                  "spherical": lambda m: m.trunk[0].in_features + m.tangent_in.in_features}[args.arch](model)
    latents = ramp(args.count, latent_dim, 1.0)
    gate_noise = None
    with torch.no_grad():
        if args.arch == "structured_gate":
            out_dim = model.magnitude_head.out_features
            # Logistic noise spans about +/-7 over the clamp range; +/-3 keeps
            # a mix of open and closed gates in every row.
            gate_noise = ramp(args.count, out_dim, 3.0).flip(1)
            out = structured_gate_forward(model, latents, gate_noise)
        else:
            out = model(latents)
            if args.arch == "mlp":
                out = out / torch.clamp(torch.linalg.vector_norm(out, dim=1, keepdim=True), min=NORMALIZE_EPS)

    args.output.write_text(json.dumps({
        "latents": latents.tolist(),
        "gate_noise": None if gate_noise is None else gate_noise.tolist(),
        "outputs": out.tolist(),
    }))
    print(f"wrote {args.count} golden vectors to {args.output}")


if __name__ == "__main__":
    main()
