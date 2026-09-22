#!/usr/bin/env python3
"""Export a trained generator's weights to the flat blob tig-challenges embeds.

Every verifier must load byte-identical weights, so the blob is committed to git
rather than downloaded. The TIGGAN02 container is self-describing about SHAPES
(every tensor carries its own rows and cols), so the Rust side never has to
guess a dimension. It is not self-describing about MEANING: the tensor order is
positional and fixed per `arch`, so adding a fourth architecture means a new
`from_container` arm in `gan_generator` and a new device driver in
`vector_search/generator.rs` as well as a new `--arch` here. What the format
buys is that neither side has to hard-code a layer count or a width.
"""

import argparse
import hashlib
import math
import struct
import sys
from pathlib import Path

import torch

MAGIC_V2 = b"TIGGAN02"
ARCH_IDS = {"mlp": 0, "structured_gate": 1, "spherical": 2}
NORMALIZE_EPS = 1.0e-8  # src/sample/generate.py:69-71 in the WGAN repo


def _tensor(t: torch.Tensor) -> bytes:
    t = t.detach().to(torch.float32).contiguous()
    if t.dim() == 1:
        t = t.reshape(-1, 1)
    rows, cols = t.shape
    return struct.pack("<II", rows, cols) + t.numpy().tobytes()


def _linears(seq) -> list:
    return [m for m in seq if isinstance(m, torch.nn.Linear)]


def _dense(layer: torch.nn.Linear) -> list:
    if layer.bias is None:
        raise SystemExit("expected a bias on this layer; the blob format stores one")
    return [layer.weight, layer.bias]


def _bake(fn, dim: int) -> torch.Tensor:
    """Matrix M with fn(x) == x @ M.T for the linear map `fn`, recovered by
    pushing the one-hot basis through the module's own code."""
    with torch.no_grad():
        return fn(torch.eye(dim)).T.contiguous()


def encode(model: torch.nn.Module, arch: str) -> bytes:
    name = type(model).__name__
    expected = {"mlp": "Generator", "structured_gate": "StructuredGateGenerator",
                "spherical": "SphericalGenerator"}[arch]
    if name != expected:
        raise SystemExit(f"--arch {arch} needs a {expected}, but the config builds a {name}")

    if arch == "mlp":
        layers = _linears(model.net)
        latent, out = layers[0].in_features, layers[-1].out_features
        scalars = [NORMALIZE_EPS]
        tensors = [t for layer in layers for t in _dense(layer)]
    elif arch == "structured_gate":
        trunk = _linears(model.trunk)
        latent, out = trunk[0].in_features, model.magnitude_head.out_features
        scalars = [model.logit_clamp, model.eps * 100.0, model.eps]
        tensors = [t for layer in trunk for t in _dense(layer)]
        tensors += _dense(model.magnitude_head) + _dense(model.gate_head) + _dense(model.sparsity_head)
        tensors += [_bake(model._couple, out), _bake(model._smooth_noise, out)]
    else:
        trunk = _linears(model.trunk)
        latent = trunk[0].in_features + model.tangent_in.in_features
        out = model.direction.out_features
        r = float(model.radius.detach().to(torch.float64))
        scalars = [math.cos(r), math.sin(r), model.eps]
        tensors = [t for layer in trunk for t in _dense(layer)]
        tensors += [model.direction.weight]
        for layer in (model.tangent_in, model.gamma, model.beta, model.tangent_out):
            tensors += _dense(layer)

    blob = bytearray(MAGIC_V2)
    blob += struct.pack("<III", ARCH_IDS[arch], latent, out)
    blob += struct.pack("<I", len(scalars)) + struct.pack(f"<{len(scalars)}f", *scalars)
    blob += struct.pack("<I", len(tensors))
    for t in tensors:
        blob += _tensor(t)
    return bytes(blob)


def load_model(checkpoint: Path, run_config: Path, wgan_repo: Path) -> torch.nn.Module:
    import yaml
    sys.path.insert(0, str(wgan_repo))
    from src.models.generator import build_generator
    config = yaml.safe_load(run_config.read_text())
    model = build_generator(config["model"], output_dim=int(config["data"]["descriptor_dim"]))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["generator_state_dict"])
    return model.eval()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--arch", required=True, choices=sorted(ARCH_IDS))
    p.add_argument("--run-config", required=True, type=Path)
    p.add_argument("--wgan-repo", required=True, type=Path)
    args = p.parse_args()
    blob = encode(load_model(args.checkpoint, args.run_config, args.wgan_repo), args.arch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(blob)
    print(f"checkpoint sha256 : {hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}")
    print(f"blob bytes        : {len(blob)}")
    print(f"blob sha256       : {hashlib.sha256(blob).hexdigest()}")


if __name__ == "__main__":
    sys.exit(main())
