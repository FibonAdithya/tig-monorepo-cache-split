"""Tests for the TIGGAN02 exporter. Run with the WGAN venv:
    ~/TIG/wgan-synthetic/.venv/bin/python -m pytest scripts/tests/test_export_generator.py -v
"""
import struct
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
WGAN = Path.home() / "TIG" / "wgan-synthetic"
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(WGAN))

import export_generator_weights as ex  # noqa: E402
import dump_golden_vectors as gold  # noqa: E402
from src.models.generator import (  # noqa: E402
    Generator, SphericalGenerator, StructuredGateGenerator,
)


def read_blob(blob: bytes):
    assert blob[:8] == b"TIGGAN02"
    arch, latent, out, n_scalars = struct.unpack_from("<IIII", blob, 8)
    at = 24
    scalars = list(struct.unpack_from(f"<{n_scalars}f", blob, at))
    at += 4 * n_scalars
    (n_tensors,) = struct.unpack_from("<I", blob, at)
    at += 4
    tensors = []
    for _ in range(n_tensors):
        rows, cols = struct.unpack_from("<II", blob, at)
        at += 8
        data = torch.frombuffer(bytearray(blob[at:at + 4 * rows * cols]), dtype=torch.float32)
        tensors.append(data.reshape(rows, cols).clone())
        at += 4 * rows * cols
    assert at == len(blob), "trailing bytes"
    return arch, latent, out, scalars, tensors


def small_gate():
    torch.manual_seed(0)
    g = StructuredGateGenerator(latent_dim=16, output_dim=128, hidden_dims=[32, 32],
                                logit_clamp=4.0, layout=(4, 4, 8), gate_kernel=3,
                                noise_kernel_sigma=0.65)
    # Identity init would make a transposed bake invisible: perturb it.
    with torch.no_grad():
        g.gate_coupling.weight.add_(0.1 * torch.randn_like(g.gate_coupling.weight))
    return g.eval()


def test_mlp_blob_layout():
    torch.manual_seed(0)
    m = Generator(latent_dim=16, output_dim=10, hidden_dims=[32, 24]).eval()
    arch, latent, out, scalars, tensors = read_blob(ex.encode(m, "mlp"))
    assert (arch, latent, out) == (0, 16, 10)
    assert scalars == pytest.approx([1.0e-8])
    assert [tuple(t.shape) for t in tensors] == [(32, 16), (32, 1), (24, 32), (24, 1), (10, 24), (10, 1)]
    assert torch.equal(tensors[0], m.net[0].weight.detach())


def test_baked_coupling_equals_the_modules_conv():
    g = small_gate()
    _, _, _, _, tensors = read_blob(ex.encode(g, "structured_gate"))
    coupling = tensors[-2]
    torch.manual_seed(0)
    x = torch.randn(64, 128)
    with torch.no_grad():
        assert torch.allclose(x @ coupling.T, g._couple(x), atol=1e-6)


def test_baked_smoothing_equals_the_modules_smoothing():
    g = small_gate()
    _, _, _, _, tensors = read_blob(ex.encode(g, "structured_gate"))
    smoothing = tensors[-1]
    torch.manual_seed(0)
    x = torch.randn(64, 128)
    with torch.no_grad():
        assert torch.allclose(x @ smoothing.T, g._smooth_noise(x), atol=1e-6)


def test_pinned_noise_forward_equals_the_module_with_patched_rand(monkeypatch):
    """The golden dump thresholds `logit + noise > 0` instead of calling
    `_sample_gate`. This proves the two agree when the module's uniform draw
    is pinned to the same u."""
    g = small_gate()
    torch.manual_seed(1)
    z = torch.randn(32, 16)
    u = torch.rand(32, 128).clamp(g.eps, 1.0 - g.eps)
    monkeypatch.setattr(torch, "rand_like", lambda t: u.to(t.dtype))
    with torch.no_grad():
        expected = g(z)
        got = gold.structured_gate_forward(g, z, torch.log(u) - torch.log1p(-u))
    assert torch.equal(got != 0, expected != 0), "support pattern differs"
    assert torch.allclose(got, expected, atol=1e-6)


def test_spherical_scalars_are_cos_and_sin_of_the_learned_radius():
    torch.manual_seed(0)
    s = SphericalGenerator(latent_dim=24, output_dim=8, hidden_dims=[16, 16],
                           skip_dim=8, tangent_hidden_dim=12).eval()
    arch, latent, out, scalars, tensors = read_blob(ex.encode(s, "spherical"))
    assert (arch, latent, out) == (2, 24, 8)
    r = float(s.radius)
    assert scalars[0] == pytest.approx(torch.cos(torch.tensor(r, dtype=torch.float64)).item(), abs=1e-7)
    assert scalars[1] == pytest.approx(torch.sin(torch.tensor(r, dtype=torch.float64)).item(), abs=1e-7)
    assert len(tensors) == 2 * 2 + 9
    assert tuple(tensors[4].shape) == (8, 16), "direction is [output_dim][hidden]"


def test_encode_rejects_a_module_of_the_wrong_class():
    m = Generator(latent_dim=16, output_dim=10, hidden_dims=[32]).eval()
    with pytest.raises(SystemExit):
        ex.encode(m, "spherical")
