# Multi-architecture GAN scenarios Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `vector_search` generate instances from the three gate-accepted WGAN generators: SIFT v4 (`structured_gate`), NYTimes v3 (`spherical`) and GloVe v1 (`mlp`), as tracks `s=sift_128`, `s=nytimes_256`, `s=glove_100`.

**Architecture:** A `TIGGAN02` blob format carries an architecture tag, scalars and positional tensors. A CPU-only, feature-ungated crate module `gan_generator` parses blobs and holds one reference forward pass per architecture, so it builds and tests on a machine with no CUDA. Inside `vector_search`, one GPU driver per architecture composes the unchanged `gan_linear` with five new row-wise kernels.

**Tech Stack:** Rust `nightly-2025-02-10`, `cudarc`, CUDA PTX kernels (`nvcc -ptx`, dialect: `__global__`, `__shared__`, `__syncthreads()`), Python 3 + PyTorch 2.13 (exporter, goldens), `gpuq` job queue on `tig-gpu`.

**Spec:** `docs/ai/specs/2026-09-21-multi-arch-gan-scenarios-design.md`. Read it first; this plan argues from it.

## Global Constraints

- Track wire strings are exactly `s=sift_128`, `s=nytimes_256`, `s=glove_100`.
- All three scenarios: `n_queries = 7_000`, `database_size = 700_000`, `min_recall = 0.9`, `recall_tolerance = 1e-6`, `audit_samples = 1_000`.
- `gan_sample_latents`, `gan_linear` and `evaluate_total_distance` in `kernels.cu` must not change, byte for byte. Check with `git diff` on those line ranges before every commit that touches `kernels.cu`.
- `Challenge`'s field layout must not change (every algorithm `.so` reads it by offset).
- LeakyReLU slope is `0.2` everywhere. Normalisation `eps = 1e-8`. Softplus uses PyTorch's threshold: `x > 20 ? x : log1p(exp(x))`.
- Dense accumulation order is fixed: start from the bias, then `acc = x[k].mul_add(w[k], acc)` for `k = 0..in_dim`, in that order. CPU references must use `mul_add`, because the kernel uses `fmaf`.
- Checkpoints are never committed. Blobs are committed and exceed 1 MB, so those commits need both bypasses: append `# allow-large-commit` to the command and prefix `ALLOW_LARGE_COMMIT=1`.
- Never `git add -A`, `git add .` or `git commit -a`. Stage explicit paths; run `git status --short` before every commit.
- Every number written to a doc or commit message is labelled MEASURED (with the command) or ESTIMATE (unverified).
- No wall-clock-derived or unseeded randomness in any test.

## Environment facts (all MEASURED 2026-09-21; do not rediscover)

| fact | how it was established |
|---|---|
| This machine has no `nvcc` and no GPU. `cargo test -p tig-challenges --features vector_search` fails in cudarc's build script: ``Failed to execute `nvcc` ``. | ran it |
| Crate-level modules outside the `c004` gate DO test locally: `cargo test -p tig-challenges audit_sampling` → 9 passed. | ran it |
| Local PyTorch: `~/TIG/wgan-synthetic/.venv/bin/python`, torch 2.13.0, CPU only. | `python -c "import torch"` |
| `tig-gpu` is an RTX 3060 Ti, compute capability 8.6, 8192 MiB, CUDA 12.8 at `/usr/local/cuda`. | `nvidia-smi`, `nvcc --version` over SSH |
| `tig-gpu` has **no Rust toolchain** (`/root/.cargo/bin` does not exist) and **no checkout of this repo**. `/workspace/checkouts` holds only `wgan-synthetic`. | `ls` over SSH |
| `gpuq` and `gpu-claim` are at `/usr/local/bin`. Projects are registered in `/workspace/gpuq.toml` as `[project.<name>]` blocks with `remote`, `checkout`, `venv`, `commit_artifacts`. | `cat` over SSH |
| This branch tracks `origin/vector_search/gan_instance_gen` on `git@github.com:FibonAdithya/tig-monorepo-cache-split.git`. | `git remote -v`, `git status -sb` |
| The earlier handoff says cudarc rejected CUDA 13.0 and was validated on 12.6. 12.8 is **untested**; Task 0 finds out. | `docs/ai/plans/2026-08-14-handoff.md` |
| The three accepted checkpoints, sizes and sha256 are in the spec's Scenarios section. `v3_seed42/`, `v3_seed42_100k/` and `live/nytimes/v3_seed42/` hold a byte-identical `best_generator.pt` (`b1acfdda…`), so `v3_best` is unambiguous. | `sha256sum` over SSH |
| WGAN checkpoints carry a `generator_weights` field, `"live"` or `"ema"`, saying what `generator_state_dict` holds (`src/train/train_wgan_gp.py:375`). | read the source |

## File structure

```
scripts/
  box_test.sh                     NEW  runs on tig-gpu inside a gpuq job: PATH setup + cargo test + log
  box_submit.sh                   NEW  runs here: checks HEAD is pushed, submits, waits, prints result
  export_generator_weights.py     MOD  TIGGAN02 writer, --arch, --run-config, --wgan-repo, baking
  dump_golden_vectors.py          MOD  per-architecture goldens, pinned gate noise
  tests/test_export_generator.py  NEW  pytest: blob layout, baked maps, pinned-noise equivalence
tig-challenges/src/
  lib.rs                          MOD  `pub mod gan_generator;` (ungated, next to audit_sampling)
  gan_generator/
    mod.rs                        NEW  Generator enum, from_blob, dense_cpu, forward_cpu dispatch
    v1.rs                         MOVED from vector_search/generator.rs: TIGGAN01 parser, Layer, tests
    blob.rs                       NEW  TIGGAN02 container parser (header, scalars, tensors)
    mlp.rs                        NEW  Mlp: from_parts checks + forward_cpu
    spherical.rs                  NEW  Spherical: same
    structured_gate.rs            NEW  StructuredGate: same
  vector_search/
    generator.rs                  REWRITTEN  GPU side only: DeviceLayer, launch_linear, DeviceGenerator
    scenarios.rs                  MOD  three variants, Scenario::ALL, blob consts
    mod.rs                        MOD  generate_vectors delegates; AUDIT_MAX_DIMS; new GPU tests
    kernels.cu                    MOD  five new kernels appended; AUDIT_MAX_DIMS
    weights/
      glove_100_v1.bin  nytimes_256_v3.bin  sift_128_v4.bin          NEW blobs
      glove_100_v1.golden.json  nytimes_256_v3.golden.json  sift_128_v4.golden.json   NEW
      PROVENANCE.md                NEW
      v1_sift.bin  golden_vectors.json                               KEPT as the TIGGAN01 fixture
```

Shared vocabulary used by every task (exact names; do not vary them):

```rust
// gan_generator/v1.rs (existing type, moved)
pub struct Layer { pub in_dim: usize, pub out_dim: usize, pub weights: Vec<f32>, pub bias: Vec<f32> }

// gan_generator/mod.rs
pub enum Generator { Mlp(Mlp), StructuredGate(StructuredGate), Spherical(Spherical) }
impl Generator {
    pub fn from_blob(blob: &[u8]) -> anyhow::Result<Self>;   // dispatches on TIGGAN01 / TIGGAN02
    pub fn latent_dim(&self) -> usize;
    pub fn output_dim(&self) -> usize;
    /// `gate_noise` is the pre-smoothing logistic noise, one value per output
    /// coordinate. Required for StructuredGate, must be None for the others.
    pub fn forward_cpu(&self, latent: &[f32], gate_noise: Option<&[f32]>) -> anyhow::Result<Vec<f32>>;
}
pub fn dense_cpu(layer: &Layer, input: &[f32], activate: bool) -> Vec<f32>;

pub struct Mlp { pub layers: Vec<Layer>, pub normalize_eps: Option<f32> }
pub struct Spherical { pub trunk: Vec<Layer>, pub direction: Layer, pub tangent_in: Layer,
    pub gamma: Layer, pub beta: Layer, pub tangent_out: Layer, pub cos_r: f32, pub sin_r: f32, pub eps: f32 }
pub struct StructuredGate { pub trunk: Vec<Layer>, pub magnitude_head: Layer, pub gate_head: Layer,
    pub sparsity_head: Layer, pub coupling: Layer, pub smoothing: Layer,
    pub logit_clamp: f32, pub magnitude_floor: f32, pub eps: f32 }
```

The spec left open whether the v1 fixture should stay embedded in every algorithm `.so`. This structure settles it: after Task 9 `v1_sift.bin` is included only under `#[cfg(test)]` in `gan_generator::v1`, so release builds embed the three scenario blobs and nothing else.

A bias-free map (`direction`, `coupling`, `smoothing`) is a `Layer` whose `bias` is all zeros, built by the parser. That lets every matrix product go through `gan_linear` and `dense_cpu` unchanged.

`TIGGAN02` tensor and scalar order (the exporter writes this, the parser checks it):

| arch | id | scalars, in order | tensors, in order | tensor count |
|---|---|---|---|---|
| mlp | 0 | `eps` | `W_0, b_0, …, W_{L-1}, b_{L-1}` | `2L`, `L ≥ 1` |
| structured_gate | 1 | `logit_clamp, magnitude_floor, eps` | trunk `W,b × T`; `magnitude_head W,b`; `gate_head W,b`; `sparsity_head W,b`; `coupling`; `smoothing` | `2T + 8`, `T ≥ 1` |
| spherical | 2 | `cos_r, sin_r, eps` | trunk `W,b × T`; `direction W`; `tangent_in W,b`; `gamma W,b`; `beta W,b`; `tangent_out W,b` | `2T + 9`, `T ≥ 1` |

A weight tensor is `rows = out_dim`, `cols = in_dim`. A bias tensor is `rows = out_dim`, `cols = 1`.

---

### Task 0: Box build environment and a green baseline

Nothing in this plan can be tested on the GPU until the box can build this repo. This task changes shared box configuration, so **confirm with the owner before Step 2** if they have not already approved it.

**Files:**
- Create: `scripts/box_test.sh`, `scripts/box_submit.sh`
- Remote: `/workspace/gpuq.toml` on `tig-gpu` (add one project block)

**Interfaces:**
- Produces: `scripts/box_submit.sh <log-name> <lane> <cargo-test-filter…>` — used by every later GPU step. Exit code is the job's.

- [ ] **Step 1: Write the two scripts**

`scripts/box_test.sh`:

```bash
#!/usr/bin/env bash
# Runs ON tig-gpu, inside a gpuq job. Usage: box_test.sh <log-name> <cargo test filter and flags...>
set -euo pipefail
name=$1
shift
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
# Absolute and outside the runner's checkout: the runner may clean its tree
# between jobs. Only the #[ignore]d dump test in Task 10 writes here.
export TIG_DUMP_DIR="${TIG_DUMP_DIR:-/workspace/tig-dumps}"
mkdir -p runs/box "$TIG_DUMP_DIR"
# TIG_TEST_EXTRA carries extra libtest flags, e.g. --ignored. Unquoted on
# purpose so that an empty value adds no argument.
# --test-threads=1: the GPU tests share one card and one PTX build.
cargo +nightly-2025-02-10 test -p tig-challenges --features vector_search "$@" \
    -- --test-threads=1 --nocapture ${TIG_TEST_EXTRA:-} 2>&1 | tee "runs/box/$name.log"
```

`scripts/box_submit.sh`:

```bash
#!/usr/bin/env bash
# Runs HERE. Usage: [BOX_ENV="K=V ..."] box_submit.sh <log-name> <gpu|cpu> <cargo test filter...>
# BOX_ENV is passed to the job's environment, e.g. BOX_ENV="TIG_TEST_EXTRA=--ignored".
# Refuses to submit a commit the box cannot fetch.
set -euo pipefail
name=$1
lane=$2
shift 2
sha=$(git rev-parse HEAD)
br=$(git rev-parse --abbrev-ref HEAD)
git fetch -q origin "$br"
if [ "$(git rev-parse "origin/$br")" != "$sha" ]; then
    echo "HEAD $sha is not what origin/$br points at; push first" >&2
    exit 2
fi
id=$(ssh -o BatchMode=yes tig-gpu "gpuq submit --project tig-monorepo --commit $sha \
    --branch '$br' --lane $lane --timeout-s ${BOX_TIMEOUT_S:-5400} -- env ${BOX_ENV:-} bash scripts/box_test.sh $name $*" 2>/dev/null | tail -1)
echo "job: $id"
rc=0
ssh -o BatchMode=yes tig-gpu "gpuq wait $id" 2>/dev/null || rc=$?
ssh -o BatchMode=yes tig-gpu "gpuq show $id" 2>/dev/null
exit $rc
```

`chmod +x` both. Add `runs/` to `.gitignore` if `git check-ignore runs/box/x.log` prints nothing.

- [ ] **Step 2: Install Rust on the box and register the project**

```bash
ssh tig-gpu 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly-2025-02-10 --profile minimal && ~/.cargo/bin/rustc +nightly-2025-02-10 --version'
```

Expected: a `rustc 1.86.0-nightly (… 2025-02-09)` line (the date in the version string is one day before the toolchain name).

Append to `/workspace/gpuq.toml` (keep a backup: `cp /workspace/gpuq.toml /workspace/gpuq.toml.bak-$(date -u +%Y%m%d)`):

```toml
[project.tig-monorepo]
remote   = "git@github.com:FibonAdithya/tig-monorepo-cache-split.git"
checkout = "/workspace/checkouts/tig-monorepo"
venv     = "/venv/main"
commit_artifacts = false
```

Then find and rerun the queue's bootstrap so it clones the project. Locate it with:

```bash
ssh tig-gpu 'ls /workspace/*/bootstrap.sh /opt/*/bootstrap.sh 2>/dev/null; systemctl cat gpuq 2>/dev/null | grep -i -E "ExecStart|WorkingDirectory"'
```

Run the `bootstrap.sh` that command finds, then verify: `ssh tig-gpu 'git -C /workspace/checkouts/tig-monorepo remote -v'` prints the remote above. If the clone fails on authentication, the box's GitHub key does not cover this repo: stop and report that; do not paste credentials.

- [ ] **Step 3: Commit and push the scripts**

```bash
git add scripts/box_test.sh scripts/box_submit.sh .gitignore
git status --short
git commit -m "build(c004): scripts to run the GPU test suite on tig-gpu through gpuq"
timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=8 -T git@github.com; git push origin HEAD
```

- [ ] **Step 4: Baseline. Run the existing suite unchanged and record the result**

```bash
BOX_TIMEOUT_S=14400 scripts/box_submit.sh baseline gpu
```

Run it with a Bash timeout of at least `7200000` ms: the first build on the box compiles every dependency and its duration is unmeasured. Expected: the job exits 0 and `gpuq show` points at a stdout log ending in `test result: ok.`. Record the passed count and the wall time of `audit_is_much_cheaper_than_a_naive_solve` (it prints its ms) in the task notes as MEASURED; Task 7 compares against it.

If the build fails inside `cudarc` on CUDA 12.8, that is the untested-version risk from the facts table. Report the exact error. The fallback is installing the CUDA 12.6 toolkit beside 12.8 and pointing `CUDA_ROOT` at it in `box_test.sh`; do that only with the owner's approval.

**Done when:** the baseline job is green and its test count and audit ms are written down. This task has no mutation check: it adds no test.

---

### Task 1: Exporter, blobs and goldens

All Python, all local. Uses `~/TIG/wgan-synthetic/.venv/bin/python` (called `$PY` below) because it has torch and can import the WGAN model code.

**Files:**
- Modify: `scripts/export_generator_weights.py` (rewrite; keep the `TIGGAN01` path working for the v1 fixture)
- Modify: `scripts/dump_golden_vectors.py` (rewrite)
- Create: `scripts/tests/test_export_generator.py`
- Create: `tig-challenges/src/vector_search/weights/{glove_100_v1,nytimes_256_v3,sift_128_v4}.bin`, the three `.golden.json`, `PROVENANCE.md`

**Interfaces:**
- Produces: the three blobs in the `TIGGAN02` layout tabled above, and goldens shaped
  `{"latents": [[f32]], "gate_noise": [[f32]] | null, "outputs": [[f32]]}` with 8 rows each.

- [ ] **Step 1: Fetch the three checkpoints (standing transfer rule)**

```bash
D=/tmp/claude-ckpt && mkdir -p $D
ssh tig-gpu 'cd /workspace && tar czf - sift-v4/v4_sift1m_x100k/best_generator.pt sift-v4/v4_sift1m_x100k/run_config.yaml nytimes-v3/v3_seed42/best_generator.pt nytimes-v3/v3_seed42/run_config.yaml glove-probes/probe_spectrum_seed42/best_generator.pt glove-probes/probe_spectrum_seed42/run_config.yaml' | tar xzf - -C $D
sha256sum $D/sift-v4/v4_sift1m_x100k/best_generator.pt $D/nytimes-v3/v3_seed42/best_generator.pt $D/glove-probes/probe_spectrum_seed42/best_generator.pt
```

Use the session scratchpad directory for `$D` if one is provided. Expected hashes, in that order: `09eaac8f…79a1e6c`, `b1acfdda…27ee363a`, `38d8013a…4b66d7a3` (full values in the spec). A mismatch means a truncated transfer: re-stream, do not proceed. If a `run_config.yaml` is missing from a run directory, use the WGAN repo's config instead (`configs/sift/v4_sift1m_x100k.yaml`, `configs/nytimes/v3_seed42.yaml`, `configs/glove/v1_seed42.yaml`) and say so in `PROVENANCE.md`.

- [ ] **Step 2: Record what `generator_state_dict` holds**

```bash
PY=~/TIG/wgan-synthetic/.venv/bin/python
for f in $D/sift-v4/v4_sift1m_x100k $D/nytimes-v3/v3_seed42 $D/glove-probes/probe_spectrum_seed42; do
  $PY -c "import torch,sys; c=torch.load(sys.argv[1]+'/best_generator.pt',map_location='cpu',weights_only=False); print(sys.argv[1].split('/')[-1], c.get('generator_weights'), c.get('step'), sorted(k for k in c if 'state' in k))" $f
done
```

Write the three printed lines into `PROVENANCE.md`. Whatever the field says, export `generator_state_dict`: it is the key `src/sample/generate.py:57` loads, so it is what the gates measured. Do not switch keys.

- [ ] **Step 3: Write the failing pytest**

`scripts/tests/test_export_generator.py`:

```python
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
```

What each catches: layout test → wrong tensor order or a transposed weight; the two bake tests → a transposed or interior-only baked map (the perturbation makes the coupling non-symmetric, so a transpose fails); pinned-noise test → a wrong claim that `sigmoid(x/T) > 0.5` equals `x > 0`, or a missing argmax fallback; scalar test → swapped or float32-computed `cos`/`sin`; class test → exporting a checkpoint under the wrong `--arch`.

- [ ] **Step 4: Run it, expect failure**

```bash
$PY -m pytest scripts/tests/test_export_generator.py -v
```

Expected: collection error, `module 'export_generator_weights' has no attribute 'encode'`.

- [ ] **Step 5: Rewrite `scripts/export_generator_weights.py`**

Keep the existing module docstring's first paragraph. Replace the body with:

```python
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
```

The `TIGGAN01` writer is dropped from the script: `v1_sift.bin` already exists and is never regenerated. Say so in the commit message.

- [ ] **Step 6: Rewrite `scripts/dump_golden_vectors.py`**

```python
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
```

- [ ] **Step 7: Run the pytest, expect 6 passed**

```bash
$PY -m pytest scripts/tests/test_export_generator.py -v
```

- [ ] **Step 8: Mutation-check the two tests most likely to be vacuous**

In `_bake`, change `fn(torch.eye(dim)).T` to `fn(torch.eye(dim))` → `test_baked_coupling_equals_the_modules_conv` must FAIL. Restore. In `structured_gate_forward`, change `> 0` to `> 0.5` → `test_pinned_noise_forward…` must FAIL. Restore. Re-run: 6 passed.

- [ ] **Step 9: Export the three blobs and goldens**

```bash
W=tig-challenges/src/vector_search/weights; R=~/TIG/wgan-synthetic
for spec in "glove_100_v1 mlp glove-probes/probe_spectrum_seed42" "nytimes_256_v3 spherical nytimes-v3/v3_seed42" "sift_128_v4 structured_gate sift-v4/v4_sift1m_x100k"; do
  set -- $spec
  $PY scripts/export_generator_weights.py $D/$3/best_generator.pt $W/$1.bin --arch $2 --run-config $D/$3/run_config.yaml --wgan-repo $R
  $PY scripts/dump_golden_vectors.py     $D/$3/best_generator.pt $W/$1.golden.json --arch $2 --run-config $D/$3/run_config.yaml --wgan-repo $R
done
ls -l $W
```

Run under `bash` (zsh does not word-split `$spec`). Expected blob sizes, computed from layer shapes in the spec: 6,973,840 / 13,124,608 / 7,748,612 bytes plus a header under 200 B each. A size off by more than the header means a wrong layer list: stop. Check that the SIFT golden has a mix of zeros and non-zeros in every output row:

```bash
$PY -c "import json; o=json.load(open('$W/sift_128_v4.golden.json'))['outputs']; print([sum(1 for v in r if v==0.0) for r in o])"
```

Expected: eight counts, each strictly between 0 and 128. All-0 or all-128 means the pinned noise does not exercise the gate; widen or narrow the `3.0` scale and re-dump.

- [ ] **Step 10: Write `weights/PROVENANCE.md`**

One table row per blob: blob file, blob bytes (from `ls -l`), blob sha256, checkpoint path on the box, checkpoint sha256, `generator_weights` value from Step 2, WGAN repo commit (`git -C ~/TIG/wgan-synthetic rev-parse --short HEAD`), exporter command. Mark every value MEASURED with today's date.

- [ ] **Step 11: Commit**

```bash
git add scripts/export_generator_weights.py scripts/dump_golden_vectors.py scripts/tests/test_export_generator.py tig-challenges/src/vector_search/weights/glove_100_v1.bin tig-challenges/src/vector_search/weights/nytimes_256_v3.bin tig-challenges/src/vector_search/weights/sift_128_v4.bin tig-challenges/src/vector_search/weights/glove_100_v1.golden.json tig-challenges/src/vector_search/weights/nytimes_256_v3.golden.json tig-challenges/src/vector_search/weights/sift_128_v4.golden.json tig-challenges/src/vector_search/weights/PROVENANCE.md
git status --short
ALLOW_LARGE_COMMIT=1 git commit -m "feat(c004): export SIFT v4, NYTimes v3 and GloVe v1 as TIGGAN02 blobs" # allow-large-commit
```

---

### Task 2: Ungated `gan_generator` module and the `TIGGAN02` container parser

Moves the CPU-only generator code out from behind the `c004` feature so it tests locally, and adds the container parser. Gated code still has to compile, and that can only be checked on the box, so this task ends with a box run.

**Files:**
- Create: `tig-challenges/src/gan_generator/mod.rs`, `tig-challenges/src/gan_generator/blob.rs`
- Move: `tig-challenges/src/vector_search/generator.rs` → `tig-challenges/src/gan_generator/v1.rs` (`git mv`)
- Create: `tig-challenges/src/vector_search/generator.rs` (a shim, replaced in Task 7)
- Modify: `tig-challenges/src/lib.rs:213` (add `pub mod gan_generator;` on the line after `pub mod audit_sampling;`)

**Interfaces:**
- Produces:
  ```rust
  // gan_generator/blob.rs
  pub struct Tensor { pub rows: usize, pub cols: usize, pub data: Vec<f32> }
  pub struct Container { pub arch: u32, pub latent_dim: usize, pub output_dim: usize,
                         pub scalars: Vec<f32>, pub tensors: Vec<Tensor> }
  pub fn parse_container(blob: &[u8]) -> anyhow::Result<Container>;
  pub const MAX_SCALARS: usize = 16;
  pub const MAX_TENSORS: usize = 64;
  // gan_generator/v1.rs: unchanged public items Layer, GeneratorWeights, parse_weights,
  // forward_cpu, LATENT_DIM; `take`, `read_u32`, `read_f32s` become pub(super).
  ```

- [ ] **Step 1: Move the file and wire the modules**

```bash
cd tig-challenges/src && mkdir gan_generator && git mv vector_search/generator.rs gan_generator/v1.rs
```

In `gan_generator/v1.rs`:
- change `fn take`, `fn read_u32`, `fn read_f32s` to `pub(super) fn`;
- replace `pub(super) const V1_BLOB: &[u8] = include_bytes!("weights/v1_sift.bin");` with
  ```rust
  /// Test fixture only. Scenarios embed their own blobs; an ungated
  /// `include_bytes!` here would put 7 MB into every build of the crate.
  #[cfg(test)]
  const V1_BLOB: &[u8] = include_bytes!("../vector_search/weights/v1_sift.bin");
  ```
- delete `pub fn v1_weights()` and `pub fn weights_from()`; in the tests replace `v1_weights().unwrap()` with `parse_weights(V1_BLOB).unwrap()`;
- change the golden include to `include_str!("../vector_search/weights/golden_vectors.json")`;
- cut the test `every_scenario_blob_matches_its_declared_dims` out (it needs `Scenario`, which is gated). Keep its text: it goes into the shim below.

`gan_generator/mod.rs`:

```rust
//! CPU-only generator code: blob parsing and one reference forward pass per
//! architecture. Deliberately outside the `c004` feature gate, like
//! `audit_sampling`, so it builds and tests on a machine with no CUDA toolkit.
//! The GPU drivers live in `vector_search::generator`.

pub mod v1;

pub use v1::Layer;
```

(`pub mod blob;` is added in Step 3, together with the file. Declaring it here would make Step 2's test run fail to compile for a reason unrelated to the move.)

New `vector_search/generator.rs` (shim; Task 7 replaces it with the GPU drivers):

```rust
pub use crate::gan_generator::v1::{parse_weights as weights_from, Layer, LATENT_DIM};

/// Weights are committed rather than fetched: every verifier regenerates the
/// instance independently, so a single differing byte would fail verification
/// network-wide.
pub(super) const V1_BLOB: &[u8] = include_bytes!("weights/v1_sift.bin");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_scenario_blob_matches_its_declared_dims() {
        // (body moved verbatim from the old generator.rs)
        for scenario in [crate::vector_search::Scenario::SIFT_128] {
            let config = crate::vector_search::ScenarioConfig::from(scenario);
            let weights = weights_from(config.weights).unwrap_or_else(|e| {
                panic!("scenario {} has an unparseable blob: {}", scenario, e)
            });
            let derived = weights.layers.last().unwrap().out_dim;
            assert_eq!(derived, config.vector_dims,
                "scenario {}: blob produces {} dims but config declares {}",
                scenario, derived, config.vector_dims);
            assert_eq!(weights.layers[0].in_dim, LATENT_DIM,
                "scenario {}: first layer must consume LATENT_DIM", scenario);
        }
    }
}
```

`vector_search/mod.rs` line 13 (`use generator::{weights_from, LATENT_DIM};`) and its uses of `generator::Layer` keep working through the shim's re-exports; do not edit them in this task.

- [ ] **Step 2: Confirm the moved tests pass locally**

```bash
cargo test -p tig-challenges gan_generator::v1 2>&1 | tail -15
```

Expected: `test result: ok. 8 passed` (the 9 old generator tests minus the one moved to the shim). If the count differs, a test was lost in the move: diff against `git show HEAD:tig-challenges/src/vector_search/generator.rs`.

- [ ] **Step 3: Write the failing container tests**

Add `pub mod blob;` to `gan_generator/mod.rs`, then create `gan_generator/blob.rs`, tests first:

```rust
use super::v1::{read_f32s, read_u32, take};
use anyhow::{anyhow, Result};

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A well-formed container whose tensor values are a small deterministic
    /// ramp, so tests that check values are not comparing constants.
    pub(crate) fn container_bytes(arch: u32, latent: u32, out: u32,
                                  scalars: &[f32], tensors: &[(u32, u32)]) -> Vec<u8> {
        let mut v = Vec::from(*b"TIGGAN02");
        for x in [arch, latent, out, scalars.len() as u32] { v.extend_from_slice(&x.to_le_bytes()); }
        for s in scalars { v.extend_from_slice(&s.to_le_bytes()); }
        v.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        let mut n = 0u32;
        for &(rows, cols) in tensors {
            v.extend_from_slice(&rows.to_le_bytes());
            v.extend_from_slice(&cols.to_le_bytes());
            for _ in 0..rows * cols {
                v.extend_from_slice(&(((n % 17) as f32 - 8.0) * 0.0625).to_le_bytes());
                n += 1;
            }
        }
        v
    }

    #[test]
    fn parses_header_scalars_and_tensors() {
        let c = parse_container(&container_bytes(2, 24, 8, &[0.5, 0.25], &[(3, 2), (3, 1)])).unwrap();
        assert_eq!((c.arch, c.latent_dim, c.output_dim), (2, 24, 8));
        assert_eq!(c.scalars, vec![0.5, 0.25]);
        assert_eq!((c.tensors[0].rows, c.tensors[0].cols), (3, 2));
        assert_eq!(c.tensors[0].data, vec![-0.5, -0.4375, -0.375, -0.3125, -0.25, -0.1875]);
        assert_eq!(c.tensors[1].data[0], -0.125, "second tensor continues the ramp");
    }

    #[test]
    fn rejects_bad_magic() {
        let mut b = container_bytes(0, 4, 2, &[1e-8], &[(2, 4), (2, 1)]);
        b[7] = b'3';
        assert!(parse_container(&b).unwrap_err().to_string().contains("magic"));
    }

    #[test]
    fn rejects_truncation() {
        let b = container_bytes(0, 4, 2, &[1e-8], &[(2, 4), (2, 1)]);
        assert!(parse_container(&b[..b.len() - 4]).unwrap_err().to_string().contains("truncated"));
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut b = container_bytes(0, 4, 2, &[1e-8], &[(2, 4), (2, 1)]);
        b.push(0);
        assert!(parse_container(&b).unwrap_err().to_string().contains("trailing"));
    }

    #[test]
    fn rejects_absurd_scalar_count_before_reading_them() {
        let mut b = Vec::from(*b"TIGGAN02");
        for x in [0u32, 4, 2, u32::MAX] { b.extend_from_slice(&x.to_le_bytes()); }
        assert!(parse_container(&b).unwrap_err().to_string().contains("scalars"));
    }

    #[test]
    fn rejects_absurd_tensor_count_before_reading_them() {
        let mut b = Vec::from(*b"TIGGAN02");
        for x in [0u32, 4, 2, 0, u32::MAX] { b.extend_from_slice(&x.to_le_bytes()); }
        assert!(parse_container(&b).unwrap_err().to_string().contains("tensors"));
    }

    #[test]
    fn rejects_tensor_dims_that_overflow_the_element_count() {
        let mut b = Vec::from(*b"TIGGAN02");
        for x in [0u32, 4, 2, 0, 1, u32::MAX, u32::MAX] { b.extend_from_slice(&x.to_le_bytes()); }
        assert!(parse_container(&b).is_err());
    }

    #[test]
    fn rejects_a_zero_dimension() {
        let b = container_bytes(0, 4, 2, &[1e-8], &[(0, 4)]);
        assert!(parse_container(&b).unwrap_err().to_string().contains("zero"));
    }

    #[test]
    fn rejects_zero_latent_or_output_dim() {
        let b = container_bytes(0, 0, 2, &[1e-8], &[(2, 4), (2, 1)]);
        assert!(parse_container(&b).unwrap_err().to_string().contains("zero"));
    }
}
```

Mutations these catch, one each: the magic compare removed; the bounds check in `take` bypassed; the trailing-byte check removed; the `MAX_SCALARS` guard removed (the test would then hit the truncation error instead, whose message lacks "scalars"); the `MAX_TENSORS` guard removed; the `checked_mul` removed; the zero-dim check removed; the header zero check removed. `parses_header…` catches a wrong field order or a column-major read.

- [ ] **Step 4: Run, expect a compile failure**

```bash
cargo test -p tig-challenges gan_generator::blob 2>&1 | tail -5
```

Expected: `cannot find function parse_container`.

- [ ] **Step 5: Implement the parser (above the tests module in `blob.rs`)**

```rust
const MAGIC: &[u8; 8] = b"TIGGAN02";

/// Upper bounds checked BEFORE the counts are used as loop bounds. The largest
/// real architecture has 3 scalars and 15 tensors. Without these a hostile
/// count would still fail on truncation, but only after a very long loop of
/// failing reads is ruled out by inspection; an explicit bound needs no such
/// argument.
pub const MAX_SCALARS: usize = 16;
pub const MAX_TENSORS: usize = 64;

pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    /// Row-major `[rows][cols]`, matching PyTorch `nn.Linear.weight`.
    pub data: Vec<f32>,
}

pub struct Container {
    pub arch: u32,
    pub latent_dim: usize,
    pub output_dim: usize,
    pub scalars: Vec<f32>,
    pub tensors: Vec<Tensor>,
}

pub fn parse_container(blob: &[u8]) -> Result<Container> {
    let mut at = 0usize;
    if take(blob, &mut at, 8)? != MAGIC {
        return Err(anyhow!("weight blob has wrong magic; expected TIGGAN02"));
    }
    let arch = read_u32(blob, &mut at)?;
    let latent_dim = read_u32(blob, &mut at)? as usize;
    let output_dim = read_u32(blob, &mut at)? as usize;
    if latent_dim == 0 || output_dim == 0 {
        return Err(anyhow!("weight blob declares a zero latent or output dimension"));
    }

    let n_scalars = read_u32(blob, &mut at)? as usize;
    if n_scalars > MAX_SCALARS {
        return Err(anyhow!("weight blob declares {} scalars; at most {} allowed", n_scalars, MAX_SCALARS));
    }
    let scalars = read_f32s(blob, &mut at, n_scalars)?;

    let n_tensors = read_u32(blob, &mut at)? as usize;
    if n_tensors > MAX_TENSORS {
        return Err(anyhow!("weight blob declares {} tensors; at most {} allowed", n_tensors, MAX_TENSORS));
    }
    let mut tensors = Vec::new();
    for i in 0..n_tensors {
        let rows = read_u32(blob, &mut at)? as usize;
        let cols = read_u32(blob, &mut at)? as usize;
        if rows == 0 || cols == 0 {
            return Err(anyhow!("tensor {} has a zero dimension", i));
        }
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("tensor {} element count overflow: {} * {}", i, rows, cols))?;
        let data = read_f32s(blob, &mut at, count)?;
        tensors.push(Tensor { rows, cols, data });
    }
    if at != blob.len() {
        return Err(anyhow!("weight blob has {} trailing bytes", blob.len() - at));
    }
    Ok(Container { arch, latent_dim, output_dim, scalars, tensors })
}
```

- [ ] **Step 6: Run, expect 9 passed; then mutation-check two guards**

```bash
cargo test -p tig-challenges gan_generator 2>&1 | tail -5     # expect 17 passed (8 + 9)
```

Delete the `n_tensors > MAX_TENSORS` block → `rejects_absurd_tensor_count…` must FAIL on its message assertion. Restore. Delete the trailing-bytes block → `rejects_trailing_bytes` must FAIL. Restore. Re-run: 17 passed.

- [ ] **Step 7: Commit, push, and prove the gated build still compiles**

```bash
git add tig-challenges/src/lib.rs tig-challenges/src/gan_generator tig-challenges/src/vector_search/generator.rs
git status --short
git commit -m "refactor(c004): move CPU generator code out of the c004 gate; add the TIGGAN02 container parser"
git push origin HEAD
scripts/box_submit.sh task2 gpu generator
```

Expected on the box: exit 0, and the log shows `every_scenario_blob_matches_its_declared_dims ... ok` plus the 17 `gan_generator` tests. A compile error here is the gated shim: fix, recommit, resubmit.

---

### Task 3: `Generator`, `Mlp`, and the GloVe golden

**Files:**
- Modify: `tig-challenges/src/gan_generator/mod.rs`
- Create: `tig-challenges/src/gan_generator/mlp.rs`

**Interfaces:**
- Consumes: `blob::{Container, Tensor, parse_container}`, `v1::{Layer, parse_weights}`.
- Produces: `Generator` (with only the `Mlp` variant for now; Tasks 4 and 5 add the others), `Generator::from_blob`, `latent_dim`, `output_dim`, `forward_cpu`, `dense_cpu`, `normalize_cpu`, and the helpers `next_tensor`, `dense`, `no_bias`, `check_chain`, `expect_scalars` that Tasks 4 and 5 reuse.

- [ ] **Step 1: Write the failing tests** (at the bottom of `gan_generator/mod.rs`)

```rust
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::gan_generator::blob::tests::container_bytes;

    #[derive(serde::Deserialize)]
    struct Golden {
        latents: Vec<Vec<f32>>,
        gate_noise: Option<Vec<Vec<f32>>>,
        outputs: Vec<Vec<f32>>,
    }

    /// 1e-5 absolute on unit-norm outputs. PyTorch sums dot products in a
    /// different order than `dense_cpu`, so exact equality is not expected; the
    /// v1 fixture's largest observed deviation at this bound was 4.47e-7.
    /// `exact_support`: zeros must be zeros and non-zeros non-zeros (SIFT gate).
    pub(crate) fn assert_matches_golden(blob: &[u8], golden_json: &str, exact_support: bool) {
        let golden: Golden = serde_json::from_str(golden_json).unwrap();
        let generator = Generator::from_blob(blob).unwrap();
        assert_eq!(golden.latents.len(), 8, "golden file should carry 8 rows");
        for (row, (latent, expected)) in golden.latents.iter().zip(&golden.outputs).enumerate() {
            let noise = golden.gate_noise.as_ref().map(|n| n[row].as_slice());
            let got = generator.forward_cpu(latent, noise).unwrap();
            assert_eq!(got.len(), expected.len());
            for (i, (g, e)) in got.iter().zip(expected).enumerate() {
                if exact_support {
                    assert_eq!(*g == 0.0, *e == 0.0, "row {row} coord {i}: support differs (got {g}, expected {e})");
                }
                assert!((g - e).abs() < 1e-5, "row {row} coord {i}: got {g}, expected {e}");
            }
        }
    }

    #[test]
    fn glove_blob_matches_pytorch() {
        assert_matches_golden(
            include_bytes!("../vector_search/weights/glove_100_v1.bin"),
            include_str!("../vector_search/weights/glove_100_v1.golden.json"),
            false,
        );
    }

    #[test]
    fn glove_blob_declares_its_shape() {
        let g = Generator::from_blob(include_bytes!("../vector_search/weights/glove_100_v1.bin")).unwrap();
        assert_eq!((g.latent_dim(), g.output_dim()), (128, 100));
        assert!(matches!(g, Generator::Mlp(Mlp { normalize_eps: Some(e), .. }) if e == 1.0e-8));
    }

    #[test]
    fn tiggan01_still_loads_as_an_unnormalised_mlp() {
        let g = Generator::from_blob(include_bytes!("../vector_search/weights/v1_sift.bin")).unwrap();
        assert_eq!((g.latent_dim(), g.output_dim()), (128, 128));
        assert!(matches!(g, Generator::Mlp(Mlp { normalize_eps: None, .. })));
    }

    #[test]
    fn mlp_output_is_unit_norm() {
        let g = Generator::from_blob(include_bytes!("../vector_search/weights/glove_100_v1.bin")).unwrap();
        let latent: Vec<f32> = (0..128).map(|i| (i as f32) / 64.0 - 1.0).collect();
        let out = g.forward_cpu(&latent, None).unwrap();
        let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }

    #[test]
    fn rejects_unknown_arch() {
        let b = container_bytes(9, 4, 2, &[1e-8], &[(2, 4), (2, 1)]);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("arch 9"));
    }

    #[test]
    fn mlp_rejects_an_odd_tensor_count() {
        let b = container_bytes(0, 4, 2, &[1e-8], &[(2, 4)]);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("tensors"));
    }

    #[test]
    fn mlp_rejects_a_broken_layer_chain() {
        // layer 1 consumes 5 but layer 0 produces 3
        let b = container_bytes(0, 4, 2, &[1e-8], &[(3, 4), (3, 1), (2, 5), (2, 1)]);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("consumes 5"));
    }

    #[test]
    fn mlp_rejects_a_header_that_disagrees_with_the_tensors() {
        let b = container_bytes(0, 7, 2, &[1e-8], &[(2, 4), (2, 1)]);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("latent"));
        let b = container_bytes(0, 4, 3, &[1e-8], &[(2, 4), (2, 1)]);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("output"));
    }

    #[test]
    fn mlp_rejects_a_bias_of_the_wrong_shape() {
        let b = container_bytes(0, 4, 2, &[1e-8], &[(2, 4), (2, 2)]);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("bias"));
    }

    #[test]
    fn mlp_rejects_the_wrong_scalar_count() {
        let b = container_bytes(0, 4, 2, &[], &[(2, 4), (2, 1)]);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("scalars"));
    }

    #[test]
    fn gate_noise_is_refused_for_an_mlp() {
        let b = container_bytes(0, 4, 2, &[1e-8], &[(2, 4), (2, 1)]);
        let g = Generator::from_blob(&b).unwrap();
        assert!(g.forward_cpu(&[0.0; 4], Some(&[0.0; 2])).is_err());
        assert!(g.forward_cpu(&[0.0; 3], None).is_err(), "wrong latent length must be an error, not a panic");
    }
}
```

Mutations caught: golden → a transposed weight, a dropped bias, activation on the last layer, normalisation skipped; `declares_its_shape` → exporter wrote the wrong `eps` or parser dropped it; `tiggan01…` → the magic dispatch removed or v1 silently normalised; each `rejects_*` → the named check removed.

- [ ] **Step 2: Run, expect a compile failure** (`Generator` not found)

```bash
cargo test -p tig-challenges gan_generator 2>&1 | tail -5
```

- [ ] **Step 3: Implement**

`gan_generator/mod.rs`, replacing the file's head (keep the tests module from Step 1 below it):

```rust
//! CPU-only generator code: blob parsing and one reference forward pass per
//! architecture. Deliberately outside the `c004` feature gate, like
//! `audit_sampling`, so it builds and tests on a machine with no CUDA toolkit.
//! The GPU drivers live in `vector_search::generator`.

pub mod blob;
pub mod mlp;
pub mod v1;

use anyhow::{anyhow, Result};
use blob::Tensor;
pub use mlp::Mlp;
pub use v1::Layer;

pub const ARCH_MLP: u32 = 0;

pub enum Generator {
    Mlp(Mlp),
}

impl Generator {
    pub fn from_blob(blob: &[u8]) -> Result<Self> {
        if blob.len() >= 8 && &blob[..8] == b"TIGGAN01" {
            // The v1 format has no normalisation step and no header dims.
            let weights = v1::parse_weights(blob)?;
            return Ok(Generator::Mlp(Mlp { layers: weights.layers, normalize_eps: None }));
        }
        let container = blob::parse_container(blob)?;
        match container.arch {
            ARCH_MLP => Ok(Generator::Mlp(Mlp::from_container(container)?)),
            other => Err(anyhow!("weight blob declares unknown arch {}", other)),
        }
    }

    pub fn latent_dim(&self) -> usize {
        match self {
            Generator::Mlp(m) => m.layers[0].in_dim,
        }
    }

    pub fn output_dim(&self) -> usize {
        match self {
            Generator::Mlp(m) => m.layers.last().unwrap().out_dim,
        }
    }

    /// Reference forward pass. Exists to pin each GPU driver against PyTorch;
    /// `generate_instance` never calls it.
    ///
    /// `gate_noise` is the pre-smoothing logistic noise, one value per output
    /// coordinate: required for `StructuredGate`, refused for the others.
    pub fn forward_cpu(&self, latent: &[f32], gate_noise: Option<&[f32]>) -> Result<Vec<f32>> {
        if latent.len() != self.latent_dim() {
            return Err(anyhow!("latent has {} values; generator needs {}", latent.len(), self.latent_dim()));
        }
        match self {
            Generator::Mlp(m) => {
                if gate_noise.is_some() {
                    return Err(anyhow!("gate noise supplied to an mlp generator"));
                }
                Ok(m.forward_cpu(latent))
            }
        }
    }
}

/// One dense layer, in exactly `gan_linear`'s order: start from the bias, then
/// fused multiply-adds over k = 0..in_dim. `mul_add` because the kernel uses
/// `fmaf`; a plain `a * b + c` rounds differently.
pub fn dense_cpu(layer: &Layer, input: &[f32], activate: bool) -> Vec<f32> {
    let mut out = Vec::with_capacity(layer.out_dim);
    for col in 0..layer.out_dim {
        let w = &layer.weights[col * layer.in_dim..(col + 1) * layer.in_dim];
        let mut acc = layer.bias[col];
        for k in 0..layer.in_dim {
            acc = input[k].mul_add(w[k], acc);
        }
        out.push(if activate && acc < 0.0 { acc * 0.2 } else { acc });
    }
    out
}

/// `x / max(||x||, eps)`, sum of squares accumulated in index order.
pub fn normalize_cpu(x: &mut [f32], eps: f32) {
    let mut ss = 0.0f32;
    for v in x.iter() {
        ss = v.mul_add(*v, ss);
    }
    let norm = ss.sqrt().max(eps);
    for v in x.iter_mut() {
        *v /= norm;
    }
}

// ---- helpers shared by every architecture's `from_container` ----

pub(crate) fn expect_scalars(scalars: &[f32], want: usize, arch: &str) -> Result<()> {
    if scalars.len() != want {
        return Err(anyhow!("{} blob has {} scalars; expected {}", arch, scalars.len(), want));
    }
    if scalars.iter().any(|s| !s.is_finite()) {
        return Err(anyhow!("{} blob has a non-finite scalar", arch));
    }
    Ok(())
}

pub(crate) fn next_tensor(it: &mut std::vec::IntoIter<Tensor>, what: &str) -> Result<Tensor> {
    it.next().ok_or_else(|| anyhow!("weight blob ran out of tensors at {}", what))
}

/// A weight tensor plus its `rows x 1` bias tensor.
pub(crate) fn dense(it: &mut std::vec::IntoIter<Tensor>, what: &str) -> Result<Layer> {
    let w = next_tensor(it, what)?;
    let b = next_tensor(it, what)?;
    if b.cols != 1 || b.rows != w.rows {
        return Err(anyhow!("{}: bias is {}x{} but the weight has {} rows", what, b.rows, b.cols, w.rows));
    }
    Ok(Layer { in_dim: w.cols, out_dim: w.rows, weights: w.data, bias: b.data })
}

/// A bias-free map, as a `Layer` with a zero bias so it runs through the same
/// dense code on both CPU and GPU.
pub(crate) fn no_bias(it: &mut std::vec::IntoIter<Tensor>, what: &str) -> Result<Layer> {
    let w = next_tensor(it, what)?;
    Ok(Layer { in_dim: w.cols, out_dim: w.rows, bias: vec![0.0; w.rows], weights: w.data })
}

/// Each layer must consume what the previous one produces.
pub(crate) fn check_chain(layers: &[Layer], what: &str) -> Result<()> {
    for i in 1..layers.len() {
        if layers[i].in_dim != layers[i - 1].out_dim {
            return Err(anyhow!("{} layer {} consumes {} but layer {} produces {}",
                what, i, layers[i].in_dim, i - 1, layers[i - 1].out_dim));
        }
    }
    Ok(())
}
```

`gan_generator/mlp.rs`:

```rust
use super::{blob::Container, check_chain, dense, dense_cpu, expect_scalars, normalize_cpu, Layer};
use anyhow::{anyhow, Result};

pub struct Mlp {
    pub layers: Vec<Layer>,
    /// `Some(eps)` for TIGGAN02 blobs, whose rows are unit-normalised as the
    /// WGAN sampler does. `None` only for the TIGGAN01 fixture.
    pub normalize_eps: Option<f32>,
}

impl Mlp {
    pub fn from_container(c: Container) -> Result<Self> {
        expect_scalars(&c.scalars, 1, "mlp")?;
        if c.tensors.is_empty() || c.tensors.len() % 2 != 0 {
            return Err(anyhow!("mlp blob has {} tensors; expected a positive even count", c.tensors.len()));
        }
        let count = c.tensors.len() / 2;
        let mut it = c.tensors.into_iter();
        let mut layers = Vec::new();
        for i in 0..count {
            layers.push(dense(&mut it, &format!("mlp layer {}", i))?);
        }
        check_chain(&layers, "mlp")?;
        if layers[0].in_dim != c.latent_dim {
            return Err(anyhow!("mlp header says latent {} but layer 0 consumes {}", c.latent_dim, layers[0].in_dim));
        }
        if layers[count - 1].out_dim != c.output_dim {
            return Err(anyhow!("mlp header says output {} but the last layer produces {}", c.output_dim, layers[count - 1].out_dim));
        }
        Ok(Mlp { layers, normalize_eps: Some(c.scalars[0]) })
    }

    pub fn forward_cpu(&self, latent: &[f32]) -> Vec<f32> {
        let mut current = latent.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            current = dense_cpu(layer, &current, i + 1 < self.layers.len());
        }
        if let Some(eps) = self.normalize_eps {
            normalize_cpu(&mut current, eps);
        }
        current
    }
}
```

- [ ] **Step 4: Run, expect 28 passed (17 + 11)**

```bash
cargo test -p tig-challenges gan_generator 2>&1 | tail -5
```

- [ ] **Step 5: Mutation-check**

(a) In `Mlp::forward_cpu` change `i + 1 < self.layers.len()` to `true` → `glove_blob_matches_pytorch` must FAIL. (b) Remove the `normalize_cpu` call → `glove_blob_matches_pytorch` and `mlp_output_is_unit_norm` must FAIL. (c) In `dense_cpu` index `w[k]` as `layer.weights[k * layer.out_dim + col]` (transpose) → golden must FAIL or panic. Restore each; re-run: 28 passed.

- [ ] **Step 6: Commit**

```bash
git add tig-challenges/src/gan_generator/mod.rs tig-challenges/src/gan_generator/mlp.rs
git status --short
git commit -m "feat(c004): Generator::from_blob and the mlp reference pass, pinned to the GloVe v1 golden"
```

---

### Task 4: `Spherical` reference pass and the NYTimes golden

**Files:**
- Create: `tig-challenges/src/gan_generator/spherical.rs`
- Modify: `tig-challenges/src/gan_generator/mod.rs` (variant, `ARCH_SPHERICAL = 2`, three `match` arms, tests)

**Interfaces:**
- Consumes: helpers from Task 3.
- Produces: `Spherical` (fields as in the shared vocabulary), `Generator::Spherical`.

- [ ] **Step 1: Write the failing tests** (append inside `gan_generator/mod.rs`'s tests module)

```rust
    const NYT: &[u8] = include_bytes!("../vector_search/weights/nytimes_256_v3.bin");

    #[test]
    fn nytimes_blob_matches_pytorch() {
        assert_matches_golden(NYT, include_str!("../vector_search/weights/nytimes_256_v3.golden.json"), false);
    }

    #[test]
    fn nytimes_blob_declares_its_shape() {
        let g = Generator::from_blob(NYT).unwrap();
        assert_eq!((g.latent_dim(), g.output_dim()), (512, 256));
        let Generator::Spherical(s) = &g else { panic!("expected the spherical variant") };
        assert_eq!(s.trunk[0].in_dim, 256, "trunk consumes latent_dim - skip_dim");
        assert_eq!(s.tangent_in.in_dim, 256, "tangent_in consumes skip_dim");
        assert!((s.cos_r * s.cos_r + s.sin_r * s.sin_r - 1.0).abs() < 1e-6);
        assert!(s.sin_r > 0.19 && s.cos_r > 0.07, "r lies in [0.2, 1.5], so both are positive");
    }

    #[test]
    fn spherical_output_is_unit_norm_and_the_tangent_is_orthogonal() {
        let g = Generator::from_blob(NYT).unwrap();
        let Generator::Spherical(s) = &g else { panic!() };
        let latent: Vec<f32> = (0..512).map(|i| ((i * 37 % 101) as f32) / 50.0 - 1.0).collect();
        let out = g.forward_cpu(&latent, None).unwrap();
        let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
        // out . u == cos_r exactly when t is orthogonal to u and both are unit.
        let u = s.direction_cpu(&latent);
        let dot: f32 = out.iter().zip(&u).map(|(a, b)| a * b).sum();
        assert!((dot - s.cos_r).abs() < 1e-5, "out.u was {dot}, cos_r is {}", s.cos_r);
    }

    /// trunk 4->3, direction 2x3, tangent_in 5x4 (+b), gamma 5x3 (+b), beta 5x3 (+b), tangent_out 2x5 (+b)
    fn tiny_spherical(latent: u32, shapes: &[(u32, u32)]) -> Vec<u8> {
        container_bytes(2, latent, 2, &[0.6, 0.8, 1e-8], shapes)
    }
    const TINY_SPH: [(u32, u32); 11] =
        [(3, 4), (3, 1), (2, 3), (5, 4), (5, 1), (5, 3), (5, 1), (5, 3), (5, 1), (2, 5), (2, 1)];

    #[test]
    fn spherical_accepts_a_consistent_tiny_blob() {
        assert!(Generator::from_blob(&tiny_spherical(8, &TINY_SPH)).is_ok());
    }

    #[test]
    fn spherical_rejects_a_latent_that_is_not_trunk_plus_skip() {
        assert!(Generator::from_blob(&tiny_spherical(9, &TINY_SPH)).unwrap_err().to_string().contains("latent"));
    }

    #[test]
    fn spherical_rejects_a_gamma_that_does_not_match_tangent_in() {
        let mut shapes = TINY_SPH;
        shapes[5] = (6, 3);
        shapes[6] = (6, 1);
        assert!(Generator::from_blob(&tiny_spherical(8, &shapes)).unwrap_err().to_string().contains("gamma"));
    }

    #[test]
    fn spherical_rejects_an_even_tensor_count() {
        assert!(Generator::from_blob(&tiny_spherical(8, &TINY_SPH[..10])).unwrap_err().to_string().contains("tensors"));
    }
```

Mutations caught: golden → `a·γ` where `a·(1+γ)` belongs, swapped `cos`/`sin`, projection done with un-normalised `u`, trunk-final activation dropped; orthogonality test → the projection step deleted (then `out·u ≠ cos_r`); shape tests → each relation check removed.

- [ ] **Step 2: Run, expect a compile failure**

- [ ] **Step 3: Implement `gan_generator/spherical.rs`**

```rust
use super::{blob::Container, check_chain, dense, dense_cpu, expect_scalars, no_bias, normalize_cpu, Layer};
use anyhow::{anyhow, Result};

pub struct Spherical {
    pub trunk: Vec<Layer>,
    pub direction: Layer,
    pub tangent_in: Layer,
    pub gamma: Layer,
    pub beta: Layer,
    pub tangent_out: Layer,
    pub cos_r: f32,
    pub sin_r: f32,
    pub eps: f32,
}

impl Spherical {
    pub fn from_container(c: Container) -> Result<Self> {
        expect_scalars(&c.scalars, 3, "spherical")?;
        let n = c.tensors.len();
        if n < 11 || n % 2 == 0 {
            return Err(anyhow!("spherical blob has {} tensors; expected an odd count of at least 11", n));
        }
        let trunk_layers = (n - 9) / 2;
        let mut it = c.tensors.into_iter();
        let mut trunk = Vec::new();
        for i in 0..trunk_layers {
            trunk.push(dense(&mut it, &format!("spherical trunk layer {}", i))?);
        }
        check_chain(&trunk, "spherical trunk")?;
        let direction = no_bias(&mut it, "direction")?;
        let tangent_in = dense(&mut it, "tangent_in")?;
        let gamma = dense(&mut it, "gamma")?;
        let beta = dense(&mut it, "beta")?;
        let tangent_out = dense(&mut it, "tangent_out")?;

        let hidden = trunk[trunk_layers - 1].out_dim;
        let checks = [
            (trunk[0].in_dim + tangent_in.in_dim == c.latent_dim, "latent_dim must equal trunk input + tangent_in input"),
            (direction.in_dim == hidden, "direction must consume the trunk output"),
            (direction.out_dim == c.output_dim, "direction must produce output_dim"),
            (gamma.in_dim == hidden && gamma.out_dim == tangent_in.out_dim, "gamma must map the trunk output to tangent_in's width"),
            (beta.in_dim == hidden && beta.out_dim == tangent_in.out_dim, "beta must map the trunk output to tangent_in's width"),
            (tangent_out.in_dim == tangent_in.out_dim, "tangent_out must consume tangent_in's width"),
            (tangent_out.out_dim == c.output_dim, "tangent_out must produce output_dim"),
        ];
        for (ok, message) in checks {
            if !ok {
                return Err(anyhow!("spherical blob: {}", message));
            }
        }
        Ok(Spherical { trunk, direction, tangent_in, gamma, beta, tangent_out,
                       cos_r: c.scalars[0], sin_r: c.scalars[1], eps: c.scalars[2] })
    }

    fn trunk_cpu(&self, latent: &[f32]) -> Vec<f32> {
        let mut h = latent[..self.trunk[0].in_dim].to_vec();
        for layer in &self.trunk {
            // Activation after EVERY trunk layer, the last included: the
            // PyTorch trunk is [Linear, LeakyReLU] x T, unlike the mlp's.
            h = dense_cpu(layer, &h, true);
        }
        h
    }

    /// The unit direction `u`. Public so a test can check `out . u == cos_r`.
    pub fn direction_cpu(&self, latent: &[f32]) -> Vec<f32> {
        let mut u = dense_cpu(&self.direction, &self.trunk_cpu(latent), false);
        normalize_cpu(&mut u, self.eps);
        u
    }

    pub fn forward_cpu(&self, latent: &[f32]) -> Vec<f32> {
        let h = self.trunk_cpu(latent);
        let mut u = dense_cpu(&self.direction, &h, false);
        normalize_cpu(&mut u, self.eps);

        let z_skip = &latent[self.trunk[0].in_dim..];
        let a = dense_cpu(&self.tangent_in, z_skip, false);
        let g = dense_cpu(&self.gamma, &h, false);
        let b = dense_cpu(&self.beta, &h, false);
        let modulated: Vec<f32> = (0..a.len())
            .map(|j| {
                let v = a[j].mul_add(1.0 + g[j], b[j]);
                if v < 0.0 { v * 0.2 } else { v }
            })
            .collect();
        let mut t = dense_cpu(&self.tangent_out, &modulated, false);

        let mut dot = 0.0f32;
        for j in 0..t.len() {
            dot = t[j].mul_add(u[j], dot);
        }
        for j in 0..t.len() {
            t[j] = (-dot).mul_add(u[j], t[j]);
        }
        normalize_cpu(&mut t, self.eps);

        (0..u.len()).map(|j| self.cos_r.mul_add(u[j], self.sin_r * t[j])).collect()
    }
}
```

In `mod.rs`: add `pub mod spherical; pub use spherical::Spherical; pub const ARCH_SPHERICAL: u32 = 2;`, the variant `Spherical(Spherical)`, and arms:
`from_blob`: `ARCH_SPHERICAL => Ok(Generator::Spherical(Spherical::from_container(container)?)),`
`latent_dim`: `Generator::Spherical(s) => s.trunk[0].in_dim + s.tangent_in.in_dim,`
`output_dim`: `Generator::Spherical(s) => s.direction.out_dim,`
`forward_cpu`: same shape as the `Mlp` arm, error text `"gate noise supplied to a spherical generator"`, calling `s.forward_cpu(latent)`.

- [ ] **Step 4: Run, expect 35 passed (28 + 7)**

- [ ] **Step 5: Mutation-check**

(a) `1.0 + g[j]` → `g[j]`: golden must FAIL. (b) Delete the projection loop (`t[j] = (-dot)…`): the orthogonality test must FAIL. (c) Swap `self.cos_r` and `self.sin_r` in the last line: golden must FAIL. (d) `dense_cpu(layer, &h, true)` → activation `false` on the last trunk layer only: golden must FAIL. Restore each; re-run: 35 passed.

- [ ] **Step 6: Commit**

```bash
git add tig-challenges/src/gan_generator/mod.rs tig-challenges/src/gan_generator/spherical.rs
git status --short
git commit -m "feat(c004): spherical reference pass, pinned to the NYTimes v3 golden"
```

---

### Task 5: `StructuredGate` reference pass and the SIFT v4 golden

**Files:**
- Create: `tig-challenges/src/gan_generator/structured_gate.rs`
- Modify: `tig-challenges/src/gan_generator/mod.rs` (variant, `ARCH_STRUCTURED_GATE = 1`, arms, tests)

**Interfaces:**
- Produces: `StructuredGate`, `Generator::StructuredGate`, `pub fn softplus(x: f32) -> f32`, and
  `pub fn gate_margin_cpu(&self, latent: &[f32], gate_noise: &[f32]) -> Vec<f32>`, returning `logit_j + smoothed_noise_j` per coordinate. Tasks 9 and 10 use it to find gates too close to the threshold to compare across CPU and GPU.

- [ ] **Step 1: Write the failing tests** (append inside the tests module)

```rust
    const SIFT: &[u8] = include_bytes!("../vector_search/weights/sift_128_v4.bin");

    #[test]
    fn sift_v4_blob_matches_pytorch_with_an_exact_support_pattern() {
        assert_matches_golden(SIFT, include_str!("../vector_search/weights/sift_128_v4.golden.json"), true);
    }

    #[test]
    fn sift_v4_blob_declares_its_shape() {
        let g = Generator::from_blob(SIFT).unwrap();
        assert_eq!((g.latent_dim(), g.output_dim()), (128, 128));
        let Generator::StructuredGate(s) = &g else { panic!("expected the structured_gate variant") };
        assert_eq!(s.logit_clamp, 4.0, "configs/sift/v4.yaml sets logit_clamp: 4.0");
        assert_eq!(s.magnitude_floor, 1.0e-6);
        assert_eq!((s.sparsity_head.out_dim, s.coupling.in_dim, s.smoothing.out_dim), (1, 128, 128));
    }

    #[test]
    fn structured_gate_requires_noise_of_the_right_length() {
        let g = Generator::from_blob(SIFT).unwrap();
        assert!(g.forward_cpu(&[0.0; 128], None).is_err());
        assert!(g.forward_cpu(&[0.0; 128], Some(&[0.0; 127])).is_err());
    }

    #[test]
    fn structured_gate_output_is_non_negative_and_unit_norm() {
        let g = Generator::from_blob(SIFT).unwrap();
        let latent: Vec<f32> = (0..128).map(|i| ((i * 29 % 97) as f32) / 48.0 - 1.0).collect();
        let noise: Vec<f32> = (0..128).map(|i| ((i * 53 % 89) as f32) / 15.0 - 3.0).collect();
        let out = g.forward_cpu(&latent, Some(&noise)).unwrap();
        assert!(out.iter().all(|v| *v >= 0.0));
        assert!((out.iter().map(|x| x * x).sum::<f32>().sqrt() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_row_with_every_gate_closed_falls_back_to_the_argmax_logit() {
        let g = Generator::from_blob(SIFT).unwrap();
        let latent: Vec<f32> = (0..128).map(|i| ((i * 29 % 97) as f32) / 48.0 - 1.0).collect();
        // -1e6 everywhere closes every gate whatever the logits are (|logit| <= 4).
        // Smoothing is linear with positive weights, so the smoothed noise is
        // large and negative at every coordinate too.
        let out = g.forward_cpu(&latent, Some(&[-1.0e6; 128])).unwrap();
        let open: Vec<usize> = (0..128).filter(|j| out[*j] != 0.0).collect();
        assert_eq!(open.len(), 1, "exactly one coordinate is rescued");
        assert!((out[open[0]] - 1.0).abs() < 1e-6, "a one-hot row normalises to 1.0");
    }

    #[test]
    fn softplus_matches_pytorch_including_the_threshold() {
        use crate::gan_generator::structured_gate::softplus;
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
        assert_eq!(softplus(25.0), 25.0, "above 20 PyTorch returns x unchanged");
        assert!(softplus(-100.0) >= 0.0 && softplus(-100.0) < 1e-30);
    }

    #[test]
    fn structured_gate_rejects_a_sparsity_head_wider_than_one() {
        // trunk 3x4, magnitude 2x3, gate 2x3, sparsity 2x3 (WRONG), coupling 2x2, smoothing 2x2
        let shapes = [(3, 4), (3, 1), (2, 3), (2, 1), (2, 3), (2, 1), (2, 3), (2, 1), (2, 2), (2, 2)];
        let b = container_bytes(1, 4, 2, &[4.0, 1e-6, 1e-8], &shapes);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("sparsity"));
    }

    #[test]
    fn structured_gate_rejects_a_non_square_coupling() {
        let shapes = [(3, 4), (3, 1), (2, 3), (2, 1), (2, 3), (2, 1), (1, 3), (1, 1), (2, 3), (2, 2)];
        let b = container_bytes(1, 4, 2, &[4.0, 1e-6, 1e-8], &shapes);
        assert!(Generator::from_blob(&b).unwrap_err().to_string().contains("coupling"));
    }
```

The fallback test's premise, "smoothing has positive weights", is a property of the exported matrix (a Gaussian kernel times a positive scale). Check it once while writing the test: `python -c` load the blob's last tensor and assert `min() >= 0`. If it is false, replace the constant noise with one that the test computes to be negative after smoothing.

Mutations caught: golden with exact support → wrong threshold direction, `tanh` clamp omitted, sparsity not broadcast, `coupling` applied to the wrong operand, softplus floor missing; fallback test → fallback deleted (row would be all zeros, norm 0, NaN or zeros) or picking `argmin`; softplus test → threshold branch missing (25.0 would still pass approximately, so the test asserts exact equality) .

- [ ] **Step 2: Run, expect a compile failure**

- [ ] **Step 3: Implement `gan_generator/structured_gate.rs`**

```rust
use super::{blob::Container, check_chain, dense, dense_cpu, expect_scalars, no_bias, normalize_cpu, Layer};
use anyhow::{anyhow, Result};

pub struct StructuredGate {
    pub trunk: Vec<Layer>,
    pub magnitude_head: Layer,
    pub gate_head: Layer,
    pub sparsity_head: Layer,
    /// The Conv3d gate coupling, baked to a dense map by the exporter.
    pub coupling: Layer,
    /// The fixed noise-smoothing kernel times the per-position scale, baked.
    pub smoothing: Layer,
    pub logit_clamp: f32,
    pub magnitude_floor: f32,
    pub eps: f32,
}

/// PyTorch's `F.softplus` with its defaults: beta 1, threshold 20.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

impl StructuredGate {
    pub fn from_container(c: Container) -> Result<Self> {
        expect_scalars(&c.scalars, 3, "structured_gate")?;
        let n = c.tensors.len();
        if n < 10 || n % 2 != 0 {
            return Err(anyhow!("structured_gate blob has {} tensors; expected an even count of at least 10", n));
        }
        let trunk_layers = (n - 8) / 2;
        let mut it = c.tensors.into_iter();
        let mut trunk = Vec::new();
        for i in 0..trunk_layers {
            trunk.push(dense(&mut it, &format!("structured_gate trunk layer {}", i))?);
        }
        check_chain(&trunk, "structured_gate trunk")?;
        let magnitude_head = dense(&mut it, "magnitude_head")?;
        let gate_head = dense(&mut it, "gate_head")?;
        let sparsity_head = dense(&mut it, "sparsity_head")?;
        let coupling = no_bias(&mut it, "coupling")?;
        let smoothing = no_bias(&mut it, "smoothing")?;

        let hidden = trunk[trunk_layers - 1].out_dim;
        let out = c.output_dim;
        let checks = [
            (trunk[0].in_dim == c.latent_dim, "the trunk must consume latent_dim"),
            (magnitude_head.in_dim == hidden && magnitude_head.out_dim == out, "magnitude_head must map the trunk output to output_dim"),
            (gate_head.in_dim == hidden && gate_head.out_dim == out, "gate_head must map the trunk output to output_dim"),
            (sparsity_head.in_dim == hidden && sparsity_head.out_dim == 1, "sparsity_head must map the trunk output to one value"),
            (coupling.in_dim == out && coupling.out_dim == out, "coupling must be output_dim x output_dim"),
            (smoothing.in_dim == out && smoothing.out_dim == out, "smoothing must be output_dim x output_dim"),
            (c.scalars[0] > 0.0, "logit_clamp must be positive"),
        ];
        for (ok, message) in checks {
            if !ok {
                return Err(anyhow!("structured_gate blob: {}", message));
            }
        }
        Ok(StructuredGate { trunk, magnitude_head, gate_head, sparsity_head, coupling, smoothing,
                            logit_clamp: c.scalars[0], magnitude_floor: c.scalars[1], eps: c.scalars[2] })
    }

    fn trunk_cpu(&self, latent: &[f32]) -> Vec<f32> {
        let mut h = latent.to_vec();
        for layer in &self.trunk {
            h = dense_cpu(layer, &h, true); // activation after every trunk layer
        }
        h
    }

    fn logits_cpu(&self, h: &[f32]) -> Vec<f32> {
        let coupled = dense_cpu(&self.coupling, &dense_cpu(&self.gate_head, h, false), false);
        let s = dense_cpu(&self.sparsity_head, h, false)[0];
        coupled.iter().map(|c| self.logit_clamp * ((c + s) / self.logit_clamp).tanh()).collect()
    }

    /// `logit_j + smoothed_noise_j`: the quantity whose sign opens gate j.
    pub fn gate_margin_cpu(&self, latent: &[f32], gate_noise: &[f32]) -> Vec<f32> {
        let logits = self.logits_cpu(&self.trunk_cpu(latent));
        let smoothed = dense_cpu(&self.smoothing, gate_noise, false);
        logits.iter().zip(&smoothed).map(|(l, n)| l + n).collect()
    }

    pub fn forward_cpu(&self, latent: &[f32], gate_noise: &[f32]) -> Vec<f32> {
        let h = self.trunk_cpu(latent);
        let logits = self.logits_cpu(&h);
        let smoothed = dense_cpu(&self.smoothing, gate_noise, false);
        let magnitude: Vec<f32> = dense_cpu(&self.magnitude_head, &h, false)
            .iter()
            .map(|m| softplus(*m).max(self.magnitude_floor))
            .collect();

        let mut out = vec![0.0f32; logits.len()];
        let mut any_open = false;
        // Strict `>` keeps the FIRST maximum, as torch.argmax does.
        let mut best = 0usize;
        for j in 0..logits.len() {
            if logits[j] > logits[best] {
                best = j;
            }
            if logits[j] + smoothed[j] > 0.0 {
                out[j] = magnitude[j];
                any_open = true;
            }
        }
        if !any_open {
            out[best] = magnitude[best];
        }
        normalize_cpu(&mut out, self.eps);
        out
    }
}
```

In `mod.rs`: `pub mod structured_gate; pub use structured_gate::StructuredGate; pub const ARCH_STRUCTURED_GATE: u32 = 1;`, the variant, and arms:
`from_blob`: `ARCH_STRUCTURED_GATE => Ok(Generator::StructuredGate(StructuredGate::from_container(container)?)),`
`latent_dim`: `Generator::StructuredGate(s) => s.trunk[0].in_dim,`
`output_dim`: `Generator::StructuredGate(s) => s.magnitude_head.out_dim,`
`forward_cpu`:
```rust
            Generator::StructuredGate(s) => {
                let noise = gate_noise.ok_or_else(|| anyhow!("structured_gate needs gate noise"))?;
                if noise.len() != s.magnitude_head.out_dim {
                    return Err(anyhow!("gate noise has {} values; generator needs {}", noise.len(), s.magnitude_head.out_dim));
                }
                Ok(s.forward_cpu(latent, noise))
            }
```

- [ ] **Step 4: Run, expect 43 passed (35 + 8)**

- [ ] **Step 5: Mutation-check**

(a) `logits[j] + smoothed[j] > 0.0` → `< 0.0`: golden must FAIL on support. (b) Delete the `if !any_open` block: the fallback test must FAIL. (c) `(c + s)` → `c`: golden must FAIL. (d) `dense_cpu(&self.coupling, &dense_cpu(&self.gate_head, …))` → drop the coupling: golden must FAIL (if it does not, the trained coupling is near identity; record that in the task notes and keep test 5 of the exporter pytest as the guard). (e) `if x > 20.0 { x }` removed: `softplus(25.0)` exactness must FAIL. Restore each; re-run: 43 passed.

- [ ] **Step 6: Commit and push**

```bash
git add tig-challenges/src/gan_generator/mod.rs tig-challenges/src/gan_generator/structured_gate.rs
git status --short
git commit -m "feat(c004): structured_gate reference pass, pinned to the SIFT v4 golden"
git push origin HEAD
```

---

## GPU tasks: how every step below is run

Gated code compiles only on the box. The cycle for each GPU step is: commit, `git push origin HEAD`, then

```bash
scripts/box_submit.sh <log-name> gpu <cargo test filter>
```

with a Bash timeout of at least `5400000` ms. "Expect FAIL" steps are real: push the failing test, see the failure in the job log, then push the fix. Do not skip the failing run to save a round trip; a test that was never seen failing proves nothing. To keep history readable, the failing-test commit and the fix may be squashed **before** the task's final push only if the failing job's id is recorded in the task notes.

---

### Task 6: GPU driver refactor, `gan_row_normalize`, and the GloVe scenario

Replaces the shim with the real GPU module, moves `generate_vectors` onto it without changing what SIFT v1 generates, then adds GloVe.

**Files:**
- Rewrite: `tig-challenges/src/vector_search/generator.rs`
- Modify: `tig-challenges/src/vector_search/mod.rs` (`generate_vectors` at lines 79-211, `Database::generate`, `Challenge::for_nonce`, the two tests that call `generate_vectors` directly near lines 1697 and 1812, `gpu_instance`)
- Modify: `tig-challenges/src/vector_search/scenarios.rs`
- Modify: `tig-challenges/src/vector_search/kernels.cu` (append one kernel)

**Interfaces:**
- Consumes: `crate::gan_generator::{Generator, Layer, Mlp}`.
- Produces:
  ```rust
  // vector_search/generator.rs
  pub(super) const ROW_BLOCK: u32 = 256;
  pub(super) struct DeviceGenerator { /* private */ }
  impl DeviceGenerator {
      pub fn new(generator: &Generator, max_rows: usize, row_block: u32,
                 module: &Arc<CudaModule>, stream: Arc<CudaStream>) -> Result<Self>;
      pub fn sample_inputs(&mut self, d_seed: &CudaSlice<u8>, rows: usize, global_index: usize) -> Result<()>;
      pub fn forward(&mut self, rows: usize, dest: &mut CudaSlice<f32>, out_row_offset: usize) -> Result<()>;
      #[cfg(test)] pub fn read_inputs(&self, rows: usize) -> Result<(Vec<f32>, Option<Vec<f32>>)>;
  }
  // vector_search/mod.rs
  fn generate_vectors(seed: &[u8; 32], count: usize, index_base: usize, dest: &mut CudaSlice<f32>,
                      generator: &Generator, module: Arc<CudaModule>, stream: Arc<CudaStream>) -> Result<()>;
  fn generate_vectors_with(/* same, plus */ chunk: usize, row_block: u32) -> Result<()>;
  // scenarios.rs
  impl Scenario { pub const ALL: [Scenario; N] }     // generated by the `scenarios!` macro; 2 here, 3 after Task 8
  ```
  `read_inputs` returns latents as `rows × latent_dim` row-major, and the gate noise (pre-smoothing) as `rows × output_dim` when the architecture has one.

- [ ] **Step 1: Append the kernel to `kernels.cu`** (after `gan_linear`, before the audit section)

```c
// ---------------------------------------------------------------------------
// Row-wise generator steps. One thread owns whole rows and walks a row's
// coordinates in index order, so the result cannot depend on launch geometry.
//
// Unlike gan_linear these use sqrt and division, which --use_fast_math allows
// to be approximate. Verification is recall within a relative 1e-6, not a
// bit-exact integer, so a last-bit difference in a row does not by itself
// change a verdict. See docs/ai/specs/2026-09-21-multi-arch-gan-scenarios-design.md,
// "Determinism", for what this does and does not establish.
// ---------------------------------------------------------------------------

// x / max(||x||, eps), in place, for rows [row_offset, row_offset + n).
extern "C" __global__ void gan_row_normalize(
    float *data,
    const int n,
    const int dim,
    const float eps,
    const int row_offset
)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += blockDim.x * gridDim.x)
    {
        float *row = data + (long long)(row_offset + i) * dim;
        float ss = 0.0f;
        for (int j = 0; j < dim; ++j) {
            ss = fmaf(row[j], row[j], ss);
        }
        float norm = sqrtf(ss);
        if (norm < eps) {
            norm = eps;
        }
        for (int j = 0; j < dim; ++j) {
            row[j] = row[j] / norm;
        }
    }
}
```

- [ ] **Step 2: Rewrite `vector_search/generator.rs`**

```rust
//! GPU drivers: one forward pass per generator architecture, composed from
//! `gan_linear` and the row-wise kernels. Parsing and the CPU references live
//! in `crate::gan_generator`, outside the c004 gate.

use crate::gan_generator::{Generator, Layer};
use anyhow::{anyhow, Result};
use cudarc::driver::{safe::LaunchConfig, CudaFunction, CudaModule, CudaSlice, CudaStream, PushKernelArg};
use std::sync::Arc;

/// Threads per block for every row-wise kernel, `gan_sample_latents` included.
/// Output must not depend on it; `output_is_invariant_to_launch_geometry` checks.
pub(super) const ROW_BLOCK: u32 = 256;

struct DeviceLayer {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl DeviceLayer {
    fn upload(stream: &Arc<CudaStream>, layer: &Layer) -> Result<Self> {
        Ok(Self {
            weight: stream.memcpy_stod(&layer.weights)?,
            bias: stream.memcpy_stod(&layer.bias)?,
            in_dim: layer.in_dim,
            out_dim: layer.out_dim,
        })
    }
}

fn row_launch(rows: usize, row_block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((rows as u32 + row_block - 1) / row_block, 1, 1),
        block_dim: (row_block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One `gan_linear` launch. `block_dim` is fixed at (16,16) by the kernel's
/// tile constants and is not a tuning knob.
fn launch_linear(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    input: &CudaSlice<f32>,
    layer: &DeviceLayer,
    output: &mut CudaSlice<f32>,
    rows: usize,
    activate: bool,
    out_row_offset: usize,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32 + 127) / 128, (layer.out_dim as u32 + 63) / 64, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(input)
            .arg(&layer.weight)
            .arg(&layer.bias)
            .arg(output)
            .arg(&(rows as i32))
            .arg(&(layer.in_dim as i32))
            .arg(&(layer.out_dim as i32))
            .arg(&(activate as i32))
            .arg(&(out_row_offset as i32))
            .launch(cfg)?;
    }
    Ok(())
}

fn sample_latents(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    d_seed: &CudaSlice<u8>,
    latents: &mut CudaSlice<f32>,
    rows: usize,
    latent_dim: usize,
    index_offset: usize,
    row_block: u32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(d_seed)
            .arg(&(rows as i32))
            .arg(&(latent_dim as i32))
            .arg(latents)
            .arg(&(index_offset as i32))
            .launch(row_launch(rows, row_block))?;
    }
    Ok(())
}

struct MlpDevice {
    layers: Vec<DeviceLayer>,
    latents: CudaSlice<f32>,
    /// Hidden activations alternate between these; `forward` swaps them so
    /// the current input is always `a`.
    a: CudaSlice<f32>,
    b: CudaSlice<f32>,
    normalize: Option<(CudaFunction, f32)>,
}

enum DeviceArch {
    Mlp(MlpDevice),
}

pub(super) struct DeviceGenerator {
    stream: Arc<CudaStream>,
    sample_latents_kernel: CudaFunction,
    linear_kernel: CudaFunction,
    row_block: u32,
    max_rows: usize,
    arch: DeviceArch,
}

impl DeviceGenerator {
    pub fn new(
        generator: &Generator,
        max_rows: usize,
        row_block: u32,
        module: &Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> Result<Self> {
        let arch = match generator {
            Generator::Mlp(m) => {
                let widest = m.layers.iter().map(|l| l.out_dim).max().unwrap();
                let mut layers = Vec::with_capacity(m.layers.len());
                for layer in &m.layers {
                    layers.push(DeviceLayer::upload(&stream, layer)?);
                }
                DeviceArch::Mlp(MlpDevice {
                    layers,
                    latents: stream.alloc_zeros::<f32>(max_rows * generator.latent_dim())?,
                    a: stream.alloc_zeros::<f32>(max_rows * widest)?,
                    b: stream.alloc_zeros::<f32>(max_rows * widest)?,
                    normalize: match m.normalize_eps {
                        Some(eps) => Some((module.load_function("gan_row_normalize")?, eps)),
                        None => None,
                    },
                })
            }
            _ => return Err(anyhow!("no GPU driver for this generator architecture yet")),
        };
        Ok(Self {
            sample_latents_kernel: module.load_function("gan_sample_latents")?,
            linear_kernel: module.load_function("gan_linear")?,
            stream,
            row_block,
            max_rows,
            arch,
        })
    }

    /// Fill this chunk's random inputs. `global_index` is the instance-wide
    /// index of the chunk's first row; it enters the curand sequence, so a
    /// row's randomness does not depend on where chunk boundaries fall.
    pub fn sample_inputs(&mut self, d_seed: &CudaSlice<u8>, rows: usize, global_index: usize) -> Result<()> {
        if rows > self.max_rows {
            return Err(anyhow!("chunk of {} rows exceeds the {} this generator was sized for", rows, self.max_rows));
        }
        match &mut self.arch {
            DeviceArch::Mlp(m) => sample_latents(
                &self.stream, &self.sample_latents_kernel, d_seed, &mut m.latents,
                rows, m.layers[0].in_dim, global_index, self.row_block,
            ),
        }
    }

    /// Forward the sampled inputs into `dest` rows
    /// `[out_row_offset, out_row_offset + rows)`.
    pub fn forward(&mut self, rows: usize, dest: &mut CudaSlice<f32>, out_row_offset: usize) -> Result<()> {
        match &mut self.arch {
            DeviceArch::Mlp(m) => {
                let MlpDevice { layers, latents, a, b, normalize } = m;
                let count = layers.len();
                for (i, layer) in layers.iter().enumerate() {
                    let last = i + 1 == count;
                    let input: &CudaSlice<f32> = if i == 0 { latents } else { a };
                    if last {
                        launch_linear(&self.stream, &self.linear_kernel, input, layer, dest, rows, false, out_row_offset)?;
                    } else {
                        launch_linear(&self.stream, &self.linear_kernel, input, layer, b, rows, true, 0)?;
                        std::mem::swap(a, b);
                    }
                }
                if let Some((kernel, eps)) = normalize {
                    let dim = layers[count - 1].out_dim;
                    unsafe {
                        self.stream
                            .launch_builder(kernel)
                            .arg(dest)
                            .arg(&(rows as i32))
                            .arg(&(dim as i32))
                            .arg(&*eps)
                            .arg(&(out_row_offset as i32))
                            .launch(row_launch(rows, self.row_block))?;
                    }
                }
                Ok(())
            }
        }
    }

    #[cfg(test)]
    pub fn read_inputs(&self, rows: usize) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        match &self.arch {
            DeviceArch::Mlp(m) => {
                let dim = m.layers[0].in_dim;
                Ok((self.stream.memcpy_dtov(&m.latents.slice(0..rows * dim))?, None))
            }
        }
    }
}
```

The borrow in the layer loop: `input` borrows `a` immutably while `b` is borrowed mutably, which is fine because they are distinct fields obtained by destructuring. When `i == 0` and `count == 1`, `latents` goes straight to `dest`. If the borrow checker rejects `std::mem::swap(a, b)` while `input` is live, move the swap after the `if` so `input` is dead; do not reach for `unsafe`.

- [ ] **Step 3: Move `generate_vectors` onto it (`mod.rs`)**

Replace the whole function at lines 79-211 with:

```rust
/// Generate `count` vectors into `dest`.
///
/// `index_base` offsets the curand sequence so the database (base 0) and the
/// queries (base `database_size`) never share a row index.
fn generate_vectors(
    seed: &[u8; 32],
    count: usize,
    index_base: usize,
    dest: &mut CudaSlice<f32>,
    generator: &Generator,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
) -> Result<()> {
    generate_vectors_with(seed, count, index_base, dest, generator, module, stream, FORWARD_CHUNK, generator::ROW_BLOCK)
}

/// `generate_vectors` with the launch geometry exposed, so a test can show the
/// output does not depend on it.
fn generate_vectors_with(
    seed: &[u8; 32],
    count: usize,
    index_base: usize,
    dest: &mut CudaSlice<f32>,
    generator: &Generator,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    chunk: usize,
    row_block: u32,
) -> Result<()> {
    // Row indices reach the kernels as i32, and the spherical driver shifts
    // its second latent stream by 1 << 30. Both need this bound.
    if index_base + count >= (1usize << 30) {
        return Err(anyhow!("index_base {} + count {} must stay below 2^30", index_base, count));
    }
    let d_seed = stream.memcpy_stod(seed)?;
    let mut device = generator::DeviceGenerator::new(generator, chunk.min(count), row_block, &module, stream.clone())?;
    for chunk_start in (0..count).step_by(chunk) {
        let rows = chunk.min(count - chunk_start);
        device.sample_inputs(&d_seed, rows, index_base + chunk_start)?;
        device.forward(rows, dest, chunk_start)?;
    }
    Ok(())
}
```

Line 13 becomes `use crate::gan_generator::Generator;` (drop `weights_from` and `LATENT_DIM`). In `Database::generate` and `Challenge::for_nonce`, replace the `weights` / `layers` / `widest` locals with:

```rust
        let generator = Generator::from_blob(config.weights)?;
        let vector_dims = generator.output_dim();     // Database::generate only
```

keep the existing `vector_dims != config.vector_dims` error, and pass `&generator` where `layers, widest` were passed. In the two tests near lines 1697 and 1812, replace `let weights = weights_from(config.weights).unwrap(); let layers = …; let widest = …;` with `let generator = Generator::from_blob(config.weights).unwrap();` and the call's `layers, widest,` with `&generator,`. `dims` in those tests becomes `generator.output_dim()`.

In `scenarios.rs` the SIFT arm still says `weights: super::generator::V1_BLOB`; move that const into `scenarios.rs` itself:

```rust
const SIFT_128_BLOB: &[u8] = include_bytes!("weights/v1_sift.bin");
```

and delete the shim's `V1_BLOB`. One `include_bytes!` per file, referenced once, as the existing comment there requires.

- [ ] **Step 4: Prove the refactor changed nothing, on the box**

```bash
git add tig-challenges/src/vector_search/generator.rs tig-challenges/src/vector_search/mod.rs tig-challenges/src/vector_search/scenarios.rs tig-challenges/src/vector_search/kernels.cu
git status --short && git commit -m "refactor(c004): generate_vectors drives a DeviceGenerator; add gan_row_normalize" && git push origin HEAD
scripts/box_submit.sh task6-refactor gpu
```

Expected: the Task 0 baseline count + 34. The arithmetic: the baseline's 9 `vector_search::generator` tests are gone (8 moved into `gan_generator::v1` in Task 2, 1 lived in the shim this step deleted), and `gan_generator` now has 43, which include those 8. `the_database_is_identical_across_nonces…`, `for_nonce_keeps_the_index_base_offset` and `database_generate_keeps_the_index_base_at_zero` passing is the evidence that SIFT v1 output is unchanged. `every_scenario_blob_matches_its_declared_dims` is the shim test that went; Step 5 re-adds it in `scenarios.rs`. If the count is not baseline + 34, stop and account for the difference.

- [ ] **Step 5: Write the failing GloVe tests**

`scenarios.rs` tests, add:

```rust
    #[test]
    fn glove_wire_string_is_literal() {
        assert_eq!(Scenario::GLOVE_100.to_string(), "glove_100");
        assert_eq!(Scenario::from_str("glove_100").unwrap(), Scenario::GLOVE_100);
        assert_eq!(Scenario::from_str("GLOVE_100").unwrap(), Scenario::GLOVE_100);
    }

    #[test]
    fn every_scenario_blob_matches_its_declared_dims() {
        for scenario in Scenario::ALL {
            let config = ScenarioConfig::from(scenario);
            let generator = crate::gan_generator::Generator::from_blob(config.weights)
                .unwrap_or_else(|e| panic!("scenario {} has an unparseable blob: {}", scenario, e));
            assert_eq!(generator.output_dim(), config.vector_dims, "scenario {}", scenario);
            assert!(config.vector_dims as u32 <= super::super::AUDIT_MAX_DIMS,
                "scenario {} has {} dims but the audit kernel stages at most {}",
                scenario, config.vector_dims, super::super::AUDIT_MAX_DIMS);
            assert_eq!((config.n_queries, config.database_size), (7_000, 700_000), "scenario {}", scenario);
            assert_eq!((config.min_recall, config.recall_tolerance, config.audit_samples), (0.9, 1e-6, 1_000), "scenario {}", scenario);
        }
    }
```

(The third scenario test, `every_wire_name_round_trips`, is given in Step 7 with the macro it tests. `Scenario::ALL` does not exist until then, which is why Step 6 fails to compile.)

Change `scenario_from_str_rejects_unknown` here and `track_rejects_unknown_scenario` in `mod.rs` from `glove_300` to `deep_96` (both the input and the asserted substring). In `mod.rs`'s `track_tests` add:

```rust
    #[test]
    fn glove_track_uses_the_protocol_wire_form() {
        let encoded = serde_json::to_string(&Track { s: Scenario::GLOVE_100 }).unwrap();
        assert_eq!(encoded, r#""s=glove_100""#);
        let track: Track = serde_json::from_str(r#""s=glove_100""#).unwrap();
        assert_eq!(track.s, Scenario::GLOVE_100);
    }
```

Do **not** change `AUDIT_MAX_DIMS`'s visibility. `scenarios::tests` is a descendant of `vector_search`, and Rust lets descendants see an ancestor's private items, so `super::super::AUDIT_MAX_DIMS` already resolves. It must also stay written exactly as `const AUDIT_MAX_DIMS: u32 = …;` at the start of its line: `lib.rs`'s `rs_const` finds it with `trim().starts_with("const AUDIT_MAX_DIMS: u32 = ")` (lib.rs:288-293, verified), so a `pub(crate)` prefix would make `audit_constants_agree_between_kernels_cu_and_mod_rs` fail with "found 0".

In `mod.rs`'s `recall_audit_tests`, generalise `gpu_instance`: rename the body to `fn gpu_instance_for(scenario: Scenario, seed_byte: u8)`, use `Track { s: scenario }`, and keep `fn gpu_instance(seed_byte: u8)` as `gpu_instance_for(Scenario::SIFT_128, seed_byte)`. Add a context helper and the GPU tests:

```rust
    fn gpu_context() -> (Arc<CudaModule>, Arc<CudaStream>) {
        let ptx = Ptx::from_file(test_ptx_path().clone());
        let ctx = CudaContext::new(0).unwrap_or_else(|e| panic!("cannot open CUDA device 0: {}. These tests need a GPU and do not skip without one.", e));
        ctx.set_blocking_synchronize().unwrap();
        (ctx.load_module(ptx).unwrap(), ctx.default_stream())
    }

    /// GPU forward == CPU reference, on the inputs the GPU actually drew.
    /// Returns how many rows were compared (rows with a gate margin too close
    /// to zero are skipped; see the structured_gate caller).
    fn assert_gpu_matches_cpu(scenario: Scenario, skip_row: impl Fn(&Generator, &[f32], Option<&[f32]>) -> bool) -> usize {
        const ROWS: usize = 1024;
        let (module, stream) = gpu_context();
        let generator = Generator::from_blob(ScenarioConfig::from(scenario).weights).unwrap();
        let (latent_dim, out_dim) = (generator.latent_dim(), generator.output_dim());
        let d_seed = stream.memcpy_stod(&[7u8; 32]).unwrap();
        let mut device = generator::DeviceGenerator::new(&generator, ROWS, generator::ROW_BLOCK, &module, stream.clone()).unwrap();
        let mut dest = stream.alloc_zeros::<f32>(ROWS * out_dim).unwrap();
        device.sample_inputs(&d_seed, ROWS, 0).unwrap();
        device.forward(ROWS, &mut dest, 0).unwrap();
        stream.synchronize().unwrap();
        let got = stream.memcpy_dtov(&dest).unwrap();
        let (latents, noise) = device.read_inputs(ROWS).unwrap();

        let mut compared = 0;
        for row in 0..ROWS {
            let latent = &latents[row * latent_dim..(row + 1) * latent_dim];
            let row_noise = noise.as_ref().map(|n| &n[row * out_dim..(row + 1) * out_dim]);
            if skip_row(&generator, latent, row_noise) {
                continue;
            }
            let expected = generator.forward_cpu(latent, row_noise).unwrap();
            for j in 0..out_dim {
                let g = got[row * out_dim + j];
                assert!((g - expected[j]).abs() < 1e-5, "{} row {} coord {}: gpu {} cpu {}", scenario, row, j, g, expected[j]);
            }
            compared += 1;
        }
        compared
    }

    #[test]
    fn glove_gpu_forward_matches_the_cpu_reference() {
        assert_eq!(assert_gpu_matches_cpu(Scenario::GLOVE_100, |_, _, _| false), 1024);
    }

    fn generated_rows(scenario: Scenario, seed: [u8; 32], count: usize, chunk: usize, row_block: u32) -> Vec<f32> {
        let (module, stream) = gpu_context();
        let generator = Generator::from_blob(ScenarioConfig::from(scenario).weights).unwrap();
        let mut dest = stream.alloc_zeros::<f32>(count * generator.output_dim()).unwrap();
        generate_vectors_with(&seed, count, 0, &mut dest, &generator, module, stream.clone(), chunk, row_block).unwrap();
        stream.synchronize().unwrap();
        stream.memcpy_dtov(&dest).unwrap()
    }

    /// Bit-exact equality across the four geometries the 2026-08-25 spec used,
    /// and a different seed as the control that the comparison can fail.
    /// Over 200,000 rows, not the full 700,000: enough for four chunks at the
    /// smallest chunk size, and a third of the generation time.
    fn assert_invariant_to_launch_geometry(scenario: Scenario) {
        const COUNT: usize = 200_000;
        let reference = generated_rows(scenario, [3u8; 32], COUNT, 65_536, 256);
        for (chunk, block) in [(32_768, 256), (65_536, 128), (131_072, 512)] {
            let other = generated_rows(scenario, [3u8; 32], COUNT, chunk, block);
            let differing = reference.iter().zip(&other).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(differing, 0, "{}: chunk {} block {} changed {} values", scenario, chunk, block, differing);
        }
        let control = generated_rows(scenario, [4u8; 32], COUNT, 65_536, 256);
        assert!(reference.iter().zip(&control).any(|(a, b)| a.to_bits() != b.to_bits()), "a different seed must change the output");
    }

    #[test]
    fn glove_output_is_invariant_to_launch_geometry() {
        assert_invariant_to_launch_geometry(Scenario::GLOVE_100);
    }

    /// Every database row is unit-norm. Returns the rows for further checks.
    fn assert_database_rows_are_unit_norm(scenario: Scenario) -> (Vec<f32>, usize) {
        let (challenge, _module, stream, _prop) = gpu_instance_for(scenario, 11);
        let dims = challenge.vector_dims as usize;
        let rows = stream.memcpy_dtov(&challenge.d_database_vectors).unwrap();
        assert_eq!(rows.len(), 700_000 * dims);
        for (i, row) in rows.chunks_exact(dims).enumerate() {
            let norm = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "{} row {} has norm {}", scenario, i, norm);
        }
        (rows, dims)
    }

    #[test]
    fn glove_database_rows_are_unit_norm() {
        assert_database_rows_are_unit_norm(Scenario::GLOVE_100);
    }

    /// Same two probes the SIFT tests at mod.rs:1044-1066 use, on another scenario.
    fn assert_recall_probes(scenario: Scenario) {
        let (challenge, module, stream, prop) = gpu_instance_for(scenario, 1);
        let exact = brute_force_1nn(&challenge, module.clone(), stream.clone());
        let r = challenge.measure_recall(&exact, &[9u8; 32], module.clone(), stream.clone(), &prop).unwrap();
        assert_eq!(r, 1.0, "{}: the exact 1-NN must score recall 1.0", scenario);

        let zeros = Solution { indexes: vec![0; challenge.num_queries as usize] };
        let r = challenge.measure_recall(&zeros, &[9u8; 32], module, stream, &prop).unwrap();
        assert!(r < 0.01, "{}: all-zeros scored recall {}", scenario, r);
    }

    #[test]
    fn recall_probes_hold_on_glove() {
        assert_recall_probes(Scenario::GLOVE_100);
    }
```

`brute_force_1nn`, `measure_recall` and `Solution { indexes }` are the existing names (verified at mod.rs:1044-1066). That is 4 GPU tests in this step.

Mutations caught: `gpu_forward_matches` → `gan_row_normalize` wrong formula, wrong `out_row_offset` in either the last linear or the normalise, latents read at the wrong stride; `invariant_to_launch_geometry` → any dependence on chunking, including normalising rows `[0, n)` where `[row_offset, row_offset + n)` is meant (chunks after the first would stay un-normalised in one geometry and not another); `unit_norm` → normalise skipped for chunks after the first; `recall_probes_hold_on_glove` → the audit mishandling 100 dims (not a multiple of `AUDIT_KC = 16`): over-counting would break the all-zeros bound, under-counting the exact-1-NN equality.

- [ ] **Step 6: Push and see them fail to compile** (`Scenario::GLOVE_100` does not exist). Record the job id.

- [ ] **Step 7: Add the scenario**

`scenarios.rs`: add the variant `GLOVE_100`, the const and the arm:

```rust
const GLOVE_100_BLOB: &[u8] = include_bytes!("weights/glove_100_v1.bin");

            // GloVe v1, seed 42 (WGAN run probe_spectrum_seed42). Angular corpus:
            // rows are unit-normalised by the mlp driver, which makes Euclidean
            // 1-NN equal to angular 1-NN. min_recall is copied from SIFT and has
            // NOT been measured for this corpus; see the spec's Follow-ups.
            Scenario::GLOVE_100 => ScenarioConfig {
                n_queries: 7_000,
                database_size: 700_000,
                vector_dims: 100,
                weights: GLOVE_100_BLOB,
                min_recall: 0.9,
                recall_tolerance: 1e-6,
                audit_samples: 1_000,
            },
```

Then replace the hand-written `enum Scenario`, its `Display` impl and its `FromStr` impl with one macro invocation, so the variant list exists in exactly one place:

```rust
/// Generates the enum, `Scenario::ALL`, `Display` and `FromStr` from one list.
/// `ALL`'s length is computed from the same list the enum is built from, so a
/// variant cannot exist without being in `ALL`, and the wire name cannot drift
/// between `Display` and `FromStr`.
macro_rules! scenarios {
    ($($variant:ident => $wire:literal),+ $(,)?) => {
        /// One scenario per real embedding corpus. Tracks are one-to-one with
        /// scenarios, so adding a variant adds a track.
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        #[allow(non_camel_case_types)]
        pub enum Scenario { $($variant),+ }

        impl Scenario {
            pub const ALL: [Scenario; [$($wire),+].len()] = [$(Scenario::$variant),+];
        }

        impl std::fmt::Display for Scenario {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self { $(Scenario::$variant => write!(f, $wire)),+ }
            }
        }

        impl std::str::FromStr for Scenario {
            type Err = anyhow::Error;
            fn from_str(s: &str) -> Result<Self> {
                match s.to_lowercase().as_str() {
                    $($wire => Ok(Scenario::$variant),)+
                    _ => Err(anyhow!("Invalid scenario type: {}", s)),
                }
            }
        }
    };
}

scenarios! {
    SIFT_128 => "sift_128",
    GLOVE_100 => "glove_100",
}
```

Keep the existing doc comment's second paragraph about kernels, reworded: adding a variant is runtime-only **provided the architecture is one of the three the kernels support**. `From<Scenario> for ScenarioConfig` stays a hand-written `match` with no wildcard arm, so a new variant without a config is a compile error too. The error text `Invalid scenario type: {}` is unchanged, which `scenario_from_str_rejects_unknown` depends on.

The third scenario test checks the one thing the macro does not guarantee, that wire names are lowercase (`from_str` lowercases its input, so an uppercase wire literal would be unparseable):

```rust
    #[test]
    fn every_wire_name_round_trips() {
        for scenario in Scenario::ALL {
            assert_eq!(Scenario::from_str(&scenario.to_string()).unwrap(), scenario);
        }
    }
```

Here a round-trip IS the right assertion: the literal-string tests pin what the names are, and this one catches `SIFT_128 => "Sift_128"`, which `Display` would emit and `FromStr` could never match.

- [ ] **Step 8: Push, run, expect green**

```bash
scripts/box_submit.sh task6-glove gpu
```

Expected: Step 4's count + 3 (scenarios) + 1 (track) + 4 (GPU). Record the wall time of `glove_database_rows_are_unit_norm` as a rough generation-time signal (MEASURED, includes a 280 MB readback).

- [ ] **Step 9: Mutation-check on the box** (one job, three mutations, each in its own commit on a throwaway branch `mut/task6`, never pushed to the feature branch)

(a) In `gan_row_normalize` change `data + (long long)(row_offset + i) * dim` to `data + (long long)i * dim` → `glove_output_is_invariant_to_launch_geometry` and `glove_database_rows_are_unit_norm` must FAIL. (b) In `forward`, pass `0` as the last linear's `out_row_offset` → `unit_norm` must FAIL. (c) In `gan_row_normalize` change `row[j] / norm` to `row[j]` → `gpu_forward_matches` must FAIL. Submit with `box_submit.sh` from the throwaway branch (it must be pushed for the box to fetch it: push `mut/task6`, run, then `git push origin --delete mut/task6`). Record the three job ids and which tests failed.

- [ ] **Step 10: Commit** the scenario and tests on the feature branch (explicit paths, `git status --short` first), push.

---

### Task 7: Audit width, decided by measurement

**Files:**
- Modify: `tig-challenges/src/vector_search/kernels.cu:190` (`AUDIT_MAX_DIMS`), `mod.rs:76`, and `REF_MAX_DIMS` in the test-only `REFERENCE_1NN_KERNEL` string in `mod.rs`
- Possibly modify (fallback only): `kernels.cu` (new kernel), `mod.rs` `measure_recall_with_samples`, `lib.rs` agreement test

- [ ] **Step 1: Raise all three to 256** (`#define AUDIT_MAX_DIMS 256`, `const AUDIT_MAX_DIMS: u32 = 256;`, `#define REF_MAX_DIMS 256`). Update the shared-memory figure in the kernel comment from 27,720 to 36,936 bytes and say it was computed (`18×256×4 + 256×17×4 + 1,024 + 72`), not read from `ptxas`. Commit, push.

- [ ] **Step 2: Measure**

```bash
scripts/box_submit.sh task7-audit gpu recall_audit_tests
```

Read the ms that `audit_is_much_cheaper_than_a_naive_solve` prints. Run the job three times and take all three values (the kernel comment says the spread between runs is clock-driven).

- [ ] **Step 3: Decide, by this rule**

- All three runs under 150 ms and every `recall_audit_tests` test green → keep 256. Record baseline ms (Task 0) and the three new values in the kernel comment as MEASURED, with the GPU model (RTX 3060 Ti; the existing table was taken on an RTX 3060, so say the two are not directly comparable). Commit. Done.
- Any run at or over 150 ms → revert Step 1 for `AUDIT_MAX_DIMS` only (keep `REF_MAX_DIMS 256`), and do Step 4.

- [ ] **Step 4 (fallback only): `recall_audit_wide`**

Copy the whole `recall_audit` kernel to a new `extern "C" __global__ void recall_audit_wide(…)` with the same parameter list. In the copy, replace `AUDIT_MAX_DIMS` with `AUDIT_WIDE_MAX_DIMS` and `AUDIT_TQ` with `AUDIT_WIDE_TQ`, defined as:

```c
// The wide variant stages half as many queries at twice the width, so s_query
// is 9 x 256 x 4 = 9,216 bytes: the same as recall_audit's 18 x 128 x 4. The
// narrow kernel's tuned shared-memory footprint is therefore untouched.
#define AUDIT_WIDE_MAX_DIMS 256
#define AUDIT_WIDE_TQ 9
```

In `mod.rs` add `const AUDIT_WIDE_TQ: u32 = 9;` and `const AUDIT_WIDE_MAX_DIMS: u32 = 256;` (each at the start of its line with no visibility prefix, for the same `rs_const` reason as `AUDIT_MAX_DIMS`), and in `measure_recall_with_samples` select by width:

```rust
            let (kernel_name, tq, max_dims) = if self.vector_dims > AUDIT_MAX_DIMS {
                ("recall_audit_wide", AUDIT_WIDE_TQ, AUDIT_WIDE_MAX_DIMS)
            } else {
                ("recall_audit", AUDIT_TQ, AUDIT_MAX_DIMS)
            };
```

using `max_dims` in the existing dims guard at line 508, `kernel_name` at line 531 and `tq` in `grid_dim` at line 556. Extend the name list in `lib.rs`'s `audit_constants_agree_between_kernels_cu_and_mod_rs` with `"AUDIT_WIDE_TQ", "AUDIT_WIDE_MAX_DIMS"`, and change the scenario dims assertion from Task 6 to compare against `AUDIT_WIDE_MAX_DIMS`. Any `AUDIT_TQ`-derived arithmetic inside the kernel body (the comment block at `kernels.cu:258` onward lists the relationships) must use the wide constant in the copy: read that block before editing. Run `recall_audit_tests`; SIFT's ms must be back at the Task 0 baseline.

**Done when:** the decision and its three measurements are in the kernel comment and the suite is green. No new test here: NYTimes's recall tests in Task 8 are what exercise 256 dims.

---

### Task 8: Spherical kernels, driver, and the NYTimes scenario

**Files:** `kernels.cu` (two kernels), `vector_search/generator.rs` (`SphericalDevice`), `scenarios.rs` (`NYTIMES_256`, `ALL` grows to 3), `mod.rs` (tests).

**Interfaces:**
- Consumes: `launch_linear`, `sample_latents`, `row_launch`, `DeviceLayer` from Task 6; `crate::gan_generator::Spherical`.
- Produces: `DeviceArch::Spherical`; `Scenario::NYTIMES_256`. `read_inputs` returns each row's latent as `[z_t | z_s]`, the order `Spherical::forward_cpu` splits.

- [ ] **Step 1: Append the kernels** (after `gan_row_normalize`)

```c
// out = leaky(a * (1 + gamma) + beta), element-wise over `count` values.
extern "C" __global__ void gan_film_leaky(
    const float *a,
    const float *gamma,
    const float *beta,
    float *out,
    const int count
)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < count;
         i += blockDim.x * gridDim.x)
    {
        const float v = fmaf(a[i], 1.0f + gamma[i], beta[i]);
        out[i] = (v >= 0.0f) ? v : (v * 0.2f);
    }
}

// Spherical generator's last step, per row:
//   u = unit(d);  t = unit(v - (v.u) u);  out = cos_r * u + sin_r * t
// `d` and `v` are scratch and are overwritten with u and t.
extern "C" __global__ void gan_sphere_combine(
    float *d,
    float *v,
    float *out,
    const int n,
    const int dim,
    const float cos_r,
    const float sin_r,
    const float eps,
    const int out_row_offset
)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += blockDim.x * gridDim.x)
    {
        float *u = d + (long long)i * dim;
        float *t = v + (long long)i * dim;
        float *o = out + (long long)(out_row_offset + i) * dim;

        float ss = 0.0f;
        for (int j = 0; j < dim; ++j) ss = fmaf(u[j], u[j], ss);
        float norm = sqrtf(ss);
        if (norm < eps) norm = eps;
        for (int j = 0; j < dim; ++j) u[j] = u[j] / norm;

        float dot = 0.0f;
        for (int j = 0; j < dim; ++j) dot = fmaf(t[j], u[j], dot);
        for (int j = 0; j < dim; ++j) t[j] = fmaf(-dot, u[j], t[j]);

        ss = 0.0f;
        for (int j = 0; j < dim; ++j) ss = fmaf(t[j], t[j], ss);
        norm = sqrtf(ss);
        if (norm < eps) norm = eps;

        for (int j = 0; j < dim; ++j) {
            o[j] = fmaf(cos_r, u[j], sin_r * (t[j] / norm));
        }
    }
}
```

Every `fmaf` here has a matching `mul_add` in `Spherical::forward_cpu`, in the same operand order. Keep them in step if either changes.

- [ ] **Step 2: Write the failing tests** (`mod.rs`, GPU test module; and the wire/track tests as in Task 6 Step 5 with `nytimes_256` / `NYTIMES_256`)

```rust
    #[test]
    fn nytimes_gpu_forward_matches_the_cpu_reference() {
        assert_eq!(assert_gpu_matches_cpu(Scenario::NYTIMES_256, |_, _, _| false), 1024);
    }

    #[test]
    fn nytimes_output_is_invariant_to_launch_geometry() {
        assert_invariant_to_launch_geometry(Scenario::NYTIMES_256);
    }

    #[test]
    fn nytimes_database_rows_are_unit_norm() {
        assert_database_rows_are_unit_norm(Scenario::NYTIMES_256);
    }

    #[test]
    fn recall_probes_hold_on_nytimes() {
        assert_recall_probes(Scenario::NYTIMES_256);
    }

    /// Pearson correlation of two equal-length samples, in f64.
    fn correlation(x: &[f32], y: &[f32]) -> f64 {
        let n = x.len() as f64;
        let (mx, my) = (x.iter().map(|v| *v as f64).sum::<f64>() / n, y.iter().map(|v| *v as f64).sum::<f64>() / n);
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for (a, b) in x.iter().zip(y) {
            let (da, db) = (*a as f64 - mx, *b as f64 - my);
            sxy += da * db; sxx += da * da; syy += db * db;
        }
        sxy / (sxx * syy).sqrt()
    }

    fn column(data: &[f32], width: usize, offset: usize) -> Vec<f32> {
        data.chunks_exact(width).map(|row| row[offset]).collect()
    }

    /// The trunk and skip latents must be independent draws. If the 1 << 30
    /// offset were dropped, both calls would seed curand identically and
    /// column j of z_t would EQUAL column j of z_s: correlation 1.0.
    /// For independent columns over 65,536 rows the correlation has standard
    /// deviation 1/sqrt(65536) = 0.0039, so 0.05 is about 13 sigma.
    #[test]
    fn nytimes_trunk_and_skip_latents_are_independent() {
        const ROWS: usize = 65_536;
        let (module, stream) = gpu_context();
        let generator = Generator::from_blob(ScenarioConfig::from(Scenario::NYTIMES_256).weights).unwrap();
        let d_seed = stream.memcpy_stod(&[7u8; 32]).unwrap();
        let mut device = generator::DeviceGenerator::new(&generator, ROWS, generator::ROW_BLOCK, &module, stream.clone()).unwrap();
        device.sample_inputs(&d_seed, ROWS, 0).unwrap();
        stream.synchronize().unwrap();
        let (latents, _) = device.read_inputs(ROWS).unwrap();
        for j in 0..256 {
            let r = correlation(&column(&latents, 512, j), &column(&latents, 512, 256 + j));
            assert!(r.abs() < 0.05, "z_t column {j} correlates with z_s column {j}: r = {r}");
        }
    }
```

Mutations caught: `gpu_forward_matches` → any formula error in either kernel, or `tangent_in` fed `z_t`; `invariant` → the second latent call using a chunk-local index; `unit_norm` → `sin_r * t` without the normalise; independence → the offset dropped; recall tests → an audit overrun at 256 dims.

- [ ] **Step 3: Push, see the compile failure, record the job id.**

- [ ] **Step 4: Implement the driver** (`generator.rs`)

```rust
/// Shift applied to the second latent stream's curand index. `generate_vectors_with`
/// guarantees every row index is below this, so the two streams cannot meet.
const SKIP_LATENT_INDEX_SHIFT: usize = 1 << 30;

struct SphericalDevice {
    trunk: Vec<DeviceLayer>,
    direction: DeviceLayer,
    tangent_in: DeviceLayer,
    gamma: DeviceLayer,
    beta: DeviceLayer,
    tangent_out: DeviceLayer,
    film_kernel: CudaFunction,
    combine_kernel: CudaFunction,
    cos_r: f32,
    sin_r: f32,
    eps: f32,
    z_trunk: CudaSlice<f32>,
    z_skip: CudaSlice<f32>,
    /// Trunk activations alternate between h_a and h_b; after the trunk the
    /// result is in h_a.
    h_a: CudaSlice<f32>,
    h_b: CudaSlice<f32>,
    d: CudaSlice<f32>,       // direction(h), then u
    a: CudaSlice<f32>,       // tangent_in(z_skip), then the modulated value
    g: CudaSlice<f32>,       // gamma(h)
    b: CudaSlice<f32>,       // beta(h)
    v: CudaSlice<f32>,       // tangent_out(..), then t
}
```

`new` arm, for `Generator::Spherical(s)`: upload the six layers and the trunk; `widest = max trunk out_dim`; allocate `z_trunk` as `max_rows × s.trunk[0].in_dim`, `z_skip` as `max_rows × s.tangent_in.in_dim`, `h_a`/`h_b` as `max_rows × widest`, `d` and `v` as `max_rows × output_dim`, `a`/`g`/`b` as `max_rows × s.tangent_in.out_dim`; load `gan_film_leaky` and `gan_sphere_combine`.

`sample_inputs` arm:

```rust
            DeviceArch::Spherical(s) => {
                sample_latents(&self.stream, &self.sample_latents_kernel, d_seed, &mut s.z_trunk,
                    rows, s.trunk[0].in_dim, global_index, self.row_block)?;
                sample_latents(&self.stream, &self.sample_latents_kernel, d_seed, &mut s.z_skip,
                    rows, s.tangent_in.in_dim, global_index + SKIP_LATENT_INDEX_SHIFT, self.row_block)
            }
```

`forward` arm (destructure `s` into its fields first, as the mlp arm does):

```rust
                for (i, layer) in trunk.iter().enumerate() {
                    if i == 0 {
                        launch_linear(&self.stream, &self.linear_kernel, z_trunk, layer, h_a, rows, true, 0)?;
                    } else {
                        launch_linear(&self.stream, &self.linear_kernel, h_a, layer, h_b, rows, true, 0)?;
                        std::mem::swap(h_a, h_b);
                    }
                }
                launch_linear(&self.stream, &self.linear_kernel, h_a, direction, d, rows, false, 0)?;
                launch_linear(&self.stream, &self.linear_kernel, z_skip, tangent_in, a, rows, false, 0)?;
                launch_linear(&self.stream, &self.linear_kernel, h_a, gamma, g, rows, false, 0)?;
                launch_linear(&self.stream, &self.linear_kernel, h_a, beta, b, rows, false, 0)?;
                let width = tangent_in.out_dim;
                unsafe {
                    // Element-wise over rows * width values. `m` is a separate
                    // buffer because cudarc will not lend `a` as both input
                    // and output.
                    self.stream.launch_builder(film_kernel)
                        .arg(&*a).arg(&*g).arg(&*b).arg(&mut *m)
                        .arg(&((rows * width) as i32))
                        .launch(row_launch(rows * width, self.row_block))?;
                }
```

`m` is one more field on `SphericalDevice`: `m: CudaSlice<f32>, // the modulated value, max_rows x tangent_in.out_dim`. Add it to the struct and to the `new` arm's allocations. Then:

```rust
                launch_linear(&self.stream, &self.linear_kernel, m, tangent_out, v, rows, false, 0)?;
                let dim = direction.out_dim;
                unsafe {
                    self.stream.launch_builder(combine_kernel)
                        .arg(&mut *d).arg(&mut *v).arg(dest)
                        .arg(&(rows as i32)).arg(&(dim as i32))
                        .arg(&*cos_r).arg(&*sin_r).arg(&*eps)
                        .arg(&(out_row_offset as i32))
                        .launch(row_launch(rows, self.row_block))?;
                }
                Ok(())
```

`read_inputs` arm: read `z_trunk` and `z_skip` back and interleave per row into `[z_t | z_s]`:

```rust
            DeviceArch::Spherical(s) => {
                let (t, k) = (s.trunk[0].in_dim, s.tangent_in.in_dim);
                let zt = self.stream.memcpy_dtov(&s.z_trunk.slice(0..rows * t))?;
                let zs = self.stream.memcpy_dtov(&s.z_skip.slice(0..rows * k))?;
                let mut out = Vec::with_capacity(rows * (t + k));
                for row in 0..rows {
                    out.extend_from_slice(&zt[row * t..(row + 1) * t]);
                    out.extend_from_slice(&zs[row * k..(row + 1) * k]);
                }
                Ok((out, None))
            }
```

Remove the `_ => return Err(…)` fallthrough in `new` only when all three arms exist (Task 9); until then keep it.

- [ ] **Step 5: Add the scenario** (`scenarios.rs`): variant `NYTIMES_256`, `const NYTIMES_256_BLOB: &[u8] = include_bytes!("weights/nytimes_256_v3.bin");`, arm with `vector_dims: 256` and the same six other values, a comment naming the checkpoint (`v3_best`, `/workspace/nytimes-v3/v3_seed42/best_generator.pt`) and repeating that `min_recall` is unmeasured for this corpus; and one new line in the `scenarios!` invocation, `NYTIMES_256 => "nytimes_256",`, which extends the enum, `ALL`, `Display` and `FromStr` together.

- [ ] **Step 6: Push, run, expect green.** `scripts/box_submit.sh task8-nytimes gpu`. Expected: Task 6's final count + 7: the NYTimes wire test, the NYTimes track test, and the 5 GPU tests above. The dims loop covers the new variant without a new test. Record the wall time of `nytimes_database_rows_are_unit_norm`.

- [ ] **Step 7: Mutation-check on a throwaway branch `mut/task8`** (push, run, delete, as in Task 6 Step 9)

(a) `global_index + SKIP_LATENT_INDEX_SHIFT` → `global_index`: the independence test must FAIL with `r` near 1.0. (b) In `gan_sphere_combine` swap `cos_r` and `sin_r`: `gpu_forward_matches` must FAIL. (c) Delete the `t[j] = fmaf(-dot, …)` loop: `gpu_forward_matches` must FAIL. (d) In `gan_film_leaky` change `1.0f + gamma[i]` to `gamma[i]`: `gpu_forward_matches` must FAIL. Record job ids and failing tests.

- [ ] **Step 8: Commit and push** (explicit paths; `git status --short` first).

---

### Task 9: Structured-gate kernels, driver, and SIFT switches from v1 to v4

**Files:** `kernels.cu` (two kernels), `vector_search/generator.rs` (`GateDevice`), `scenarios.rs` (SIFT arm points at `sift_128_v4.bin`), `mod.rs` (tests).

**Interfaces:**
- Consumes: Task 6 helpers; `crate::gan_generator::StructuredGate` and its `gate_margin_cpu`.
- Produces: `DeviceArch::StructuredGate`. `read_inputs` returns `(latents, Some(raw_noise))`, where `raw_noise` is the pre-smoothing logistic noise: exactly the `gate_noise` argument of `forward_cpu`.

- [ ] **Step 1: Append the kernels** (after `gan_sphere_combine`)

```c
// Sequence base for the gate's noise stream. gan_sample_latents passes an
// int index as the curand sequence, so it cannot reach 2^40: the two streams
// are disjoint for the same seed word.
#define GATE_NOISE_SEQUENCE_BASE (1ULL << 40)

// Logistic noise for the structured gate: log(u) - log1p(-u), u uniform.
//
// curand_uniform returns (0, 1], and in float32 `1 - 1e-8` IS 1.0, so the
// PyTorch-style clamp(eps, 1 - eps) would let u = 1.0 through: log1p(-1) is
// -inf, the noise +inf, and after smoothing (a linear map) a NaN wherever it
// meets a -inf. At 89.6 million draws per database that is expected several
// times per instance, not a corner case. So the upper clamp is the largest
// float below 1.0, written out.
extern "C" __global__ void gan_gate_noise(
    const uint8_t *seed,
    const int n,
    const int dim,
    const float eps,
    float *noise,
    const int index_offset
)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += blockDim.x * gridDim.x)
    {
        const int global_i = index_offset + i;
        curandState state;
        curand_init(((const uint64_t *)(seed))[global_i % 4],
                    GATE_NOISE_SEQUENCE_BASE + (unsigned long long)global_i, 0, &state);
        float *row = noise + (long long)i * dim;
        for (int j = 0; j < dim; ++j) {
            float u = curand_uniform(&state);
            if (u > 0.99999994f) u = 0.99999994f;
            if (u < eps) u = eps;
            row[j] = logf(u) - log1pf(-u);
        }
    }
}

// The structured gate's last step, per row. `noise` is ALREADY smoothed.
//   logit_j = clamp * tanh((coupled_j + sparsity) / clamp)
//   open_j  = logit_j + noise_j > 0       (== sigmoid(./T) > 0.5 for any T > 0)
//   none open -> open only the first argmax of logit
//   m_j     = max(softplus(mag_pre_j), floor),  softplus(x) = x > 20 ? x : log1p(exp(x))
//   out     = unit(open * m)
extern "C" __global__ void gan_gate_apply(
    const float *mag_pre,
    const float *coupled,
    const float *sparsity,
    const float *noise,
    float *out,
    const int n,
    const int dim,
    const float logit_clamp,
    const float magnitude_floor,
    const float eps,
    const int out_row_offset
)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += blockDim.x * gridDim.x)
    {
        const float *m = mag_pre + (long long)i * dim;
        const float *c = coupled + (long long)i * dim;
        const float *z = noise + (long long)i * dim;
        const float s = sparsity[i];
        float *o = out + (long long)(out_row_offset + i) * dim;

        int best = 0;
        // 3.0e38 rather than INFINITY: --use_fast_math permits relaxations
        // around infinities (same choice as the audit kernel).
        float best_logit = -3.0e38f;
        float best_mag = 0.0f;
        int any_open = 0;
        float ss = 0.0f;
        for (int j = 0; j < dim; ++j) {
            const float logit = logit_clamp * tanhf((c[j] + s) / logit_clamp);
            float mag = (m[j] > 20.0f) ? m[j] : log1pf(expf(m[j]));
            if (mag < magnitude_floor) mag = magnitude_floor;
            if (logit > best_logit) {   // strict: keeps the FIRST maximum, as torch.argmax
                best_logit = logit;
                best = j;
                best_mag = mag;
            }
            const int open = (logit + z[j] > 0.0f) ? 1 : 0;
            const float value = open ? mag : 0.0f;
            o[j] = value;
            ss = fmaf(value, value, ss);
            any_open |= open;
        }
        if (!any_open) {
            o[best] = best_mag;
            ss = best_mag * best_mag;
        }
        float norm = sqrtf(ss);
        if (norm < eps) norm = eps;
        for (int j = 0; j < dim; ++j) {
            o[j] = o[j] / norm;
        }
    }
}
```

If `curand_uniform` does not exist in this toolchain's `curand_kernel.h`, the PTX build fails with an undeclared identifier. Then derive `u` from `curand(&state)` (a 32-bit unsigned draw) as `((float)(curand(&state) >> 8) + 0.5f) * (1.0f / 16777216.0f)`, which lies strictly inside (0, 1), and keep both clamps.

- [ ] **Step 2: Write the failing tests** (`mod.rs` GPU test module)

```rust
    /// Rows where some gate's margin is within 1e-4 of zero are skipped: the
    /// GPU's tanh and the CPU's differ in the last bits, a gate that close can
    /// legitimately fall either way, and a flipped gate changes the whole row.
    /// ESTIMATE (unverified): about 4 of 1024 rows. The assertion below fails
    /// if far more are skipped, so the skip cannot hide a broken kernel.
    #[test]
    fn sift_gpu_forward_matches_the_cpu_reference() {
        let compared = assert_gpu_matches_cpu(Scenario::SIFT_128, |generator, latent, noise| {
            let Generator::StructuredGate(s) = generator else { panic!("SIFT_128 should be structured_gate") };
            s.gate_margin_cpu(latent, noise.unwrap()).iter().any(|m| m.abs() < 1e-4)
        });
        assert!(compared >= 1000, "only {compared} of 1024 rows were comparable");
    }

    #[test]
    fn sift_output_is_invariant_to_launch_geometry() {
        assert_invariant_to_launch_geometry(Scenario::SIFT_128);
    }

    #[test]
    fn sift_database_rows_are_unit_norm_non_negative_and_sparse_like_sift() {
        let (rows, _dims) = assert_database_rows_are_unit_norm(Scenario::SIFT_128);
        assert!(rows.iter().all(|v| *v >= 0.0), "a SIFT coordinate is negative");
        let zero_fraction = rows.iter().filter(|v| **v == 0.0).count() as f64 / rows.len() as f64;
        // docs/datasets/sift.md (WGAN repo): v4's exact-zero fraction is 0.239,
        // real SIFT's 0.230. An inverted gate would give about 0.76.
        assert!((0.20..=0.28).contains(&zero_fraction), "exact-zero fraction is {zero_fraction}");
    }

    #[test]
    fn sift_gate_noise_is_finite_standard_logistic_and_independent_of_the_latents() {
        const ROWS: usize = 65_536;
        let (module, stream) = gpu_context();
        let generator = Generator::from_blob(ScenarioConfig::from(Scenario::SIFT_128).weights).unwrap();
        let d_seed = stream.memcpy_stod(&[7u8; 32]).unwrap();
        let mut device = generator::DeviceGenerator::new(&generator, ROWS, generator::ROW_BLOCK, &module, stream.clone()).unwrap();
        device.sample_inputs(&d_seed, ROWS, 0).unwrap();
        stream.synchronize().unwrap();
        let (latents, noise) = device.read_inputs(ROWS).unwrap();
        let noise = noise.expect("structured_gate has gate noise");

        assert!(noise.iter().all(|v| v.is_finite()), "gate noise contains inf or NaN");
        let n = noise.len() as f64;
        let mean = noise.iter().map(|v| *v as f64).sum::<f64>() / n;
        let var = noise.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n;
        // Standard logistic: mean 0, variance pi^2/3 = 3.2899. With 8.4 million
        // draws the standard error of the mean is 0.0006 and of the variance
        // about 0.002, so these bounds are many sigma wide and still reject
        // log(u) alone (mean -1) or a uniform left untransformed (variance 0.083).
        assert!(mean.abs() < 0.02, "gate noise mean is {mean}");
        assert!((var - 3.2899).abs() < 0.1, "gate noise variance is {var}");

        for j in 0..128 {
            let r = correlation(&column(&latents, 128, j), &column(&noise, 128, j));
            assert!(r.abs() < 0.05, "latent column {j} correlates with gate noise column {j}: r = {r}");
        }
    }

    /// The gate noise must not come from the curand state the latents use.
    /// Statistics cannot show this: Box-Muller maps its uniforms through a
    /// cosine, so a normal and the logit of the uniform behind it are
    /// uncorrelated even when they share a state. So compare directly against
    /// what the UNSHIFTED sequence produces, using a test-only kernel.
    #[test]
    fn sift_gate_noise_does_not_reuse_the_latent_sequence() {
        const ROWS: usize = 4096;
        let (module, stream) = gpu_context();
        let generator = Generator::from_blob(ScenarioConfig::from(Scenario::SIFT_128).weights).unwrap();
        let d_seed = stream.memcpy_stod(&[7u8; 32]).unwrap();
        let mut device = generator::DeviceGenerator::new(&generator, ROWS, generator::ROW_BLOCK, &module, stream.clone()).unwrap();
        device.sample_inputs(&d_seed, ROWS, 0).unwrap();
        let (_, noise) = device.read_inputs(ROWS).unwrap();
        let production = noise.unwrap();

        let kernel = module.load_function("test_gate_noise_unshifted").unwrap();
        let mut d_unshifted = stream.alloc_zeros::<f32>(ROWS * 128).unwrap();
        unsafe {
            stream.launch_builder(&kernel)
                .arg(&d_seed).arg(&(ROWS as i32)).arg(&128i32).arg(&1.0e-8f32)
                .arg(&mut d_unshifted).arg(&0i32)
                .launch(LaunchConfig { grid_dim: ((ROWS as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                .unwrap();
        }
        stream.synchronize().unwrap();
        let unshifted = stream.memcpy_dtov(&d_unshifted).unwrap();
        let equal = production.iter().zip(&unshifted).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        assert!(equal < production.len() / 100,
            "{} of {} gate-noise values equal the unshifted sequence's", equal, production.len());
    }
```

Append the test-only kernel to the `REFERENCE_1NN_KERNEL` string in `mod.rs` (it is concatenated into the test PTX after `kernels.cu`, so `curand_kernel.h` is already included). It is `gan_gate_noise` with the sequence base removed, and must otherwise stay textually identical to it:

```c
extern "C" __global__ void test_gate_noise_unshifted(
    const uint8_t *seed, const int n, const int dim, const float eps,
    float *noise, const int index_offset)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        const int global_i = index_offset + i;
        curandState state;
        curand_init(((const uint64_t *)(seed))[global_i % 4], global_i, 0, &state);
        float *row = noise + (long long)i * dim;
        for (int j = 0; j < dim; ++j) {
            float u = curand_uniform(&state);
            if (u > 0.99999994f) u = 0.99999994f;
            if (u < eps) u = eps;
            row[j] = logf(u) - log1pf(-u);
        }
    }
}
```

Mutations caught: `gpu_forward_matches` → any formula error in `gan_gate_apply`, `coupling` and `smoothing` swapped, `sparsity` indexed per coordinate where per row is meant; zero-fraction → inverted threshold, or `gate_temperature`-style scaling wrongly reintroduced; `does_not_reuse_the_latent_sequence` → the `2^40` base dropped (production noise would then equal the unshifted kernel's bit for bit); the statistics test → `log(u)` alone (mean −1) or an untransformed uniform (variance 0.083), and it does NOT catch a dropped base, for the Box-Muller reason in the test's comment; the upper clamp removed (infinite values appear with probability about 0.4 over 8.4 million draws, so the 700,000-row unit-norm test is the reliable catch: it sees about 5 expected occurrences).

- [ ] **Step 3: Push, see the failure** (no GPU driver for `StructuredGate` while the scenario still points at v1: the tests fail on the `let … else` panic). Record the job id.

- [ ] **Step 4: Implement the driver** (`generator.rs`)

```rust
struct GateDevice {
    trunk: Vec<DeviceLayer>,
    magnitude_head: DeviceLayer,
    gate_head: DeviceLayer,
    sparsity_head: DeviceLayer,
    coupling: DeviceLayer,
    smoothing: DeviceLayer,
    noise_kernel: CudaFunction,
    apply_kernel: CudaFunction,
    logit_clamp: f32,
    magnitude_floor: f32,
    eps: f32,
    latents: CudaSlice<f32>,
    h_a: CudaSlice<f32>,
    h_b: CudaSlice<f32>,
    mag_pre: CudaSlice<f32>,     // rows x output_dim
    gate: CudaSlice<f32>,        // rows x output_dim
    coupled: CudaSlice<f32>,     // rows x output_dim
    sparsity: CudaSlice<f32>,    // rows x 1
    raw_noise: CudaSlice<f32>,   // rows x output_dim, filled by sample_inputs
    smooth_noise: CudaSlice<f32>,
}
```

`new` arm: upload layers; allocate as commented, `h_a`/`h_b` at `max_rows × widest trunk out_dim`; load `gan_gate_noise` and `gan_gate_apply`. Replace the `_ => return Err(…)` fallthrough with this arm so the `match` is exhaustive.

`sample_inputs` arm:

```rust
            DeviceArch::StructuredGate(g) => {
                sample_latents(&self.stream, &self.sample_latents_kernel, d_seed, &mut g.latents,
                    rows, g.trunk[0].in_dim, global_index, self.row_block)?;
                let dim = g.magnitude_head.out_dim;
                unsafe {
                    self.stream.launch_builder(&g.noise_kernel)
                        .arg(d_seed).arg(&(rows as i32)).arg(&(dim as i32)).arg(&g.eps)
                        .arg(&mut g.raw_noise).arg(&(global_index as i32))
                        .launch(row_launch(rows, self.row_block))?;
                }
                Ok(())
            }
```

`forward` arm (after destructuring):

```rust
                for (i, layer) in trunk.iter().enumerate() {
                    if i == 0 {
                        launch_linear(&self.stream, &self.linear_kernel, latents, layer, h_a, rows, true, 0)?;
                    } else {
                        launch_linear(&self.stream, &self.linear_kernel, h_a, layer, h_b, rows, true, 0)?;
                        std::mem::swap(h_a, h_b);
                    }
                }
                launch_linear(&self.stream, &self.linear_kernel, h_a, magnitude_head, mag_pre, rows, false, 0)?;
                launch_linear(&self.stream, &self.linear_kernel, h_a, gate_head, gate, rows, false, 0)?;
                launch_linear(&self.stream, &self.linear_kernel, gate, coupling, coupled, rows, false, 0)?;
                launch_linear(&self.stream, &self.linear_kernel, h_a, sparsity_head, sparsity, rows, false, 0)?;
                launch_linear(&self.stream, &self.linear_kernel, raw_noise, smoothing, smooth_noise, rows, false, 0)?;
                let dim = magnitude_head.out_dim;
                unsafe {
                    self.stream.launch_builder(apply_kernel)
                        .arg(&*mag_pre).arg(&*coupled).arg(&*sparsity).arg(&*smooth_noise).arg(dest)
                        .arg(&(rows as i32)).arg(&(dim as i32))
                        .arg(&*logit_clamp).arg(&*magnitude_floor).arg(&*eps)
                        .arg(&(out_row_offset as i32))
                        .launch(row_launch(rows, self.row_block))?;
                }
                Ok(())
```

`read_inputs` arm: `Ok((dtov(latents[..rows*latent_dim]), Some(dtov(raw_noise[..rows*dim]))))`.

- [ ] **Step 5: Switch SIFT to v4** (`scenarios.rs`)

```rust
const SIFT_128_BLOB: &[u8] = include_bytes!("weights/sift_128_v4.bin");
```

and rewrite the SIFT arm's comment: SIFT v4, 100k retrain, `structured_gate`; rows are non-negative and unit-norm with about 24% exact zeros; `v1_sift.bin` is no longer wired to a scenario and survives only as the `TIGGAN01` test fixture in `gan_generator::v1`.

- [ ] **Step 6: Push, run the whole suite, expect green**

```bash
scripts/box_submit.sh task9-sift gpu
```

Every pre-existing SIFT test now runs against v4 instances. Expected outcomes to check, not assume: all `recall_audit_tests` green; `audit_is_much_cheaper_than_a_naive_solve` under 150 ms. A test that encoded a v1-specific value would fail here: if one does, read what it asserts before changing it, and report it. Record the wall time of the SIFT unit-norm test next to GloVe's and NYTimes's.

- [ ] **Step 7: Mutation-check on `mut/task9`**

(a) `GATE_NOISE_SEQUENCE_BASE + …` → plain `global_i`: `sift_gate_noise_does_not_reuse_the_latent_sequence` must FAIL with all values equal. (b) `logit + z[j] > 0.0f` → `< 0.0f`: the zero-fraction test and `gpu_forward_matches` must FAIL. (c) `const float s = sparsity[i];` → `sparsity[0]`: `gpu_forward_matches` must FAIL. (d) Swap the `coupling` and `smoothing` layers in `forward`: `gpu_forward_matches` must FAIL. (e) Remove the `if (u > 0.99999994f)` clamp: run `sift_database_rows_are_unit_norm…`; record whether it failed (expected: usually, not always; this is the one probabilistic mutation in the plan, and the recorded result is a fact about the seed, not a guarantee).

- [ ] **Step 8: Commit and push.**

---

### Task 10: Acceptance against the WGAN gates, and the measurements the spec owes

This is the test that the port samples the distribution the gates accepted. Tasks 3-9 pin arithmetic; only this pins the distribution.

**Files:**
- Modify: `tig-challenges/src/vector_search/mod.rs` (two `#[ignore]`d tests)
- Create: `docs/measurements/2026-09-21-c004-multi-arch-generators.md`

- [ ] **Step 1: Add the dump and timing tests**

```rust
    /// Not a test of anything: writes 50,000 generated rows per scenario as raw
    /// little-endian f32 for the WGAN repo's gate check. Run with --ignored.
    #[test]
    #[ignore]
    fn dump_rows_for_the_wgan_gates() {
        let dir = PathBuf::from(std::env::var("TIG_DUMP_DIR").expect("set TIG_DUMP_DIR"));
        std::fs::create_dir_all(&dir).unwrap();
        for scenario in Scenario::ALL {
            let rows = generated_rows(scenario, [42u8; 32], 50_000, 65_536, 256);
            let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
            let path = dir.join(format!("{}.f32", scenario));
            std::fs::write(&path, bytes).unwrap();
            println!("wrote {} ({} rows x {} dims)", path.display(), 50_000, rows.len() / 50_000);
        }
    }

    /// Prints Database::generate wall time per scenario, synchronised.
    #[test]
    #[ignore]
    fn print_generation_times() {
        let (module, stream) = gpu_context();
        let prop = get_device_prop(0).unwrap();
        for scenario in Scenario::ALL {
            for run in 0..3 {
                let start = std::time::Instant::now();
                let db = Database::generate(&[run as u8 + 1; 32], &Track { s: scenario }, module.clone(), stream.clone(), &prop).unwrap();
                stream.synchronize().unwrap();
                println!("GENERATION_MS scenario={} run={} ms={}", scenario, run, start.elapsed().as_millis());
                drop(db);
            }
        }
    }

    /// Counts SIFT gates whose margin is within 1e-5 of zero, on the CPU
    /// reference over 4,096 rows of GPU-drawn inputs.
    #[test]
    #[ignore]
    fn count_sift_gates_near_the_threshold() {
        const ROWS: usize = 4096;
        let (module, stream) = gpu_context();
        let generator = Generator::from_blob(ScenarioConfig::from(Scenario::SIFT_128).weights).unwrap();
        let Generator::StructuredGate(s) = &generator else { panic!() };
        let d_seed = stream.memcpy_stod(&[7u8; 32]).unwrap();
        let mut device = generator::DeviceGenerator::new(&generator, ROWS, generator::ROW_BLOCK, &module, stream.clone()).unwrap();
        device.sample_inputs(&d_seed, ROWS, 0).unwrap();
        stream.synchronize().unwrap();
        let (latents, noise) = device.read_inputs(ROWS).unwrap();
        let noise = noise.unwrap();
        let mut near = 0usize;
        for row in 0..ROWS {
            near += s.gate_margin_cpu(&latents[row * 128..(row + 1) * 128], &noise[row * 128..(row + 1) * 128])
                .iter().filter(|m| m.abs() < 1e-5).count();
        }
        println!("NEAR_THRESHOLD rows={} gates={} within_1e-5={}", ROWS, ROWS * 128, near);
    }
```

`TIG_DUMP_DIR` is already exported by `scripts/box_test.sh` (Task 0).

- [ ] **Step 2: Run them** (one job per test, so a failure in one does not hide the others)

```bash
for t in dump_rows_for_the_wgan_gates print_generation_times count_sift_gates_near_the_threshold; do
  BOX_ENV="TIG_TEST_EXTRA=--ignored" scripts/box_submit.sh "task10-$t" gpu "$t"
done
```

Collect from the job logs: three `GENERATION_MS` lines per scenario, the `NEAR_THRESHOLD` line, and the three dump paths.

Decision rule for the gate-noise stream's cost, which the spec left open. SIFT v4 has 1.11x GloVe's weights (1,937,153 floats against 1,743,460, computed from the blob shapes), so its generation time should be of that order plus the noise kernel. If SIFT's median `GENERATION_MS` exceeds 2x GloVe's, time `sample_inputs` alone (wrap it in the same `Instant` pattern inside `print_generation_times`) to see whether `curand_init` at sequence `2^40 + i` is the cost. If it is, switch `gan_gate_noise` to the spec's fallback: seed word `(global_i + 2) % 4` with the plain `global_i` sequence, re-run Task 9 Steps 6-7, and record both timings. If SIFT is within 2x, keep the scheme and record the numbers.

- [ ] **Step 3: Convert and gate-check**

```bash
ssh tig-gpu 'cd /workspace/tig-dumps && /venv/main/bin/python - <<PY
import numpy as np
for name, dims in [("sift_128", 128), ("nytimes_256", 256), ("glove_100", 100)]:
    a = np.fromfile(f"{name}.f32", dtype="<f4").reshape(-1, dims)
    assert a.shape[0] == 50000 and np.isfinite(a).all(), (name, a.shape)
    np.save(f"{name}.npy", a)
    print(name, a.shape, "norm min/max", np.linalg.norm(a, axis=1).min(), np.linalg.norm(a, axis=1).max())
PY'
```

Then, per dataset, run the WGAN repo's measurement **exactly as its dataset doc prescribes** for a synthetic series (the recipe under `## Noise floor` / the `## Gate` section of `docs/datasets/{glove,nytimes,sift}.md`), with the synthetic path pointing at the dump. The shape of the command, from `docs/datasets/glove.md`:

```bash
python -m src.eval.eda_report --real-path <the real file that doc names> \
    --synthetic-path tig=/workspace/tig-dumps/glove_100.npy \
    --output-dir runs/glove/tig_port --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10
python -m src.eval.check_gate --dataset glove --run-dir runs/glove/tig_port --stats-name tig
```

Run this step **on this machine**, in `~/TIG/wgan-synthetic` with its `.venv`. Verified 2026-09-21: the box's `/workspace/checkouts/wgan-synthetic/data/` holds only `README.md` and one script, and this machine's `data/` holds only the NYTimes files. So:

1. Stream the three `.npy` dumps here by the standing rule: `ssh tig-gpu 'cd /workspace/tig-dumps && sha256sum *.npy'`, then `ssh tig-gpu 'cd /workspace/tig-dumps && tar czf - sift_128.npy nytimes_256.npy glove_100.npy' | tar xzf - -C <scratch>`, then compare `sha256sum` locally. They are 25.6 MB, 51.2 MB and 20.0 MB (computed: 50,000 x dims x 4 + header).
2. Fetch the two missing real corpora with the WGAN repo's own fetcher (`data/README.md`): `.venv/bin/python -m src.data.fetch glove` and `… fetch sift`. Runtime and download size are unmeasured; run each in the background with a generous timeout and a log file. The real files the dataset docs name are `data/glove_250k.npy`, `data/sift_1m.npy` and, for NYTimes, the cleaned `data/nytimes_250k_l2_clean.npy`, which is already present.

`eda_report` requires `--real-path`, but the gate verdict for `--stats-name tig` reads only the synthetic series' statistics. The real file still has to be the documented one, because `eda_report` may derive shared preprocessing from it. Do **not** pass `--allow-condition-mismatch`: `check_gate` compares the run's N, k and nlist against the conditions pinned in the gate file (`n: 20000`, `k: 100`, `nlist: 256` in all three), and that refusal is the guard against measuring under the wrong conditions.

Expected: verdict `pass` on all four statistics for all three datasets. A `fail` is a finding, not a flake: report the statistic, the value, the band, and the accepted checkpoint's own value from the gate file's comments. The two likeliest causes are a latent distribution mismatch (`curand_normal` vs `torch.randn`: both standard normal, so unlikely) and, for SIFT, the gate noise scale.

- [ ] **Step 4: Write `docs/measurements/2026-09-21-c004-multi-arch-generators.md`**

Start with the claim table, then prose:

```
| claim | value | MEASURED / ESTIMATED | command or file that produced it | when |
```

Rows: test count before (Task 0) and after; SIFT audit ms before and after Task 7 (three runs each); generation ms per scenario (three runs each); near-threshold gate count over 524,288 gates, and its linear extrapolation to 89.6 million labelled `ESTIMATE (extrapolated from a MEASURED sample)`; the twelve gate statistics with verdicts; blob sizes and hashes. The spec's earlier estimate of "1–10 near-threshold coordinates per instance" is either confirmed or **retracted explicitly in the spec, in the Determinism section where it was made**, with the measured number beside it.

- [ ] **Step 5: Commit** the tests, script edits and measurement note.

---

### Task 11: Documentation and the final full run

**Files:**
- Modify: `tig-challenges/src/vector_search/README.md`
- Modify: `docs/ai/specs/2026-09-21-multi-arch-gan-scenarios-design.md` (append `## Validation`)
- Modify: `tig-challenges/src/vector_search/weights/PROVENANCE.md` (if anything changed since Task 1)

- [ ] **Step 1: README.** Update the tracks table to three rows with dims and generator architecture; correct the `min_recall` row, which still says 0.95 at lines 91 and 123-125 while `scenarios.rs` has said 0.9 since commit `82039c2` (verify with `grep -n "0.95" README.md` before editing); list the five new kernels beside `gan_linear` with one line each; state that algorithms must read `challenge.vector_dims` at runtime because dims are now 100, 128 and 256 within one deployment; add this spec as item 4 of the reading order and renumber.

- [ ] **Step 2: Spec `## Validation`.** Copy the claim table from the measurement note. State what was and was not established, in the 2026-08-25 spec's form: launch-geometry invariance on sm_86 (shown), gate acceptance (shown or not, per Task 10), cross-architecture bit-exactness (not shown; no second architecture was available).

- [ ] **Step 3: Final runs, from a clean tree at the final commit**

```bash
git status --short                                   # expect nothing
cargo test -p tig-challenges 2>&1 | tail -5          # local, ungated: gan_generator + audit_sampling
~/TIG/wgan-synthetic/.venv/bin/python -m pytest scripts/tests/test_export_generator.py -q
git push origin HEAD && scripts/box_submit.sh final gpu
```

Report the three results with their raw tails. "Tests pass" is reported with the counts and the job id, not as a bare claim.

- [ ] **Step 4: Commit and push.** Then use `superpowers:finishing-a-development-branch`.

---

## Follow-ups this plan does not do (from the spec)

`_MIN_RECALL_BY_TRACK` in `tig-pentesting`; the protocol config's track list; a per-corpus d2/d1 measurement for `min_recall`; `calc_build_fuel_budget` at 100 and 256 dims; qualifier economics; a second-architecture digest comparison; whether `v1_sift.bin` should leave the repo entirely.
