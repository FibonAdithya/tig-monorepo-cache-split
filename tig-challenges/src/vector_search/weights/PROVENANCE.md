# TIGGAN02 weight blob provenance

All values below are MEASURED in this session on 2026-09-21 by the command shown
next to each. WGAN repo commit used to build the generator classes and run
`export_generator_weights.py` / `dump_golden_vectors.py`:
`5b96976401b6df744ab8f8abe4f4bfc3667e5539` (MEASURED: `git -C ~/TIG/wgan-synthetic rev-parse HEAD`).

## glove_100_v1.bin (arch: mlp)

| field | value |
|---|---|
| blob file | `tig-challenges/src/vector_search/weights/glove_100_v1.bin` |
| blob bytes | 6,973,936 (MEASURED: `ls -l tig-challenges/src/vector_search/weights/glove_100_v1.bin`) |
| blob sha256 | `4f8a5fcef01bc8c44e3bf892808a28f0e25a7de9c1dce0411a18baff3904029c` (MEASURED: `sha256sum tig-challenges/src/vector_search/weights/glove_100_v1.bin`) |
| checkpoint path on box | `/workspace/glove-probes/probe_spectrum_seed42/best_generator.pt` (tig-gpu, `cd /workspace`) |
| checkpoint sha256 | `38d8013a748c83b02f4e4aa99bb64fcda221cbc14e4ac8bd9c8f20404b66d7a3` (MEASURED: `sha256sum` after streaming via `ssh tig-gpu 'tar czf - ...' \| tar xzf - -C $D`) |
| `generator_weights` (checkpoint field) | `live` (step 5000) (MEASURED: `torch.load(...).get('generator_weights')` / `.get('step')`, see Step 2 output below) |
| exporter command | `$PY scripts/export_generator_weights.py $D/glove-probes/probe_spectrum_seed42/best_generator.pt tig-challenges/src/vector_search/weights/glove_100_v1.bin --arch mlp --run-config $D/glove-probes/probe_spectrum_seed42/run_config.yaml --wgan-repo ~/TIG/wgan-synthetic` |
| golden vectors command | `$PY scripts/dump_golden_vectors.py $D/glove-probes/probe_spectrum_seed42/best_generator.pt tig-challenges/src/vector_search/weights/glove_100_v1.golden.json --arch mlp --run-config $D/glove-probes/probe_spectrum_seed42/run_config.yaml --wgan-repo ~/TIG/wgan-synthetic` |

## nytimes_256_v3.bin (arch: spherical)

| field | value |
|---|---|
| blob file | `tig-challenges/src/vector_search/weights/nytimes_256_v3.bin` |
| blob bytes | 13,124,768 (MEASURED: `ls -l tig-challenges/src/vector_search/weights/nytimes_256_v3.bin`) |
| blob sha256 | `28350e24698eb8d9e8e965fcf50bfad03c482407d81ed2fcb87d13295246119e` (MEASURED: `sha256sum tig-challenges/src/vector_search/weights/nytimes_256_v3.bin`) |
| checkpoint path on box | `/workspace/nytimes-v3/v3_seed42/best_generator.pt` (tig-gpu, `cd /workspace`) |
| checkpoint sha256 | `b1acfdda41795d310ea384d75e00c975073ce47ffbe58e551c3e5b4727ee363a` (MEASURED: `sha256sum` after streaming) |
| `generator_weights` (checkpoint field) | `live` (step 9000) (MEASURED: `torch.load(...).get('generator_weights')` / `.get('step')`, see Step 2 output below) |
| exporter command | `$PY scripts/export_generator_weights.py $D/nytimes-v3/v3_seed42/best_generator.pt tig-challenges/src/vector_search/weights/nytimes_256_v3.bin --arch spherical --run-config $D/nytimes-v3/v3_seed42/run_config.yaml --wgan-repo ~/TIG/wgan-synthetic` |
| golden vectors command | `$PY scripts/dump_golden_vectors.py $D/nytimes-v3/v3_seed42/best_generator.pt tig-challenges/src/vector_search/weights/nytimes_256_v3.golden.json --arch spherical --run-config $D/nytimes-v3/v3_seed42/run_config.yaml --wgan-repo ~/TIG/wgan-synthetic` |

## sift_128_v4.bin (arch: structured_gate)

| field | value |
|---|---|
| blob file | `tig-challenges/src/vector_search/weights/sift_128_v4.bin` |
| blob bytes | 7,748,764 (MEASURED: `ls -l tig-challenges/src/vector_search/weights/sift_128_v4.bin`) |
| blob sha256 | `bbf31b28871b508cbb8f8f67c2d90e45c288bee482352e84421648703fda83b5` (MEASURED: `sha256sum tig-challenges/src/vector_search/weights/sift_128_v4.bin`) |
| checkpoint path on box | `/workspace/sift-v4/v4_sift1m_x100k/best_generator.pt` (tig-gpu, `cd /workspace`) |
| checkpoint sha256 | `09eaac8fd84d7cd4c96fd2ddbcbeb967d55d422fecfa7d26f0a34590879a1e6c` (MEASURED: `sha256sum` after streaming) |
| `generator_weights` (checkpoint field) | `ema` (step 86000) (MEASURED: `torch.load(...).get('generator_weights')` / `.get('step')`, see Step 2 output below) |
| exporter command | `$PY scripts/export_generator_weights.py $D/sift-v4/v4_sift1m_x100k/best_generator.pt tig-challenges/src/vector_search/weights/sift_128_v4.bin --arch structured_gate --run-config $D/sift-v4/v4_sift1m_x100k/run_config.yaml --wgan-repo ~/TIG/wgan-synthetic` |
| golden vectors command | `$PY scripts/dump_golden_vectors.py $D/sift-v4/v4_sift1m_x100k/best_generator.pt tig-challenges/src/vector_search/weights/sift_128_v4.golden.json --arch structured_gate --run-config $D/sift-v4/v4_sift1m_x100k/run_config.yaml --wgan-repo ~/TIG/wgan-synthetic` |

## Step 2 output (verbatim, MEASURED)

Command:
```
$PY -c "import torch,sys; c=torch.load(sys.argv[1]+'/best_generator.pt',map_location='cpu',weights_only=False); print(sys.argv[1].split('/')[-1], c.get('generator_weights'), c.get('step'), sorted(k for k in c if 'state' in k))" $f
```
for each of `$D/sift-v4/v4_sift1m_x100k`, `$D/nytimes-v3/v3_seed42`, `$D/glove-probes/probe_spectrum_seed42`:

```
v4_sift1m_x100k ema 86000 ['critic_state_dict', 'generator_state_dict', 'optim_d_state_dict', 'optim_g_state_dict']
v3_seed42 live 9000 ['critic_state_dict', 'generator_state_dict', 'optim_d_state_dict', 'optim_g_state_dict']
probe_spectrum_seed42 live 5000 ['critic_state_dict', 'generator_state_dict', 'optim_d_state_dict', 'optim_g_state_dict']
```

Whatever `generator_weights` says (`ema` or `live`), the exported key is always
`generator_state_dict` — that is the key `src/sample/generate.py:57` loads in
the WGAN repo, so it is what the gates measured. No key switching was done.

## Notes

- `PY` above is `~/TIG/wgan-synthetic/.venv/bin/python`; `$D` was this session's
  scratchpad checkpoint directory; `run_config.yaml` files came from the run
  directories on `tig-gpu` (all three were present, no fallback to the WGAN
  repo's `configs/` was needed).
- Each blob's total size exceeds the pure tensor-data byte count (6,973,840 /
  13,124,608 / 7,748,612 respectively) by a header of 96 / 160 / 152 bytes
  (magic + arch/latent/out/scalar-count/scalars/tensor-count, plus an 8-byte
  rows/cols prefix per tensor) — all under the 200-byte budget.
