<div align="center">

# PEARL: **P**er-layer **E**nergy **A**ttribution and **R**econstruction for Large **L**anguage Models

*How many joules does each layer of a language model cost, at every generation step?*

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C)
![GPU](https://img.shields.io/badge/NVIDIA-NVML-76B900)
![Tests](https://img.shields.io/badge/unit%20tests-CPU%20only-success)
![License](https://img.shields.io/badge/license-review%20only-lightgrey)

**Supplementary code for an anonymous submission. Confidential: for review only (see [License](#license)).**

</div>

---

## In one paragraph

GPU power sensors report every few tens of milliseconds, but a single layer of a language model runs for well under one millisecond. PEARL closes that gap without running layers in isolation. It measures **inside complete `generate()` calls**: hooks on every leaf module record their execution windows, and a 10 ms NVML sampler runs on the same clock. Afterwards, PEARL attributes the measured energy to each layer, separately for the prefill and for **every decode step**. 

## What this repository gives you

| | Pipeline | What it produces |
|---|---|---|
| 🔬 | **1. Sensor characterization** (`pearl.sensor`) | The behaviour of all five NVML power readings on *your* GPU: update period, averaging window, lag, pulse coverage, rise time and sampler stalls. Output is tables plus figures. |
| ⚡ | **2a. Collection** (`pearl.collect`) | Hooked LLM inference under NVML sampling, over a grid of models × batch sizes × prompt lengths × new tokens × precisions. Output is compact Parquet traces. |
| 📊 | **2b. Dataset** (`pearl.dataset`) | One row per (leaf module, generation step), with the `aligned_end` energy and 13 analytical features (MACs, bytes, operational intensity, KV cache…). Output is a single Parquet file. |

The paper grid ([configs/collect_paper.yaml](configs/collect_paper.yaml)) covers 11 open models from 0.6 B to 8 B parameters (Qwen3, Llama 3.x, Gemma 3, Phi-4-mini, Mistral). It uses batch sizes 1–128, prompts of 16–8192 tokens, 16–128 new tokens, and fp16 / bf16 / fp32.

## Quick start

```bash
pip install -r requirements.txt                                          # after installing PyTorch for your CUDA
python -m pearl.collect.run configs/collect_smoke.yaml --out results/smoke   # Qwen3-0.6B, one grid point
python -m pearl.dataset.build results/smoke                                # -> fine_firings_aligned_end.parquet
python -m pearl.dataset.attention_core results/smoke                       # -> fine_attention_core_aligned_end.parquet
```

Here is what you get: real rows from that smoke run, the prefill of one attention block of Qwen3-0.6B, with energy averaged over 20 measured `generate()` calls:

| `layer_name` | `role` | `macs` | `energy_aligned_end_j` |
|---|---|---:|---:|
| `model.layers.5.self_attn.q_proj` | q_proj | 67.1 M | 0.0068 |
| `model.layers.5.self_attn.q_norm` | norm | 0 | 0.0182 |
| `model.layers.5.self_attn.k_proj` | k_proj | 33.6 M | 0.0068 |
| `model.layers.5.self_attn.k_norm` | norm | 0 | 0.0177 |
| `model.layers.5.self_attn.v_proj` | v_proj | 33.6 M | 0.0051 |
| `model.layers.5.self_attn.attention_core` | attention_core | 4.2 M | **0.0359** |
| `model.layers.5.self_attn.o_proj` | o_proj | 67.1 M | 0.0095 |

On this run, the per-layer energies add up to 99.7% of the energy measured over the whole window.

## Repository map

```text
pearl/
  sensor/    probe.py · report.py · sources.py · load.py · sampler_check.py · stats.py
  collect/   run.py · sampler.py · hooks.py · workload.py · traces.py
  dataset/   build.py · attention_core.py · attribution.py · features.py · structure.py
configs/     collect_paper.yaml (full grid) · collect_smoke.yaml (one point)
tests/       CPU-only unit tests: attribution, attention core, stats, traces, features
```

Model training and the paper's result tables are not part of this release.

## Requirements

- Linux, an NVIDIA GPU, and a driver exposing NVML (`nvmlDeviceGetSamples` is required for pipeline 2a).
- Python ≥ 3.10.
- Install PyTorch with the wheel that matches your CUDA driver (see pytorch.org), then:

```bash
pip install -r requirements.txt
```

Gated models (Llama, Gemma) need a Hugging Face token (`huggingface-cli login`).

All commands are run from the repository root. The unit tests run on CPU only:

```bash
python -m pytest tests
```

## Pipeline 1: sensor characterization

```bash
python -m pearl.sensor.probe --gpu 0 --out results/sensor          # full suite, ~15 min
python -m pearl.sensor.probe --gpu 0 --out results/sensor --quick  # smoke test, a few minutes
python -m pearl.sensor.report results/sensor --out results/sensor_report
```

`probe` runs the following steps in order. `--steps` selects a subset, e.g. `--steps idle_10ms,square_5ms`.

| Step | What it does |
|---|---|
| `idle_{10,2}ms` | Polls all five power sources (usage, instant, average, samples, energy counter) on an idle GPU |
| `load_{10,2}ms` | Same, under a constant 8192×8192 fp16 matmul load |
| `square_5ms` | On/off square-wave load with several periods, for window, lag and pulse capture |
| `sampler_{usage,samples}` | Background-poller timing: call durations and stalls |

Output: `<out>/<gpu>_<driver>_<timestamp>/` contains `campaign.json` plus `probe/<step>/` and `sampler_check/<source>/`, each with Parquet files and a `meta.json`.

`report` writes one subfolder per (GPU, driver), containing:

- `sensor_tables.json` with three tables:
  - **S1**: call support and update period per source and condition.
  - **S2**: box-car window, lag, pulse capture window, rise time and coverage per source, plus energy-counter vs. usage ratios.
  - **S3**: sampler call durations and stalls.
- Figures `s1_raw_traces.png`, `s2_<source>.png`, `s2_coverage.png` and `s3_sampler_timing.png`.

## Pipeline 2a: per-layer energy collection

```bash
python -m pearl.collect.run configs/collect_smoke.yaml --out results/collect_smoke  # 1 grid point, minutes
python -m pearl.collect.run configs/collect_paper.yaml                              # full paper grid
```

The protocol, set in the `protocol:` block of the config:

1. Record an idle power baseline.
2. For each grid point (models × batch sizes × prompt lengths × new tokens × precisions):
   - Load the model.
   - Run warm-up `generate()` calls outside the measured window.
   - Run `forwards_per_measure` consecutive `generate()` calls while an NVML thread reads `nvmlDeviceGetSamples` every 10 ms.
   - Every leaf module carries forward hooks that record `start_ns` / `end_ns` with `time.monotonic_ns()`, the same clock as the power samples.
   - Power is also kept for 150 ms before and after the window.
3. Grid points that fail (e.g. out of memory) are logged and skipped, and the model is unloaded before the next one.

Output layout:

```text
<out>/<YYYY-mm-dd_HH-MM-SS>/layer_energy/
  layer_energy_campaign_config.json
  traces/idle_power.parquet
  traces/<combo>/{layers,segments,power_trace,firings}.parquet
```

## Pipeline 2b: dataset build

```bash
python -m pearl.dataset.build results/collect_smoke            # -> results/collect_smoke/fine_firings_aligned_end.parquet
python -m pearl.dataset.build results/collect --workers 8 --out fine.parquet
```

Options:

- `--sessions` restricts the build to some session folders.
- `--max-runs N` builds a small test file.
- `--overwrite` replaces an existing output.

If the same (model, config) appears in several sessions, the latest session is kept. Each run becomes one Parquet row group. A sidecar file `<stem>.meta.json` records the schema version, the runs included, the runs that failed (with the error) and the units.

### Schema (`fine_firings_aligned_end_v1`)

There is one row per leaf-module firing position in a `generate()` call. Each value is averaged over the `forward_passes` measured calls.

| Group | Columns | Type |
|---|---|---|
| Identity | `model_id`, `config_tag`, `session`, `phase` (`prefill`/`decode`), `layer_name`, `role`, `layer_type` | string |
| | `firing_idx` (1-based position in the call), `layer_idx`, `decode_step` (0 = prefill, then 1…T−1) | int32 |
| Config | `batch_size`, `prompt_len`, `max_new_tokens`, `dtype_bytes`, `forward_passes` | int32 |
| | `precision` | string |
| | `n_params` | int64 |
| Analytical | `macs`, `FLOPs_analytical`, `attention_core_macs`, `kv_cache_bytes`, `Bytes_input`, `Bytes_output`, `Bytes_weights`, `Bytes_total`, `dram_bytes`, `operational_intensity`, `n_tokens`, `q_len`, `context_len` | float32 |
| Targets | `energy_aligned_end_j` (J per call), `duration_s`, `measured_full_forward_j` (J per call) | float32 |

How the analytical columns are computed:

- **Matmul leaves** get `in × out × B × q_len` MACs.
- **Other leaves** get 0 MACs, and their activation bytes come from their traced width.
- **Attention core MACs and KV-cache bytes** are stored on each block's `o_proj` row and are NaN elsewhere.
- **`q_len` and `context_len`**: `q_len = context_len = S` at prefill; at decode step t, `q_len = 1` and `context_len = S + t`.

### Attention-core leaf

Some attention work has no module hook: applying rotary embeddings, updating the KV cache, and scaled dot-product attention. Under `aligned_end`, its energy falls into the window of the last hooked `self_attn` leaf before `o_proj`, called the *pre-core* leaf. That leaf is `v_proj` for Llama, Qwen3 and Mistral, `k_norm` for Gemma-3, and `qkv_proj` for Phi-4-mini.

A second step, run after the build, splits that energy out:

```bash
python -m pearl.dataset.attention_core results/collect   # -> results/collect/fine_attention_core_aligned_end.parquet
```

What the step does:

- It inserts an `attention_core` firing on the measured gap `[end_ns(pre-core), start_ns(o_proj)]` of every layer, then re-attributes.
- The pre-core leaf keeps only its own `[start_ns, end_ns]` window, and every other window is unchanged.
- The core row's `macs` is the attention score and context MAC count (`attention_core_macs`). Its bytes are Q, K and V for the new tokens, the context written back, and the KV cache. Its `layer_idx` is `10000 + layer number`.

The output schema is the fine schema plus two columns:

- `energy_aligned_end_ref_j`: the energy before the split (NaN on core rows).
- `is_pre_core` (bool).

A run is kept only if it passes two gates (relative tolerance 1e-4), and failures are listed in the sidecar:

1. Every row other than the pre-core leaf matches the reference parquet.
2. For each layer and step, the new pre-core energy plus `attention_core` equals the reference pre-core energy.

## Methodology notes

- **NVML is not a calibrated power meter.** Its readings are averaged over a window, lag behind the actual power, and update at a period that depends on the device and the driver. Pipeline 1 measures these properties for your device.
- **`aligned_end` attribution.** Each NVML reading `P_j` is timestamped at the end of the period it averages. So the reading is held backwards over `(t_{j-1}, t_j]`, and the resulting step function is integrated over each firing window. Duplicate timestamps keep the last reading. A firing's window spans from its own start to the start of the next firing, so gaps between kernels are attributed too. The last firing of a call ends at its own `end_ns`.
- **`measured_full_forward_j`** is the trapezoid integral of power over the measured window, divided by `forward_passes`.
- **Measured window.** Model loading and warm-up happen outside the measured window, and CUDA is synchronized at both of its edges.

## License

Review only: this code is confidential supplementary material for the anonymous review of the submission. It may not be copied, redistributed, or reused outside the review, and every copy must be deleted when the review ends (see `LICENSE`).
