<h1 align="center">Stormlog</h1>

<h3 align="center">
Memory and performance profiling for ML training and inference
</h3>

<p align="center">
| <a href="https://stormlog.readthedocs.io/en/latest/"><b>Documentation</b></a>
| <a href="https://stormlog.readthedocs.io/en/latest/usage.html"><b>Usage guide</b></a>
| <a href="https://stormlog.readthedocs.io/en/latest/inference.html"><b>Inference profiling</b></a>
| <a href="https://stormlog.readthedocs.io/en/latest/tui.html"><b>Terminal UI</b></a>
| <a href="https://pypi.org/project/stormlog/"><b>PyPI</b></a> |
</p>

<p align="center">
<a href="https://github.com/Silas-Asamoah/stormlog/actions/workflows/ci.yml"><img src="https://github.com/Silas-Asamoah/stormlog/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
<a href="https://pypi.org/project/stormlog/"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fstormlog%2Fjson&query=%24.info.version&label=pypi" alt="PyPI version"></a>
<a href="https://stormlog.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/stormlog/badge/?version=latest" alt="Documentation"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%E2%80%933.15-blue.svg" alt="Python 3.10–3.15"></a>
<a href="https://github.com/Silas-Asamoah/stormlog/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

---

## About

Stormlog measures what a model does to the hardware it runs on, and keeps
the evidence. It works the same way for a training job's memory and for an
inference server's latency: capture once, save an artifact, then analyze,
compare and diagnose it from the CLI, Python or a terminal UI.

**Training and memory**

- Track allocated, reserved and device memory over time for PyTorch (CUDA,
  ROCm, Apple MPS and CPU), TensorFlow, JAX, and MLX on Apple Silicon.
- Catch growth before it becomes an OOM: alert thresholds, growth and gap
  analysis, and an OOM flight recorder that keeps the moments before a crash.
- Reconstruct long captures by session, and compare ranks of a distributed
  run to find the one that diverged.
- Write diagnose bundles with a fixed exit-code and report contract for CI and
  release gates, and log to W&B or MLflow.

**Inference serving**

- Profile any OpenAI-compatible endpoint: time to first token, time per
  output token, end-to-end latency, throughput, failures and tokens, under
  closed-loop, fixed-rate, Poisson, burst or replayed traffic.
- Judge each run against an SLO with goodput, and compare a baseline with a
  candidate over repeated runs (`stormlog infer compare`), so a change is
  shown to help instead of assumed to.
- Explain slow windows (`stormlog infer diagnose`): queue saturation, KV
  preemption, prefix-cache loss, client admission and workload changes, each
  finding citing the records behind it.
- Go deeper on vLLM: its Prometheus metrics, an execution hook that records
  scheduler steps, and GPU profiler traces joined to the requests they served.
- Watch a live server and capture its incidents (`stormlog infer watch`), and
  export Prometheus metrics and OTLP spans.

## Supported runtimes

| Area | Supported |
| --- | --- |
| Training memory | PyTorch (CUDA, ROCm, MPS, CPU), TensorFlow, JAX, MLX on Apple Silicon |
| Inference serving | vLLM (metrics, execution hook, profiler traces), any OpenAI-compatible server, MLX-LM, oMLX, MLX-Audio |
| Interfaces | Python API, `gpumemprof` / `tfmemprof` / `jaxmemprof` / `mlxmemprof`, `stormlog infer`, `stormlog query`, terminal UI |
| Python | 3.10 to 3.15 |

## Getting started

```bash
pip install stormlog                 # core, inference profiling and the CLIs
pip install "stormlog[tui,torch]"    # terminal UI with PyTorch
```

Other extras: `torch`, `tf`, `jax`, `mlx` (Apple Silicon), `viz`,
`infer-tokenizers`, `infer-otlp`, `wandb`, `mlflow`, and `all`. The
`examples/` package and the scenario scripts are source-checkout only: clone
the repository and `pip install -e .` to run them.

### Training memory

```bash
gpumemprof info                      # tfmemprof info, jaxmemprof info, mlxmemprof info
gpumemprof track --duration 30 --interval 0.5 --output track.json --format json
gpumemprof analyze track.json --format txt --output analysis.txt
gpumemprof diagnose --duration 0 --output diag/
stormlog                             # the terminal UI
```

```python
import torch
from stormlog import GPUMemoryProfiler  # CUDA or ROCm; MemoryTracker also covers MPS

profiler = GPUMemoryProfiler()
model = torch.nn.Linear(1024, 128).to(profiler.device)


def train_step() -> torch.Tensor:
    return model(torch.randn(64, 1024, device=profiler.device)).sum()


profiler.profile_function(train_step)
print(f"Peak: {profiler.get_summary()['peak_memory_usage'] / 1024**2:.1f} MB")
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Silas-Asamoah/stormlog/main/docs/tui-visualizations-current.png" alt="Stormlog's terminal UI showing the memory timeline of a PyTorch training run" width="900">
  <br/>
  <em>The terminal UI during a PyTorch training run on an NVIDIA L4: allocated memory per sample, and the caching allocator's reserved pool.</em>
</p>

### Inference

```bash
stormlog infer profile \
  --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 1,8,32 --duration 60 --slo ttft:500 --vllm-metrics \
  --output run.jsonl
stormlog infer analyze run.jsonl     # latency, throughput, goodput per case
stormlog infer diagnose run.jsonl    # why slow windows were slow
stormlog infer compare --baseline base-*.jsonl --candidate cand-*.jsonl \
  --gate 'client.e2e.p95=non-inferiority:0.05'
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Silas-Asamoah/stormlog/main/docs/infer-diagnose-current.png" alt="stormlog infer diagnose explaining a queue saturation incident with its evidence" width="900">
  <br/>
  <em>An excerpt of <code>stormlog infer diagnose</code> on vLLM 0.30.0 serving Qwen3-0.6B on an NVIDIA L4, overloaded for 12 seconds.</em>
</p>

## Documentation

- [Installation](https://stormlog.readthedocs.io/en/latest/installation.html)
  and [usage guide](https://stormlog.readthedocs.io/en/latest/usage.html)
- [CLI reference](https://stormlog.readthedocs.io/en/latest/cli.html) and
  [Python API](https://stormlog.readthedocs.io/en/latest/api.html)
- [Inference profiling](https://stormlog.readthedocs.io/en/latest/inference.html),
  [SLOs and goodput](https://stormlog.readthedocs.io/en/latest/inference_slo.html),
  [comparing runs](https://stormlog.readthedocs.io/en/latest/inference_comparison.html)
  and [diagnosis](https://stormlog.readthedocs.io/en/latest/inference_diagnosis.html)
- [vLLM execution hook](https://stormlog.readthedocs.io/en/latest/vllm_execution.html),
  [incident capture](https://stormlog.readthedocs.io/en/latest/incident_capture.html)
  and [Prometheus and OTLP export](https://stormlog.readthedocs.io/en/latest/inference_export.html)
- [Terminal UI](https://stormlog.readthedocs.io/en/latest/tui.html),
  [MLX on Apple Silicon](https://stormlog.readthedocs.io/en/latest/mlx_testing_guide.html)
  and [CPU-only machines](https://stormlog.readthedocs.io/en/latest/cpu_compatibility.html)
- [Production cookbook](https://stormlog.readthedocs.io/en/latest/cookbook/index.html):
  always-on tracking, incident playbooks, and CI and release qualification
- [Report and exit-code contract](https://stormlog.readthedocs.io/en/latest/report_contract.html)

## Contributing

Contributions are welcome. See the
[contributing guide](https://github.com/Silas-Asamoah/stormlog/blob/main/CONTRIBUTING.md)
and the
[code of conduct](https://github.com/Silas-Asamoah/stormlog/blob/main/CODE_OF_CONDUCT.md).

## License

[MIT](https://github.com/Silas-Asamoah/stormlog/blob/main/LICENSE)
