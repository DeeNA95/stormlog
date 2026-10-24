"""Measure compact artifact savings on the #216 audit workload, on Modal.

Captures a Kineto trace of Qwen2.5-7B-Instruct on vLLM 0.30.0 with 64 requests
in flight on one L4, then imports it at launch and kernel detail twice: with
the revision under test, through the paired comparison in
``context_compaction_scenario``, and with the baseline revision's own import.
The trace stays in a Modal volume; only JSON comes back.

Run with the separately installed Modal CLI (not a Stormlog dependency).
Image builder 2025.06 gives a bookworm base whose compiler defaults to C++17,
which the CUDA check below needs::

    MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run \\
        examples/scenarios/modal_audit_compaction.py
    # compare a saved capture again, without the GPU:
    MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run \\
        examples/scenarios/modal_audit_compaction.py --run-id <run id>

The model is downloaded on a CPU container first; one capture then used about
eight minutes of L4 time.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
VLLM_VERSION = "0.30.0"
REPOSITORY = "https://github.com/Silas-Asamoah/stormlog.git"
# The merge of PR #296 (#259), and release/dev just before it.
REVISION = "078458cdbf9ca28984b88f8d1b5a361836fbea13"
BASELINE = "330f7feab1a55675b970af88129949ea71657e9f"
IN_FLIGHT = 64
DATA = Path("/data")
CUDA_HOME = "/usr/local/lib/python3.12/site-packages/nvidia/cu13"

app = modal.App("stormlog-issue-259-audit")
volume = modal.Volume.from_name("stormlog-259-validation", create_if_missing=True)

# The image of modal_context_trace.py, with the model cache on the volume.
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}")
    .pip_install(
        "nvidia-cuda-nvcc==13.0.88",
        "nvidia-cuda-crt==13.0.88",
        "nvidia-nvvm==13.0.88",
        "nvidia-cuda-runtime==13.0.96",
    )
    .run_commands(
        f"test -x {CUDA_HOME}/bin/nvcc",
        f"ln -s lib {CUDA_HOME}/lib64",
        f"ln -s libcudart.so.13 {CUDA_HOME}/lib/libcudart.so",
        f"{CUDA_HOME}/bin/nvcc --version",
        "printf '#include <cuda_runtime.h>\\n"
        "#include <cuda/std/type_traits>\\n"
        "__global__ void check() {}\\n"
        "int main() { return 0; }\\n' > /tmp/stormlog-cuda-check.cu",
        f"{CUDA_HOME}/bin/nvcc --cudart shared /tmp/stormlog-cuda-check.cu "
        "-I/usr/local/lib/python3.12/site-packages/flashinfer/data/cccl/"
        "libcudacxx/include -o /tmp/stormlog-cuda-check",
    )
    .env({"CUDA_HOME": CUDA_HOME, "HF_HOME": str(DATA / "hf")})
)

# The comparison scenario reads its revision with git, so check both out.
compare_image = (
    modal.Image.from_registry("python:3.12-slim-bookworm")
    .run_commands(
        "apt-get update -q && apt-get install -y -q --no-install-recommends git"
    )
    .pip_install("numpy", "pandas", "psutil", "scipy")
    .run_commands(
        f"git clone --quiet {REPOSITORY} /src/repo",
        f"git -C /src/repo worktree add --detach /src/revision {REVISION}",
        f"git -C /src/repo worktree add --detach /src/baseline {BASELINE}",
    )
)


@app.function(  # type: ignore[misc]
    image=gpu_image, cpu=2, memory=4096, timeout=1800, volumes={str(DATA): volume}
)
def download() -> str:
    from huggingface_hub import snapshot_download

    path = str(snapshot_download(MODEL, revision=MODEL_REVISION))
    volume.commit()
    return path


def _prompts(count: int) -> list[str]:
    body = "Development, king safety, pawn structure and active pieces matter. "
    return [
        f"Request {i}. Explain these chess considerations. "
        + body * (4 + i % 4 * 4)
        + "Give a detailed continuation:"
        for i in range(count)
    ]


def _wait_for_trace(trace_dir: Path, timeout_s: float = 600) -> list[Path]:
    """The profiler writes asynchronously: wait until the gzip stops growing."""
    deadline, last = time.monotonic() + timeout_s, None
    while time.monotonic() < deadline:
        traces = sorted(trace_dir.rglob("*.pt.trace.json.gz"))
        sizes = tuple(path.stat().st_size for path in traces)
        if traces and sizes == last:
            return traces
        last = sizes
        time.sleep(10)
    raise TimeoutError(f"no complete trace in {trace_dir}")


def _engine_args(trace_dir: Path) -> dict[str, Any]:
    return {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "max_model_len": 1024,
        "max_num_seqs": IN_FLIGHT,
        "gpu_memory_utilization": 0.90,
        "enable_prefix_caching": False,
        "seed": 259,
        "profiler_config": {
            "profiler": "torch",
            "torch_profiler_dir": str(trace_dir),
            "torch_profiler_use_gzip": True,
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_memory": False,
            "torch_profiler_with_flops": False,
        },
    }


@app.function(  # type: ignore[misc]
    image=gpu_image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=2400,
    retries=0,
    max_containers=1,
    volumes={str(DATA): volume},
)
def capture(run_id: str, requests: int, max_tokens: int) -> dict[str, Any]:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    root = DATA / "runs" / run_id
    trace_dir = root / "traces"
    trace_dir.mkdir(parents=True)
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    engine_args = _engine_args(trace_dir)
    started = time.monotonic()
    llm = LLM(**engine_args)
    load_s = time.monotonic() - started
    # Staggered lengths keep the engine full: finished requests are replaced
    # one at a time instead of the whole batch draining at once.
    params = [
        SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=max_tokens - 64 + (i * 37) % 128,
        )
        for i in range(requests)
    ]
    warm = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=32)
    llm.generate(_prompts(IN_FLIGHT), warm, use_tqdm=False)
    llm.start_profile()
    t0 = time.monotonic()
    try:
        outputs = llm.generate(_prompts(requests), params, use_tqdm=False)
    finally:
        profiled_s = time.monotonic() - t0
        llm.stop_profile()
    traces = _wait_for_trace(trace_dir)
    meta = {
        "run_id": run_id,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "vllm": str(vllm.__version__),
        "torch": str(torch.__version__),
        "gpu_csv": gpu,
        "device_uuid": gpu.split(",")[1].strip(),
        "engine_args": {
            key: value for key, value in engine_args.items() if key != "profiler_config"
        },
        "requests": requests,
        "max_num_seqs": IN_FLIGHT,
        "max_tokens_range": [max_tokens - 64, max_tokens + 63],
        "generated_tokens": sum(len(o.outputs[0].token_ids) for o in outputs),
        "prompt_tokens": sum(len(o.prompt_token_ids) for o in outputs),
        "engine_load_s": round(load_s, 1),
        "profiled_wall_s": round(profiled_s, 2),
        "traces": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in traces
        ],
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (root / "capture.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    volume.commit()
    return meta


SEED = """
from stormlog.infer.correlation_events import (
    ArtifactIdentityEvent,
    CorrelationContext,
)
from stormlog.infer.events import JsonlEventWriter

identity = ArtifactIdentityEvent(
    context=CorrelationContext(
        run_id="context-compaction",
        session_id="context-compaction",
        producer_id="scenario",
        source="context_compaction_scenario",
        clock_domain="scenario/wall",
        clock_kind="wall",
        collection_mode="imported",
        provenance="observed",
    ),
    event_id="artifact",
    artifact_kind="inference_jsonl",
    created_at_ns=0,
)
"""

# The baseline's own import, seeded exactly as the scenario seeds its file.
BASELINE_CHECK = (
    SEED
    + """
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from stormlog.infer.correlation_accounting import (
    account_gpu_time,
    resolve_inference_events,
)
from stormlog.infer.correlation_events import load_inference_artifact
from stormlog.infer.trace_import import (
    import_traces_into_artifact,
    parse_device_uuids,
)

trace, detail, device, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
out.mkdir(parents=True)
artifact = out / "baseline.jsonl"
with JsonlEventWriter(artifact) as writer:
    writer.append(identity.to_record())
t0 = time.perf_counter()
import_traces_into_artifact(
    artifact, [trace], detail=detail, device_uuids=parse_device_uuids([device])
)
import_s = time.perf_counter() - t0
t0 = time.perf_counter()
records = load_inference_artifact(artifact)
load_s = time.perf_counter() - t0
accounting = account_gpu_time(resolve_inference_events(records))
types = Counter(record.to_record()["event_type"] for record in records)
print(json.dumps({
    "bytes": artifact.stat().st_size,
    "physical_lines": sum(1 for line in artifact.open() if line.strip()),
    "semantic_events": len(records),
    "event_types": dict(types),
    "device_clock_totals": [
        {**asdict(dimension), **asdict(timing)}
        for dimension, timing in accounting.device_totals.items()
    ],
    "import_s": round(import_s, 2),
    "load_s": round(load_s, 2),
}))
"""
)

# Load times with the revision under test, on the scenario's paired files.
LOAD_TIMING = """
import json
import sys
import time

from stormlog.infer.correlation_events import load_inference_artifact

timings = {}
for name in ("embedded", "compact"):
    t0 = time.perf_counter()
    load_inference_artifact(f"{sys.argv[1]}/{name}.jsonl")
    timings[f"{name}_load_s"] = round(time.perf_counter() - t0, 2)
print(json.dumps(timings))
"""


def _python(checkout: str, *args: str) -> str:
    env = {**os.environ, "PYTHONPATH": checkout}
    result = subprocess.run(
        [sys.executable, *args], cwd=checkout, env=env, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"{args[:3]} failed:\n{result.stderr[-4000:]}")
    return result.stdout


@app.function(  # type: ignore[misc]
    image=compare_image,
    cpu=4,
    memory=24576,
    timeout=2400,
    volumes={str(DATA): volume},
)
def compare(run_id: str, detail: str) -> dict[str, Any]:
    volume.reload()
    meta = json.loads((DATA / "runs" / run_id / "capture.json").read_text())
    trace = str(DATA / "runs" / run_id / meta["traces"][0]["path"])
    device = f"0={meta['device_uuid']}"
    out = Path("/tmp") / detail
    scenario = "examples.scenarios.context_compaction_scenario"
    t0 = time.perf_counter()
    _python(
        "/src/revision", "-m", scenario, "--trace", trace, "--detail", detail,
        "--device-uuid", device, "--output-dir", str(out / "revision"),
    )  # fmt: skip
    scenario_s = time.perf_counter() - t0
    revision = json.loads((out / "revision" / "results.json").read_text())
    revision["scenario_s"] = round(scenario_s, 1)
    revision.update(
        json.loads(_python("/src/revision", "-c", LOAD_TIMING, str(out / "revision")))
    )
    baseline = json.loads(
        _python(
            "/src/baseline", "-c", BASELINE_CHECK, trace, detail, device,
            str(out / "baseline"),
        )  # fmt: skip
    )
    subprocess.run(["rm", "-rf", str(out)], check=True)
    return {"capture": meta, "revision": revision, "baseline": baseline}


@app.local_entrypoint()  # type: ignore[misc]
def main(
    run_id: str = "",
    requests: int = 160,
    max_tokens: int = 256,
    output_dir: str = "artifacts/issue-259-audit",
) -> None:
    if not run_id:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
        print("model:", download.remote())
        print(json.dumps(capture.remote(run_id, requests, max_tokens), indent=2))
    calls = {detail: compare.spawn(run_id, detail) for detail in ("launch", "kernel")}
    comparisons = {detail: call.get() for detail, call in calls.items()}
    capture_meta = comparisons["launch"].pop("capture")
    comparisons["kernel"].pop("capture")
    results = {
        "workload": "the #216 audit workload: Qwen2.5-7B-Instruct, "
        f"{IN_FLIGHT} requests in flight",
        "repository": REPOSITORY,
        "revision": REVISION,
        "baseline": BASELINE,
        "harness": {
            "path": "examples/scenarios/modal_audit_compaction.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "capture": capture_meta,
        "comparisons": comparisons,
    }
    path = Path(output_dir).resolve() / f"results-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print("saved", path)
