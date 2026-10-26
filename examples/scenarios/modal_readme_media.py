"""Capture the README's screenshots on an NVIDIA L4, on Modal.

Two parts, each in its own L4 container:

- ``tui``: the TUI tabs, driven by ``tui_screenshots`` while a PyTorch
  training loop runs on the GPU in the same process.
- ``inference``: vLLM 0.30.0 serving Qwen3-0.6B with Stormlog's execution
  hook. A replayed arrival trace runs calm, then overloads the server, then
  calms again; ``stormlog infer analyze`` and ``stormlog infer diagnose`` read
  the artifact, and their output is rendered as terminal SVGs.

The SVGs land in ``docs/`` (``--png`` also renders PNGs with a local Chrome),
with the run's provenance in ``docs/benchmarks/readme-media.json``. Run with
the separately installed Modal CLI (not a Stormlog dependency)::

    MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run \\
        examples/scenarios/modal_readme_media.py --part all --png

The source checkout is copied into the image, so the screenshots show the
working tree. One run of both parts used about ten minutes of L4 time.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

# In the container this file is /root/modal_readme_media.py and the checkout
# is the copy at /src.
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/src")
MODEL = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
VLLM_VERSION = "0.30.0"
CUDA_HOME = "/usr/local/lib/python3.12/site-packages/nvidia/cu13"
HOOK_DIR = "/tmp/stormlog-vllm"
PORT = 8000

# Calm at 3 requests/s for 60 s, 25/s for 12 s, then calm for 40 s: the calm
# stretch gives the diagnosis its reference (at least 114 requests).
TRACE = ((0.0, 60.0, 3.0), (60.0, 72.0, 25.0), (72.0, 112.0, 3.0))

app = modal.App("stormlog-readme-media")

# The image of modal_context_trace.py, plus this checkout and the TUI extra.
image = (
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
    )
    .pip_install("rich")
    .env({"CUDA_HOME": CUDA_HOME, "SETUPTOOLS_SCM_PRETEND_VERSION": "0.5.0.dev0"})
    .add_local_dir(
        str(REPO),
        "/src",
        copy=True,
        ignore=[
            ".git",
            "**/__pycache__",
            "**/*.egg-info",
            "artifacts",
            "docs",
            "build",
            "dist",
            ".venv",
        ],
    )
    # Stormlog's own dependencies; vLLM already pins numpy and torch.
    .run_commands(
        "pip install -e '/src[tui]'",
        "python -c 'import stormlog.infer.cli, stormlog.tui.app'",
    )
)


def _gpu() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=900)  # type: ignore[misc]
def tui(track_seconds: float) -> dict[str, Any]:
    sys.path.insert(0, "/src")
    from examples.scenarios.tui_screenshots import capture

    out = Path("/tmp/shots")
    paths = capture(out, track_seconds)
    return {
        "gpu": _gpu(),
        "svgs": {path.name: path.read_text() for path in paths},
        "readme_checks": _readme_checks(),
    }


# The README's training commands and Python example, run as written.
README_SNIPPET = """
import torch
from stormlog import GPUMemoryProfiler

profiler = GPUMemoryProfiler()
model = torch.nn.Linear(1024, 128).to(profiler.device)


def train_step() -> torch.Tensor:
    return model(torch.randn(64, 1024, device=profiler.device)).sum()


profiler.profile_function(train_step)
print(f"Peak: {profiler.get_summary()['peak_memory_usage'] / 1024**2:.1f} MB")
"""

README_COMMANDS = (
    ["gpumemprof", "info"],
    [
        "gpumemprof",
        "track",
        "--duration",
        "5",
        "--interval",
        "0.5",
        "--output",
        "track.json",
        "--format",
        "json",
    ],
    [
        "gpumemprof",
        "analyze",
        "track.json",
        "--format",
        "txt",
        "--output",
        "analysis.txt",
    ],
    ["gpumemprof", "diagnose", "--duration", "0", "--output", "diag/"],
    ["python", "-c", README_SNIPPET],
)


def _readme_checks() -> list[dict[str, Any]]:
    work = Path("/tmp/readme")
    work.mkdir(exist_ok=True)
    results = []
    for argv in README_COMMANDS:
        done = subprocess.run(
            argv, cwd=work, capture_output=True, text=True, timeout=300, check=False
        )
        results.append(
            {
                "argv": argv[:2] if argv[0] == "python" else argv,
                "exit": done.returncode,
                "stdout_tail": done.stdout[-600:],
                "stderr_tail": done.stderr[-600:],
            }
        )
    return results


def _write_trace(path: Path) -> int:
    lines = []
    for start, end, rate in TRACE:
        offset = start
        while offset < end:
            lines.append(json.dumps({"offset_ms": round(offset * 1000.0, 3)}))
            offset += 1.0 / rate
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


def _wait_ready(server: subprocess.Popen[bytes], timeout_s: float = 600) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited with {server.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError("vLLM did not become healthy")


def _run(argv: list[str], env: dict[str, str]) -> dict[str, Any]:
    started = time.monotonic()
    done = subprocess.run(
        argv, env=env, capture_output=True, text=True, cwd="/tmp/run", check=False
    )
    return {
        "argv": argv,
        "exit": done.returncode,
        "seconds": round(time.monotonic() - started, 1),
        "stdout": done.stdout,
        "stderr": done.stderr[-4000:],
    }


@app.function(image=image, gpu="L4", cpu=8, memory=32768, timeout=1500)  # type: ignore[misc]
def inference() -> dict[str, Any]:
    import vllm

    run = Path("/tmp/run")
    run.mkdir()
    requests = _write_trace(run / "arrivals.jsonl")
    env = dict(os.environ, COLUMNS="110", PYTHONUNBUFFERED="1")
    server_env = dict(env, STORMLOG_VLLM_HOOK_DIR=HOOK_DIR, VLLM_SERVER_DEV_MODE="1")
    serve = [
        "vllm",
        "serve",
        MODEL,
        "--revision",
        MODEL_REVISION,
        "--port",
        str(PORT),
        "--max-num-seqs",
        "16",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.6",
        "--seed",
        "0",
    ]
    log = (run / "vllm.log").open("wb")
    server = subprocess.Popen(serve, env=server_env, stdout=log, stderr=log)
    steps: list[dict[str, Any]] = []
    try:
        _wait_ready(server)
        profile = [
            "stormlog",
            "infer",
            "profile",
            "--base-url",
            f"http://127.0.0.1:{PORT}/v1",
            "--model",
            MODEL,
            "--engine",
            "vllm",
            "--arrival",
            "replay",
            "--arrival-trace",
            str(run / "arrivals.jsonl"),
            "--input-tokens",
            "512",
            "--output-tokens",
            "128",
            "--warmup-requests",
            "16",
            "--tokenizer",
            "none",
            "--vllm-metrics",
            "--vllm-metrics-interval",
            "1",
            "--vllm-execution-dir",
            HOOK_DIR,
            "--slo",
            "ttft:1000",
            "--slo",
            "tpot:60",
            "--output",
            str(run / "infer.jsonl"),
        ]
        steps.append(_run(profile, env))
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
    # Relative, so the diagnosis cites records as infer.jsonl:<line>.
    steps.append(_run(["stormlog", "infer", "analyze", "infer.jsonl"], env))
    steps.append(_run(["stormlog", "infer", "diagnose", "infer.jsonl"], env))
    return {
        "gpu": _gpu(),
        "vllm": vllm.__version__,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "serve": serve,
        "trace": [list(segment) for segment in TRACE],
        "trace_requests": requests,
        "steps": steps,
        "observers": _observers(run / "infer.jsonl"),
    }


def _observers(artifact: Path) -> Any:
    """The observer states the run recorded, for a reader checking its health."""
    done = subprocess.run(
        ["stormlog", "infer", "analyze", artifact.name, "--format", "json"],
        cwd=artifact.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        report = json.loads(done.stdout)
    except json.JSONDecodeError:
        return {"error": done.stderr[-2000:]}
    payload = report.get("payload", report)
    return payload.get("observers")


def _diagnosis_excerpt(text: str) -> str:
    """The verdict, the first finding whole, and the others' headlines.

    The coverage table and the other findings' detail are left out; the
    full text is kept in the provenance record.
    """
    lines = text.rstrip("\n").splitlines()
    kept: list[str] = []
    finding = 0
    skipping_coverage = False
    for line in lines:
        if line.startswith("Coverage:"):
            skipping_coverage = True
            continue
        if skipping_coverage:
            if line.strip():
                continue
            skipping_coverage = False
        if re.match(r"\[\d+\] ", line):
            finding += 1
            if finding > 1:
                kept.append(line)
                continue
        elif finding > 1:
            continue
        kept.append(line)
    return "\n".join(kept)


def _terminal_svg(command: str, output: str, title: str, width: int = 120) -> str:
    """Render output as a terminal window: wrapped lines keep their indent,
    blank runs collapse to one line, and finding labels are coloured."""
    import textwrap

    from rich.console import Console
    from rich.text import Text

    console = Console(
        record=True,
        width=width,
        file=io.StringIO(),
        force_terminal=True,
        color_system="truecolor",
    )
    console.print(Text("$ " + command, style="bold"))
    blank = False
    for line in output.rstrip("\n").splitlines():
        if not line.strip():
            if not blank:
                console.print()
            blank = True
            continue
        blank = False
        indent = line[: len(line) - len(line.lstrip())]
        wrapped = textwrap.fill(
            line, width=width, subsequent_indent=indent + "  ", break_on_hyphens=False
        )
        text = Text(wrapped)
        text.highlight_regex(r"\bWARNING\b", "bold #e5a50a")
        text.highlight_regex(r"\bINFO\b", "bold #3584e4")
        text.highlight_regex(r"^(Diagnosis|Outcome):", "bold")
        console.print(text, soft_wrap=True)
    return str(console.export_svg(title=title))


def _chrome() -> str | None:
    mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if Path(mac).exists():
        return mac
    for name in ("google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _png(svg: Path) -> Path | None:
    """Render an SVG at 2x with headless Chrome, which loads its web font."""
    chrome = _chrome()
    if chrome is None:
        return None
    box = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', svg.read_text()[:4000])
    if box is None:
        return None
    width, height = (int(float(value) + 0.999) for value in box.groups())
    png = svg.with_suffix(".png")
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={width},{height}",
            "--force-device-scale-factor=2",
            f"--screenshot={png}",
            svg.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    _shrink(png)
    return png


def _shrink(png: Path) -> None:
    """A 256-colour palette: a terminal has few colours, and the repository
    refuses files over 500 KB. Skipped without Pillow."""
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(png) as image:
        palette = image.convert("RGB").quantize(
            colors=256,
            method=Image.Quantize.FASTOCTREE,
            dither=Image.Dither.NONE,
        )
    palette.save(png, optimize=True)


@app.local_entrypoint()  # type: ignore[misc]
def main(
    part: str = "all",
    output: str = "docs",
    png: bool = False,
    track_seconds: float = 40.0,
) -> None:
    docs = Path(output)
    (docs / "benchmarks").mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    record_path = docs / "benchmarks" / "readme-media.json"
    record: dict[str, Any] = (
        json.loads(record_path.read_text()) if record_path.exists() else {}
    )
    record["captured_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if part in ("all", "tui"):
        shots = tui.remote(track_seconds)
        for name, text in shots["svgs"].items():
            path = docs / name
            path.write_text(text)
            written.append(path)
        record["tui"] = {
            "gpu": shots["gpu"],
            "files": sorted(shots["svgs"]),
            "readme_checks": [
                {"argv": check["argv"], "exit": check["exit"]}
                for check in shots["readme_checks"]
            ],
        }
        for check in shots["readme_checks"]:
            print(" ".join(check["argv"]), "->", check["exit"])
            if check["exit"]:
                print(check["stdout_tail"], check["stderr_tail"])
    if part in ("all", "inference"):
        result = inference.remote()
        texts = {" ".join(step["argv"][:3]): step["stdout"] for step in result["steps"]}
        diagnosis = texts["stormlog infer diagnose"]
        command = "stormlog infer diagnose infer.jsonl"
        path = docs / "infer-diagnose-current.svg"
        path.write_text(_terminal_svg(command, _diagnosis_excerpt(diagnosis), command))
        written.append(path)
        record["inference"] = {
            key: result[key]
            for key in (
                "gpu",
                "vllm",
                "model",
                "model_revision",
                "serve",
                "trace",
                "trace_requests",
                "observers",
            )
        } | {
            "steps": [
                {k: step[k] for k in ("argv", "exit", "seconds")}
                for step in result["steps"]
            ],
            "outputs": texts,
        }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    for path in written:
        rendered = _png(path) if png else None
        print(path, rendered or "")
