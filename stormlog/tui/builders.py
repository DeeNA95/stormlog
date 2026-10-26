"""Markdown and table-data builders for the Textual TUI."""

from __future__ import annotations

from textwrap import dedent
from typing import Any


def build_welcome_info() -> str:
    """Build the overview's guide to the tabs and keys."""
    return dedent("""
        ### Tabs

        - **PyTorch**, **TensorFlow**: device memory now, and the profiles
          recorded in this process
        - **MLX**: the Apple Silicon allocator, samples and saved MLX artifacts
        - **Monitoring**: live tracking, alert thresholds, cleanup, CSV or
          JSON export
        - **Visualizations**: the tracked timeline, saved as PNG or HTML
        - **Diagnostics**: compare ranks from live tracking or saved artifacts
        - **CLI & Actions**: run `gpumemprof` and `tfmemprof` commands and
          sample workloads

        ### Keys

        `q` quit · `r` refresh this overview · `f` focus the command log ·
        `g` and `t` log `gpumemprof` and `tfmemprof` examples

        ### Inference

        Profiling an LLM server? `stormlog infer profile` measures it and
        `stormlog infer diagnose` explains its slow requests. Both run from
        the shell; see `stormlog infer --help`.
        """).strip()


def build_system_markdown(
    *,
    system_info: dict[str, Any],
    gpu_info: dict[str, Any],
    tf_system_info: dict[str, Any],
    tf_gpu_info: dict[str, Any],
) -> str:
    lines = [
        "### System",
        "",
        f"- **Platform**: {system_info.get('platform', 'Unknown')}",
        f"- **Python**: {str(system_info.get('python_version', 'Unknown')).split()[0]}",
        f"- **TensorFlow (Python)**: {tf_system_info.get('tensorflow_version', 'N/A')}",
        f"- **CUDA Available**: {system_info.get('cuda_available', False)}",
    ]

    if system_info.get("cuda_available"):
        lines.extend(
            [
                f"- **CUDA Version**: {system_info.get('cuda_version', 'Unknown')}",
                f"- **GPU Count**: {system_info.get('cuda_device_count', 0)}",
            ]
        )

    if gpu_info:
        lines.append("")
        lines.append("### GPU")
        lines.extend(
            [
                f"- **Device Name**: {gpu_info.get('device_name', 'Unknown')}",
                f"- **Total Memory**: {gpu_info.get('total_memory', 0) / (1024**3):.2f} GB",
                f"- **Allocated**: {gpu_info.get('allocated_memory', 0) / (1024**3):.2f} GB",
                f"- **Reserved**: {gpu_info.get('reserved_memory', 0) / (1024**3):.2f} GB",
            ]
        )
    else:
        lines.append("")
        lines.append(
            "> GPU metrics are unavailable on this system. You can still run the CLI "
            "and CPU guides."
        )

    lines.append("")
    if tf_gpu_info and tf_gpu_info.get("devices"):
        lines.append("")
        lines.append("### TensorFlow GPU")
        device = tf_gpu_info["devices"][0]
        lines.extend(
            [
                f"- **TF Device Name**: {device.get('name', 'Unknown')}",
                f"- **Current Memory**: {device.get('current_memory_mb', 0):.2f} MB",
                f"- **Peak Memory**: {device.get('peak_memory_mb', 0):.2f} MB",
            ]
        )

    lines.append("")
    lines.append("### First commands")
    lines.append("")
    lines.append("- `gpumemprof info`")
    lines.append("- `gpumemprof track --duration 60 --output tracking.json`")
    lines.append("- `gpumemprof diagnose --duration 0 --output artifacts/diag`")
    lines.append("")
    lines.append("Docs: https://stormlog.readthedocs.io")
    return "\n".join(lines)


def build_pytorch_stats_rows(info: dict[str, Any]) -> list[dict[str, Any]]:
    if not info:
        return []
    return [
        {
            "device": info.get("device_name", "gpu0"),
            "current": info.get("allocated_memory", 0) / (1024**2),
            "peak": info.get("max_memory_allocated", info.get("allocated_memory", 0))
            / (1024**2),
            "reserved": info.get("reserved_memory", 0) / (1024**2),
        }
    ]


def build_tensorflow_stats_rows(gpu_info: dict[str, Any]) -> list[dict[str, Any]]:
    devices = gpu_info.get("devices", []) if gpu_info else []
    rows = []
    for device in devices:
        rows.append(
            {
                "device": device.get("name", "tf-gpu"),
                "current": device.get("current_memory_mb", 0),
                "peak": device.get("peak_memory_mb", 0),
                "reserved": gpu_info.get("total_memory", 0),
            }
        )
    return rows


def build_framework_markdown(framework: str) -> str:
    if framework == "mlx":
        return dedent("""
            **MLX** on native Apple Silicon: install `stormlog[mlx,tui]` (MLX
            0.32.3 or newer). Active allocations, reusable cache, the native
            lifetime peak and process RSS have different scopes; device
            free/total and fragmentation are unavailable. Profiles evaluate
            explicit roots and report synchronized host time.

            For another process, capture it with
            `mlxmemprof run --output artifacts/mlx-run -- script.py`, then load
            the bundle below or run `mlxmemprof analyze artifacts/mlx-run`.
            """).strip()
    if framework == "pytorch":
        return dedent("""
            **PyTorch** device memory as PyTorch reports it, and the profiles
            recorded in this process with `profile_function` or
            `profile_context`. **PyTorch Sample** in CLI & Actions records one.

            From the shell: `gpumemprof track --duration 60 --output
            tracking.json` and `gpumemprof diagnose --duration 0 --output
            artifacts/diag`. Guide:
            https://stormlog.readthedocs.io/en/latest/pytorch_testing_guide.html
            """).strip()

    return dedent("""
        **TensorFlow** device memory as TensorFlow reports it, and the
        profiles recorded in this process. **TensorFlow Sample** in CLI &
        Actions records one.

        From the shell: `tfmemprof track --output tf_results.json` and
        `tfmemprof diagnose --duration 0 --output artifacts/tf-diag`. Guide:
        https://stormlog.readthedocs.io/en/latest/tensorflow_testing_guide.html
        """).strip()


def build_cli_markdown() -> str:
    return dedent("""
        Run a command and read its output below: `gpumemprof` for PyTorch,
        `tfmemprof` for TensorFlow, `mlxmemprof` for MLX, and
        `stormlog infer` for inference servers. **OOM Scenario** and
        **Capability Matrix** run examples from a source checkout.
        """).strip()


def build_visual_markdown() -> str:
    return dedent("""
        The live tracking session's timeline. **Refresh Timeline** redraws
        it; **Generate PNG Plot** (Matplotlib) and **Generate HTML Plot**
        (interactive Plotly) save it under `./visualizations` and need
        `stormlog[viz]`.
        """).strip()


def build_diagnostics_markdown() -> str:
    return dedent("""
        Compare the ranks of one run. **Load Live** reads the tracking
        session; **Load Artifacts** merges JSON or CSV exports and diagnose
        directories, given as comma-separated paths. Filter ranks with `all`
        or `0,2,4-7`, select a rank to focus its timeline, and read the
        earliest and most severe first-cause indicators below.
        """).strip()


def build_mlx_runtime_markdown(info: dict[str, Any]) -> str:
    """One line on the MLX runtime and its device, for the MLX tab."""
    from stormlog.formatting import format_bytes

    device = info.get("raw_device_info") or {}
    parts = []
    name = device.get("device_name")
    if name:
        architecture = device.get("architecture")
        parts.append(f"**{name}**" + (f" ({architecture})" if architecture else ""))
    for key, label in (
        ("memory_size", "unified memory"),
        ("max_recommended_working_set_size", "recommended working set"),
    ):
        value = device.get(key)
        if isinstance(value, (int, float)) and value > 0:
            parts.append(f"{format_bytes(int(value))} {label}")
    error = info.get("device_info_error")
    if error:
        parts.append(f"device details unavailable: {error}")
    head = (
        f"MLX {info.get('mlx_version', 'N/A')} on Metal, observing the allocator "
        "of this TUI process."
    )
    return head + (" " + ", ".join(parts) + "." if parts else "")


def build_mlx_stats_rows(stats: dict[str, Any]) -> list[tuple[str, str]]:
    from stormlog.formatting import format_bytes

    rows = [
        ("Scope", "MLX allocator in this TUI process"),
        ("MLX version", str(stats.get("mlx_version", "N/A"))),
        ("Collector health", str(stats.get("collector_health_status", "N/A"))),
    ]
    for label, key in (
        ("Active allocator", "current_memory_bytes"),
        ("Reusable cache", "cache_bytes"),
        ("Process RSS (overlaps unified pages)", "process_rss_bytes"),
        ("Sampled peak", "max_memory_bytes"),
        ("Native lifetime peak", "runtime_lifetime_peak_bytes"),
    ):
        value = stats.get(key)
        rows.append((label, "N/A" if value is None else format_bytes(value)))
    for key in ("total_samples", "retained_samples", "dropped_samples", "total_alerts"):
        rows.append((key.replace("_", " ").title(), str(stats.get(key, 0))))
    return rows


def build_mlx_timeline_rows(timeline: dict[str, Any]) -> list[tuple[str, str]]:
    from stormlog.formatting import format_bytes

    rows = [
        ("Scope", "Saved/live MLX process allocator"),
        ("Samples", str(len(timeline.get("timestamps", [])))),
    ]
    for label, key in (
        ("Active allocator", "allocated"),
        ("Reusable cache", "cache"),
        ("Process RSS", "rss"),
    ):
        values = timeline.get(key, [])
        known = [v for v in values if v is not None]
        latest = values[-1] if values else None
        rows.append(
            (label + " latest", "N/A" if latest is None else format_bytes(latest))
        )
        rows.append(
            (label + " sampled max", format_bytes(max(known)) if known else "N/A")
        )
    return rows


def build_legacy_timeline_rows(timeline: dict[str, Any]) -> list[tuple[str, str]]:
    """PyTorch/TensorFlow timeline stats in the table's established order."""
    timestamps = timeline.get("timestamps") or []
    allocated = timeline.get("allocated") or []
    reserved = timeline.get("reserved") or []
    if not allocated or not timestamps:
        return [
            ("Samples", "0"),
            ("Duration (s)", "-"),
            ("Allocated Max", "-"),
            ("Reserved Max", "-"),
            ("Allocated Latest", "-"),
            ("Reserved Latest", "-"),
        ]
    duration = max(0.0, timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    allocated_max, allocated_latest = _legacy_max_and_latest(allocated, "-")
    # A timeline without a reserved series has always shown 0 B.
    reserved_max, reserved_latest = _legacy_max_and_latest(reserved, "0 B")
    return [
        ("Samples", str(len(allocated))),
        ("Duration (s)", f"{duration:.1f}"),
        ("Allocated Max", allocated_max),
        ("Reserved Max", reserved_max),
        ("Allocated Latest", allocated_latest),
        ("Reserved Latest", reserved_latest),
    ]


def _legacy_max_and_latest(values: list[Any], empty: str) -> tuple[str, str]:
    from stormlog.formatting import format_bytes

    if not values:
        return empty, empty
    known = [v for v in values if v is not None]
    latest = values[-1]
    return (
        format_bytes(int(max(known))) if known else "-",
        format_bytes(int(latest)) if latest is not None else "-",
    )
