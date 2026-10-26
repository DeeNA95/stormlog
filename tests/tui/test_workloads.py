from types import SimpleNamespace
from typing import Any

import pytest

from stormlog.tui.workloads import (
    format_cpu_summary,
    format_pytorch_summary,
    format_tensorflow_results,
)


def test_format_pytorch_summary_formats_negative_delta_in_gb() -> None:
    summary = {
        "total_functions_profiled": 1,
        "total_function_calls": 1,
        "peak_memory_usage": 0,
        "memory_change_from_baseline": -(1024**3),
    }

    formatted = format_pytorch_summary(summary)

    assert "from baseline: -1.00 GB" in formatted
    assert "-1073741824" not in formatted


def test_format_cpu_summary_formats_negative_delta_in_gb() -> None:
    summary = {
        "snapshots_collected": 1,
        "peak_memory_usage": 0,
        "memory_change_from_baseline": -(1024**3),
    }

    formatted = format_cpu_summary(summary)

    assert "from baseline: -1.00 GB" in formatted
    assert "-1073741824" not in formatted


def test_format_tensorflow_results_handles_missing_attributes() -> None:
    formatted = format_tensorflow_results(object())

    assert formatted == (
        "Duration: 0.00s\n"
        "Peak memory: 0.00 MB\n"
        "Average memory: 0.00 MB\n"
        "Snapshots: 0"
    )


def test_format_tensorflow_results_handles_none_values() -> None:
    results = SimpleNamespace(
        duration=None,
        peak_memory_mb=None,
        average_memory_mb=None,
        snapshots=None,
    )

    formatted = format_tensorflow_results(results)

    assert formatted == (
        "Duration: 0.00s\n"
        "Peak memory: 0.00 MB\n"
        "Average memory: 0.00 MB\n"
        "Snapshots: 0"
    )


def test_the_pytorch_sample_records_into_the_profiler_the_tab_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stormlog.context_profiler as context_profiler
    import stormlog.tui.app as appmod

    class FakeProfiler:
        def __init__(self) -> None:
            self.profiled: list[Any] = []
            self.results: list[Any] = []

        def profile_function(self, workload: Any) -> None:
            self.profiled.append(workload)
            size = 100 * len(self.profiled)
            self.results.append(
                SimpleNamespace(
                    peak_memory_usage=lambda: size, memory_diff=lambda: size // 10
                )
            )

    fake = FakeProfiler()
    fake_torch = SimpleNamespace(
        randn=lambda *args, **kwargs: None, matmul=lambda *args: None
    )
    monkeypatch.setattr(context_profiler, "_global_profiler", fake)
    monkeypatch.setattr(appmod, "GPUMemoryProfiler", object())
    monkeypatch.setattr(appmod, "torch", fake_torch)

    first = appmod.GPUMemoryProfilerTUI._pytorch_sample_workload()
    second = appmod.GPUMemoryProfilerTUI._pytorch_sample_workload()

    # Each summary is its own run, though the profiler is shared.
    assert len(fake.profiled) == 2
    assert first["peak_memory_usage"] == 100 and second["peak_memory_usage"] == 200
    assert second["total_function_calls"] == 1
    assert second["memory_change_from_baseline"] == 20


def test_the_tensorflow_sample_shows_in_the_tensorflow_tab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tensorflow")
    import stormlog.tensorflow.context_profiler as tf_context
    import stormlog.tui.app as appmod
    from stormlog.tui.profiles import fetch_tensorflow_profiles

    monkeypatch.setattr(tf_context, "_global_profiler", None)
    appmod._ensure_tensorflow_runtime()

    appmod.GPUMemoryProfilerTUI._tensorflow_sample_workload()
    second = appmod.GPUMemoryProfilerTUI._tensorflow_sample_workload()

    assert [row.name for row in fetch_tensorflow_profiles()] == ["tf_sample"]
    # The second run reports its own two snapshots, not both runs'.
    assert len(second.snapshots) == 2
    assert second.duration < 5
