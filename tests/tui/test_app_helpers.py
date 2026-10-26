from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("textual")

from textual.widgets import Button

from stormlog.tui import app as appmod


def test_build_system_markdown_uses_system_info_fallback_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _raise_system_info() -> dict[str, object]:
        raise RuntimeError("system info unavailable")

    def _fake_build_system_markdown(
        *,
        system_info: dict[str, object],
        gpu_info: dict[str, object],
        tf_system_info: dict[str, object],
        tf_gpu_info: dict[str, object],
    ) -> str:
        captured["system_info"] = system_info
        captured["gpu_info"] = gpu_info
        captured["tf_system_info"] = tf_system_info
        captured["tf_gpu_info"] = tf_gpu_info
        return "ok"

    monkeypatch.setattr(appmod, "get_system_info", _raise_system_info)
    monkeypatch.setattr(appmod, "_safe_get_gpu_info", lambda: {})
    monkeypatch.setattr(appmod, "_safe_get_tf_system_info", lambda: {})
    monkeypatch.setattr(appmod, "_safe_get_tf_gpu_info", lambda: {})
    monkeypatch.setattr(
        appmod.tui_builders, "build_system_markdown", _fake_build_system_markdown
    )

    assert appmod._build_system_markdown() == "ok"
    assert captured["system_info"] == {}


@pytest.mark.parametrize(
    ("button_id", "method_name", "args", "kwargs", "is_async"),
    [
        ("btn-refresh-overview", "action_refresh_overview", (), {}, False),
        ("btn-log-system", "run_cli_command", ("gpumemprof info",), {}, True),
        (
            "btn-log-pytorch",
            "run_cli_command",
            ("gpumemprof monitor --duration 30 --interval 0.5",),
            {},
            True,
        ),
        (
            "btn-log-tensorflow",
            "run_cli_command",
            ("tfmemprof monitor --duration 30 --interval 0.5",),
            {},
            True,
        ),
        (
            "btn-log-diagnose",
            "run_cli_command",
            ("gpumemprof diagnose --duration 0 --output artifacts/tui_diagnose",),
            {},
            True,
        ),
        ("btn-run-pytorch", "run_pytorch_sample", (), {}, True),
        ("btn-run-tf", "run_tensorflow_sample", (), {}, True),
        (
            "btn-run-oom-scenario",
            "run_cli_command",
            (
                "python -m examples.scenarios.oom_flight_recorder_scenario --mode simulated",
            ),
            {},
            True,
        ),
        (
            "btn-run-cap-matrix",
            "run_cli_command",
            (
                "python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated --skip-tui",
            ),
            {},
            True,
        ),
        ("btn-cli-run", "run_cli_command", ("custom command",), {}, True),
        ("btn-cli-cancel", "cancel_cli_command", (), {}, True),
        ("btn-start-tracking", "start_live_tracking", (), {}, True),
        ("btn-stop-tracking", "stop_live_tracking", (), {}, False),
        ("btn-toggle-watchdog", "toggle_auto_cleanup", (), {}, False),
        ("btn-force-cleanup", "force_cleanup", (), {}, False),
        (
            "btn-force-cleanup-aggressive",
            "force_cleanup",
            (),
            {"aggressive": True},
            False,
        ),
        ("btn-export-csv", "export_tracker_events", ("csv",), {}, True),
        ("btn-export-json", "export_tracker_events", ("json",), {}, True),
        ("btn-apply-thresholds", "apply_thresholds", (), {}, False),
        ("btn-clear-monitor-log", "clear_monitor_log", (), {}, False),
        ("btn-refresh-visual", "refresh_visualizations", (), {}, True),
        ("btn-visual-png", "generate_visual_plot", ("png",), {}, True),
        ("btn-visual-html", "generate_visual_plot", ("html",), {}, True),
        ("btn-diag-load-live", "load_diagnostics_live", (), {}, True),
        ("btn-diag-load-artifacts", "load_diagnostics_artifacts", (), {}, True),
        ("btn-diag-refresh", "refresh_diagnostics", (), {}, True),
        ("btn-diag-apply-session", "apply_diagnostics_session_selection", (), {}, True),
        ("btn-diag-apply-filter", "apply_diagnostics_rank_filter", (), {}, False),
        ("btn-diag-reset-filter", "reset_diagnostics_rank_filter", (), {}, False),
        ("btn-refresh-pt-profiles", "refresh_pytorch_profiles", (), {}, True),
        ("btn-clear-pt-profiles", "clear_pytorch_profiles", (), {}, True),
        ("btn-refresh-tf-profiles", "refresh_tensorflow_profiles", (), {}, True),
        ("btn-clear-tf-profiles", "clear_tensorflow_profiles", (), {}, True),
    ],
)
def test_button_dispatch_preserves_action_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
    button_id: str,
    method_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    is_async: bool,
) -> None:
    app = appmod.GPUMemoryProfilerTUI()
    monkeypatch.setattr(
        app, "cli_command_input", Mock(value="custom command"), raising=False
    )
    handler = AsyncMock() if is_async else Mock()
    monkeypatch.setattr(app, method_name, handler)

    asyncio.run(app.on_button_pressed(Button.Pressed(Button(id=button_id))))

    handler.assert_called_once_with(*args, **kwargs)
    if is_async:
        handler.assert_awaited_once_with(*args, **kwargs)


@pytest.mark.parametrize("button_id", [None, "btn-unrecognized"])
def test_button_dispatch_ignores_unknown_buttons(button_id: str | None) -> None:
    app = appmod.GPUMemoryProfilerTUI()

    asyncio.run(app.on_button_pressed(Button.Pressed(Button(id=button_id))))


@pytest.mark.parametrize(
    ("timeline", "expected"),
    [
        (
            {},
            [
                ("Samples", "0"),
                ("Duration (s)", "-"),
                ("Allocated Max", "-"),
                ("Reserved Max", "-"),
                ("Allocated Latest", "-"),
                ("Reserved Latest", "-"),
            ],
        ),
        (
            {"timestamps": [1.0, 3.5], "allocated": [1048576, 3145728], "reserved": []},
            [
                ("Samples", "2"),
                ("Duration (s)", "2.5"),
                ("Allocated Max", "3.00 MB"),
                ("Reserved Max", "0 B"),
                ("Allocated Latest", "3.00 MB"),
                ("Reserved Latest", "0 B"),
            ],
        ),
        (
            {
                "timestamps": [1.0, 3.5],
                "allocated": [1048576, 3145728],
                "reserved": [1048576, 5242880],
            },
            [
                ("Samples", "2"),
                ("Duration (s)", "2.5"),
                ("Allocated Max", "3.00 MB"),
                ("Reserved Max", "5.00 MB"),
                ("Allocated Latest", "3.00 MB"),
                ("Reserved Latest", "5.00 MB"),
            ],
        ),
    ],
)
def test_legacy_timeline_rows_keep_their_order_and_placeholders(
    timeline: dict[str, Any], expected: list[tuple[str, str]]
) -> None:
    from stormlog.tui.builders import build_legacy_timeline_rows

    assert build_legacy_timeline_rows(timeline) == expected


def test_repeated_alerts_share_one_row_with_their_count() -> None:
    from stormlog.tui.monitor import TrackerEventView

    def alert(at: float, kind: str, message: str) -> TrackerEventView:
        return TrackerEventView(at, kind, message, "-", "-", "-")

    app = appmod.GPUMemoryProfilerTUI()
    app.recent_alerts = []
    app._capture_alerts(
        [
            alert(1.0, "warning", "High fragmentation: 95.5%"),
            alert(2.0, "warning", "WARNING: Memory usage at 81.0%"),
            alert(3.0, "info", "Collected telemetry sample."),
            alert(4.0, "warning", "High fragmentation: 96.0%"),
        ]
    )
    app._capture_alerts([alert(5.0, "warning", "High fragmentation: 95.7%")])

    assert [(a["message"], a["count"]) for a in app.recent_alerts] == [
        ("WARNING: Memory usage at 81.0%", 1),
        ("High fragmentation: 95.7%", 3),
    ]
    assert app.recent_alerts[-1]["timestamp"] == 5.0


def test_the_indicator_table_names_the_phase() -> None:
    from textual.app import App, ComposeResult

    from stormlog.tui.distributed_diagnostics import AnomalyIndicator
    from stormlog.tui.widgets.tables import AnomalySummaryTable

    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield AnomalySummaryTable()

    async def scenario() -> None:
        app = _App()
        async with app.run_test() as pilot:
            table = app.query_one(AnomalySummaryTable)
            table.update_rows(
                [
                    AnomalyIndicator(
                        kind="earliest",
                        rank=0,
                        severity="warning",
                        timestamp_ns=0,
                        signal="alert:warning",
                        details="Large allocation detected",
                        phase_path="epoch 2 > train",
                    )
                ]
            )
            await pilot.pause()
            assert table.get_row_at(0)[4] == "epoch 2 > train"
            table.update_rows([])
            await pilot.pause()
            assert table.get_row_at(0)[5] == "No anomaly indicators detected."

    asyncio.run(scenario())


def test_different_errors_keep_their_own_rows() -> None:
    from stormlog.tui.monitor import TrackerEventView

    app = appmod.GPUMemoryProfilerTUI()
    app.recent_alerts = []
    app._capture_alerts(
        [
            TrackerEventView(1.0, "error", "Tracking error: CUDA busy", "-", "-", "-"),
            TrackerEventView(
                2.0, "error", "Tracking error: device lost", "-", "-", "-"
            ),
            TrackerEventView(3.0, "error", "Tracking error: CUDA busy", "-", "-", "-"),
        ]
    )

    assert [(a["message"], a["count"]) for a in app.recent_alerts] == [
        ("Tracking error: device lost", 1),
        ("Tracking error: CUDA busy", 2),
    ]
