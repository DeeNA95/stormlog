"""Capture the TUI screenshots used by the README and the TUI guide.

Drives the shipped Textual app headless while a small PyTorch training loop
runs in the same process, so live tracking, the timeline and diagnostics show
real samples rather than placeholders. Each tab is saved as an SVG.

Run it on the machine whose numbers the screenshots should show; the README's
come from an NVIDIA L4 through ``modal_readme_media.py``::

    python -m examples.scenarios.tui_screenshots --output docs
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Sequence

SIZE = (140, 44)

# Tab title, file stem. The MLX tab needs native MLX: see ``--mlx``.
TABS = (
    ("Overview", "tui-overview-current"),
    ("PyTorch", "tui-pytorch-current"),
    ("Monitoring", "tui-monitoring-current"),
    ("Visualizations", "tui-visualizations-current"),
    ("Diagnostics", "tui-diagnostics-current"),
    ("CLI & Actions", "tui-cli-actions-current"),
)


class TrainingLoop:
    """A small transformer trained with varying batch sizes in a thread.

    The batch size cycles, so allocated memory rises and falls the way a real
    training job's does with its data, and the tracker has something to show.
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="training", daemon=True)
        self.steps = 0
        self.error: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)

    def _run(self) -> None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.manual_seed(0)
            layer = torch.nn.TransformerEncoderLayer(
                d_model=512, nhead=8, dim_feedforward=2048, batch_first=True
            )
            model = torch.nn.TransformerEncoder(layer, num_layers=6).to(device)
            head = torch.nn.Linear(512, 1000).to(device)
            params = list(model.parameters()) + list(head.parameters())
            optimizer = torch.optim.AdamW(params, lr=1e-4)
            batches = (8, 16, 32, 64, 96, 64, 32, 16)
            while not self._stop.is_set():
                batch = batches[self.steps % len(batches)]
                x = torch.randn(batch, 256, 512, device=device)
                target = torch.randint(0, 1000, (batch, 256), device=device)
                loss = torch.nn.functional.cross_entropy(
                    head(model(x)).reshape(-1, 1000), target.reshape(-1)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if device == "cuda":
                    torch.cuda.synchronize()
                self.steps += 1
                time.sleep(0.25)
        except BaseException as exc:  # reported by the caller
            self.error = exc


def _pane_id(app: Any, title: str) -> str:
    from textual.widgets import TabPane

    for pane in app.query(TabPane):
        label = getattr(pane, "_title", None) or getattr(pane, "title", "")
        if getattr(label, "plain", str(label)) == title:
            return str(pane.id)
    raise LookupError(f"no tab titled {title!r}")


async def _show(app: Any, pilot: Any, title: str) -> None:
    from textual.widgets import TabbedContent

    app.query_one(TabbedContent).active = _pane_id(app, title)
    await pilot.pause(0.5)


async def _press(app: Any, pilot: Any, button_id: str, settle: float) -> None:
    """Press a button by ID: a click would land elsewhere if it is scrolled off."""
    from textual.widgets import Button

    app.query_one(f"#{button_id}", Button).press()
    await pilot.pause(settle)


async def _capture(output: Path, track_seconds: float) -> list[Path]:
    from stormlog.tui.app import GPUMemoryProfilerTUI

    output.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    app = GPUMemoryProfilerTUI()
    training = TrainingLoop()

    def save(title: str) -> None:
        stem = dict(TABS)[title]
        path = output / f"{stem}.svg"
        app.save_screenshot(filename=path.name, path=str(output))
        saved[title] = path

    async with app.run_test(headless=True, size=SIZE) as pilot:
        await pilot.pause(1.0)
        save("Overview")

        training.start()
        await _show(app, pilot, "Monitoring")
        await _press(app, pilot, "btn-start-tracking", 0.5)
        deadline = time.monotonic() + track_seconds
        while time.monotonic() < deadline:
            await pilot.pause(1.0)
        save("Monitoring")

        await _show(app, pilot, "Visualizations")
        await _press(app, pilot, "btn-refresh-visual", 1.5)
        save("Visualizations")

        await _show(app, pilot, "Diagnostics")
        await _press(app, pilot, "btn-diag-load-live", 2.0)
        save("Diagnostics")

        await _show(app, pilot, "CLI & Actions")
        await _press(app, pilot, "btn-run-pytorch", 8.0)
        save("CLI & Actions")

        await _show(app, pilot, "PyTorch")
        await _press(app, pilot, "btn-refresh-pt-profiles", 1.5)
        save("PyTorch")

        await _show(app, pilot, "Monitoring")
        await _press(app, pilot, "btn-stop-tracking", 0.5)

    training.stop()
    if training.error is not None:
        raise RuntimeError("the training loop failed") from training.error
    return [saved[title] for title, _ in TABS]


async def _capture_mlx(output: Path) -> Path:
    """The MLX tab after Runtime Info and a native sample (Apple Silicon only)."""
    from stormlog.tui.app import GPUMemoryProfilerTUI

    output.mkdir(parents=True, exist_ok=True)
    app = GPUMemoryProfilerTUI(framework="mlx")
    path = output / "tui-mlx-current.svg"
    async with app.run_test(headless=True, size=SIZE) as pilot:
        await pilot.pause(1.0)
        await _show(app, pilot, "MLX")
        await _press(app, pilot, "btn-mlx-info", 2.0)
        await _press(app, pilot, "btn-run-mlx", 6.0)
        app.save_screenshot(filename=path.name, path=str(output))
    return path


def capture_mlx(output: Path) -> Path:
    """Save the MLX tab under ``output``; needs native MLX on Apple Silicon."""
    return asyncio.run(_capture_mlx(output))


def capture(output: Path, track_seconds: float = 30.0) -> list[Path]:
    """Save one SVG per tab under ``output`` and return their paths."""
    return asyncio.run(_capture(output, track_seconds))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("docs"))
    parser.add_argument("--track-seconds", type=float, default=30.0)
    parser.add_argument(
        "--mlx",
        action="store_true",
        help="Capture only the MLX tab, with a native sample (Apple Silicon)",
    )
    args = parser.parse_args(argv)
    if args.mlx:
        print(capture_mlx(args.output))
        return 0
    for path in capture(args.output, args.track_seconds):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
