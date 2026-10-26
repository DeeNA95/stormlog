"""Capture the TUI screenshots used by the README and the TUI guide.

Drives the shipped Textual app headless while ResNet-18 trains on CIFAR-10 in
the same process. Live tracking follows the whole run. Each epoch's training
and evaluation run inside tracker phases (``epoch N`` > ``train`` / ``eval``)
and the context profiler, so Diagnostics attributes its indicators to phases
and the PyTorch tab lists a profile per phase; each epoch's loss, accuracy,
throughput and peak memory go to the TUI's log. Each tab is saved as an SVG.

CIFAR-10 is decoded once from its Hugging Face mirror into ``--data``. The
README's screenshots come from an NVIDIA L4 through ``modal_readme_media.py``::

    python -m examples.scenarios.tui_screenshots --output docs --epochs 6
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import threading
import time
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Sequence, cast

SIZE = (140, 44)
# Diagnostics is the README's training image: taller, so the timeline, its
# markers, the indicators and the log all fit below the controls.
TALL = (140, 58)
DATA = Path.home() / ".cache" / "stormlog" / "cifar10"
CIFAR10_MIRROR = "uoft-cs/cifar10"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

# Tab title, file stem. The MLX tab needs native MLX: see ``--mlx``.
TABS = (
    ("Overview", "tui-overview-current"),
    ("PyTorch", "tui-pytorch-current"),
    ("Monitoring", "tui-monitoring-current"),
    ("Visualizations", "tui-visualizations-current"),
    ("Diagnostics", "tui-diagnostics-current"),
    ("CLI & Actions", "tui-cli-actions-current"),
)


def load_cifar10(root: Path = DATA) -> dict[str, dict[str, Any]]:
    """CIFAR-10 as uint8 image and int64 label tensors, decoded on first use."""
    import torch

    if not all((root / f"{split}.pt").exists() for split in ("train", "test")):
        _decode_cifar10(root)
    return {
        split: torch.load(root / f"{split}.pt", weights_only=True)
        for split in ("train", "test")
    }


def _decode_cifar10(root: Path) -> None:
    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        path = hf_hub_download(
            CIFAR10_MIRROR,
            f"plain_text/{split}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        frame = pd.read_parquet(path)
        image_column = next(
            column
            for column in frame.columns
            if isinstance(frame[column].iloc[0], dict)
        )
        label_column = next(
            column for column in frame.columns if column != image_column
        )
        images = np.stack(
            [
                np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
                for cell in frame[image_column]
            ]
        )
        labels = frame[label_column].to_numpy().astype(np.int64)
        torch.save(
            {"images": torch.from_numpy(images), "labels": torch.from_numpy(labels)},
            root / f"{split}.pt",
        )


class CifarTraining:
    """ResNet-18 on CIFAR-10 for a few epochs, in a thread.

    The data lives on the device and is augmented there (random crop and
    flip), so the GPU is never starved by a loader and memory follows the
    model: high while training, lower while evaluating without gradients.
    """

    def __init__(
        self,
        data: dict[str, dict[str, Any]],
        *,
        epochs: int,
        batch_size: int,
        phase: Callable[[str], ContextManager[Any]],
        log: Callable[[str], None],
    ) -> None:
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self._phase = phase
        self._log = log
        self.epoch = 0
        self.stage = ""
        self.error: BaseException | None = None
        self.done = threading.Event()
        self._thread = threading.Thread(target=self._run, name="training", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _run(self) -> None:
        try:
            self._train()
        except BaseException as exc:  # reported by the caller
            self.error = exc
        finally:
            self.done.set()

    def _train(self) -> None:
        import torch
        import torch.nn.functional as F
        import torchvision

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        torch.manual_seed(0)
        model = torchvision.models.resnet18(num_classes=10)
        model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)  # 32x32 input
        model.maxpool = torch.nn.Identity()
        model = model.to(device, memory_format=torch.channels_last)

        train_x = self.data["train"]["images"].to(device)
        train_y = self.data["train"]["labels"].to(device)
        test_x = self.data["test"]["images"].to(device)
        test_y = self.data["test"]["labels"].to(device)
        mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) * 255
        std = torch.tensor(STD, device=device).view(1, 3, 1, 1) * 255

        def prepare(images: Any, augment: bool) -> Any:
            x = images.permute(0, 3, 1, 2).float()
            if augment:
                x = F.pad(x, (4, 4, 4, 4), mode="reflect")
                top, left = torch.randint(0, 9, (2,)).tolist()
                x = x[:, :, top : top + 32, left : left + 32]
                flip = torch.rand(x.shape[0], device=device) < 0.5
                x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
            return ((x - mean) / std).contiguous(memory_format=torch.channels_last)

        batch = self.batch_size
        steps = (len(train_x) + batch - 1) // batch
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4
        )
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.2, total_steps=self.epochs * steps
        )
        cuda = device.type == "cuda"

        def autocast() -> ContextManager[Any]:
            if cuda:
                return cast(
                    ContextManager[Any], torch.autocast("cuda", dtype=torch.bfloat16)
                )
            return contextlib.nullcontext()

        def profiled(name: str) -> ContextManager[Any]:
            if cuda:
                from stormlog.context_profiler import profile_context

                return profile_context(name)
            return contextlib.nullcontext()

        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            with self._phase(f"epoch {epoch}"):
                self.stage = "train"
                with self._phase("train"), profiled("train epoch"):
                    model.train()
                    started = time.perf_counter()
                    order = torch.randperm(len(train_x), device=device)
                    loss_sum = torch.zeros((), device=device)
                    correct = torch.zeros((), device=device, dtype=torch.long)
                    for step in range(steps):
                        index = order[step * batch : (step + 1) * batch]
                        targets = train_y[index]
                        with autocast():
                            logits = model(prepare(train_x[index], augment=True))
                            loss = F.cross_entropy(logits, targets, label_smoothing=0.1)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        schedule.step()
                        loss_sum += loss.detach() * len(index)
                        correct += (logits.argmax(1) == targets).sum()
                    if cuda:
                        torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                    peak = torch.cuda.max_memory_allocated() if cuda else 0
                self.stage = "eval"
                with self._phase("eval"), profiled("evaluate"):
                    model.eval()
                    right = torch.zeros((), device=device, dtype=torch.long)
                    with torch.inference_mode():
                        for start in range(0, len(test_x), 1000):
                            with autocast():
                                logits = model(
                                    prepare(test_x[start : start + 1000], augment=False)
                                )
                            right += (
                                logits.argmax(1) == test_y[start : start + 1000]
                            ).sum()
            images = len(train_x)
            self._log(
                f"epoch {epoch}/{self.epochs}: loss {loss_sum.item() / images:.3f}, "
                f"train acc {correct.item() / images:.1%}, "
                f"test acc {right.item() / len(test_x):.1%}, "
                f"{images / seconds:,.0f} img/s, train peak {peak / 2**30:.2f} GiB"
            )


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


async def _wait(pilot: Any, condition: Callable[[], bool], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("the training run did not get there in time")
        await pilot.pause(0.5)


def _tracker_phase(app: Any) -> Callable[[str], ContextManager[Any]]:
    """The live tracker's phase(), or a no-op where the tracker has none."""

    @contextlib.contextmanager
    def phase(name: str) -> Iterator[None]:
        session = getattr(app, "tracker_session", None)
        tracker = getattr(session, "_tracker", None)
        if tracker is None or not hasattr(tracker, "phase"):
            yield
            return
        with tracker.phase(name):
            yield

    return phase


async def _capture(
    output: Path, data: dict[str, dict[str, Any]], epochs: int, batch_size: int
) -> list[Path]:
    from stormlog.tui.app import GPUMemoryProfilerTUI

    output.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    app = GPUMemoryProfilerTUI()

    def save(title: str) -> None:
        stem = dict(TABS)[title]
        path = output / f"{stem}.svg"
        app.save_screenshot(filename=path.name, path=str(output))
        saved[title] = path

    def log(message: str) -> None:
        app.call_from_thread(app.log_message, "ResNet-18 on CIFAR-10", message)

    training = CifarTraining(
        data,
        epochs=epochs,
        batch_size=batch_size,
        phase=_tracker_phase(app),
        log=log,
    )
    async with app.run_test(headless=True, size=SIZE) as pilot:
        await pilot.pause(1.0)
        save("Overview")

        await _show(app, pilot, "Monitoring")
        await _press(app, pilot, "btn-start-tracking", 1.0)
        training.start()
        # Live, part way through: the middle epoch's training.
        middle = max(1, (epochs + 1) // 2)
        await _wait(
            pilot,
            lambda: training.done.is_set()
            or (training.epoch >= middle and training.stage == "train"),
            timeout=1800,
        )
        await pilot.pause(3.0)
        save("Monitoring")

        await _wait(pilot, training.done.is_set, timeout=3600)
        await pilot.pause(2.0)
        if training.error is not None:
            raise RuntimeError("the training run failed") from training.error

        await _show(app, pilot, "Visualizations")
        await _press(app, pilot, "btn-refresh-visual", 1.5)
        save("Visualizations")

        await _show(app, pilot, "Diagnostics")
        await _press(app, pilot, "btn-diag-load-live", 2.0)
        await pilot.resize_terminal(*TALL)
        await pilot.pause(1.0)
        # The timeline takes its width when drawn, so redraw it at this size.
        await _press(app, pilot, "btn-diag-refresh", 2.0)
        save("Diagnostics")
        await pilot.resize_terminal(*SIZE)
        await pilot.pause(1.0)

        await _show(app, pilot, "PyTorch")
        await _press(app, pilot, "btn-refresh-pt-profiles", 1.5)
        save("PyTorch")

        await _show(app, pilot, "CLI & Actions")
        save("CLI & Actions")

        await _show(app, pilot, "Monitoring")
        await _press(app, pilot, "btn-stop-tracking", 0.5)

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


def capture(
    output: Path, *, data: Path = DATA, epochs: int = 6, batch_size: int = 256
) -> list[Path]:
    """Train, save one SVG per tab under ``output``, and return their paths."""
    return asyncio.run(_capture(output, load_cifar10(data), epochs, batch_size))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("docs"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--mlx",
        action="store_true",
        help="Capture only the MLX tab, with a native sample (Apple Silicon)",
    )
    args = parser.parse_args(argv)
    if args.mlx:
        print(capture_mlx(args.output))
        return 0
    paths = capture(
        args.output, data=args.data, epochs=args.epochs, batch_size=args.batch_size
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
