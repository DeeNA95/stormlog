"""Markdown panel widgets used by the Textual TUI."""

from __future__ import annotations

from typing import Any, Callable

from textual.widgets import Markdown


class MarkdownPanel(Markdown):
    """Markdown built by ``builder``, rebuilt by ``refresh_content``.

    The first build is the widget's initial document. Markdown renders that
    when it mounts, after any ``on_mount`` handler here, so an update made in
    ``on_mount`` would be replaced by an empty document.
    """

    def __init__(self, builder: Callable[[], str], **kwargs: Any) -> None:
        super().__init__(builder(), **kwargs)
        self.builder = builder

    def refresh_content(self) -> None:
        self.update(self.builder())
