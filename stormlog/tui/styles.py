"""Styles shared by the Textual TUI app."""

from __future__ import annotations

TUI_APP_CSS = """
    TabbedContent {
        padding: 1 1 0 1;
    }

    TabPane {
        padding: 1 0 0 0;
    }

    Markdown {
        margin: 0 0 1 0;
        padding: 0 1;
    }

    Button {
        margin: 0 1 0 0;
        min-width: 10;
        width: auto;
        text-style: bold;
    }

    /* A row of buttons, inputs and their labels. */
    .controls {
        layout: horizontal;
        height: auto;
        margin: 0 0 1 0;
    }

    .controls Label {
        height: 3;
        width: auto;
        content-align: left middle;
        padding: 0 1 0 0;
    }

    .controls Input {
        width: 1fr;
        margin: 0 1 0 0;
    }

    DataTable {
        height: auto;
        max-height: 14;
        border: solid $panel-lighten-2;
        margin: 0 0 1 0;
    }

    RichLog {
        height: 12;
        border: solid $panel-lighten-2;
    }

    #overview-welcome {
        border: round $primary;
        padding: 0 1;
        margin: 0 0 1 0;
        background: $panel;
        text-align: center;
        text-style: bold;
        color: $accent;
        height: auto;
        content-align: center middle;
    }

    #overview-columns {
        height: auto;
    }

    #overview MarkdownH3,
    #welcome-info MarkdownH3 {
        margin: 1 0 0 0;
        color: $accent;
    }

    #overview,
    #welcome-info {
        width: 1fr;
        height: auto;
        border: solid $primary;
        padding: 0 1;
        margin: 0 1 1 0;
    }

    #monitor-thresholds {
        layout: horizontal;
        width: auto;
        height: auto;
    }

    #monitor-thresholds Input {
        width: 9;
    }

    #diagnostics-session-input {
        width: 30;
    }

    #diagnostics-rank-filter {
        width: 20;
    }

    #timeline-canvas,
    #diagnostics-timeline-canvas,
    #mlx-timeline-canvas {
        height: auto;
        min-height: 4;
        border: solid $panel-lighten-2;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    #command-log {
        height: 1fr;
        min-height: 10;
    }

    #cli-loader {
        height: 3;
    }
"""
