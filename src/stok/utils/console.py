from __future__ import annotations

import builtins
import sys
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text


class _TrainLine:
    def __init__(self) -> None:
        self.msg: str = ""

    def __rich__(self) -> Text:
        return Text(self.msg, no_wrap=True, overflow="ellipsis")


class _RateColumn(ProgressColumn):
    def __init__(self, unit: str) -> None:
        super().__init__()
        self.unit = unit

    def render(self, task) -> Text:
        if task.speed is None:
            return Text("--/s")
        return Text(f"{task.speed:>.2f} {self.unit}/s")


class ConsoleLogger:
    def __init__(
        self,
        *,
        total_steps: int,
        initial_step: int = 0,
        is_main: bool = True,
        enabled: bool = True,
        unit: str = "step",
        file=None,
    ):
        # Pretty rendering only if both enabled and main process.
        # Assumes exactly one rank reports is_main=True; concurrent Live
        # regions on the same stdout would scramble output.
        self.is_main: bool = bool(is_main)
        self.enabled: bool = bool(enabled) and self.is_main
        self.file = file or sys.stdout

        self._console: Optional[Console] = None
        self._progress: Optional[Progress] = None
        self._task_id = None
        self._train_line: Optional[_TrainLine] = None
        self._live: Optional[Live] = None

        # Live in-place rendering only when stdout is a real terminal. In
        # captured/piped contexts (CliRunner, log files) we degrade to
        # printing each train/eval/print update as its own line so that
        # intermediate state survives in scrollback.
        self._console = Console(file=self.file) if self.is_main else None
        use_live = self.enabled and self._console is not None and self._console.is_terminal

        if use_live:
            self._progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                TextColumn("•"),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("<"),
                TimeRemainingColumn(),
                TextColumn("•"),
                _RateColumn(unit),
                console=self._console,
                auto_refresh=False,
                expand=False,
            )
            self._task_id = self._progress.add_task(
                "", total=total_steps, completed=initial_step
            )
            self._train_line = _TrainLine()
            self._live = Live(
                Group(self._progress, self._train_line),
                console=self._console,
                refresh_per_second=10,
                transient=False,
            )
            self._live.start()

    def step(self, n: int = 1):
        if self._progress is not None:
            self._progress.advance(self._task_id, n)

    def set_step(self, step: int):
        if self._progress is not None:
            self._progress.update(self._task_id, completed=step)

    def train(self, msg: str):
        if self._train_line is not None:
            self._train_line.msg = msg
        elif self.enabled:
            # Non-TTY enabled mode: emit each train update as a line so
            # intermediate state shows up in captured/log output.
            self.print(msg)

    def eval(self, msg: str):
        if self._live is not None:
            self._live.console.print(msg)
            return
        if self.is_main:
            self.print(msg)

    def print(self, msg: str):
        if not self.is_main:
            return
        if self._live is not None:
            self._live.console.print(str(msg))
        elif self._console is not None:
            self._console.print(str(msg))
        else:
            builtins.print(str(msg), file=self.file, flush=True)

    def close(self):
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._progress = None
        self._train_line = None
