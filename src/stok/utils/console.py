from __future__ import annotations

import sys
from typing import Optional

from tqdm import tqdm


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
        # Pretty rendering only if both enabled and main process
        self.is_main: bool = bool(is_main)
        self.enabled: bool = bool(enabled) and self.is_main
        self.file = file or sys.stdout
        self._eval_lines: list[tqdm] = []

        if self.enabled:
            self.step_bar: Optional[tqdm] = tqdm(
                total=total_steps,
                initial=initial_step,
                unit=unit,
                position=0,
                leave=True,
                dynamic_ncols=True,
                file=self.file,
                disable=not self.enabled,
            )
            self.train_line: Optional[tqdm] = tqdm(
                total=0,
                position=1,
                bar_format="{desc}",
                leave=False,
                dynamic_ncols=True,
                file=self.file,
                disable=not self.enabled,
            )
        else:
            self.step_bar = None
            self.train_line = None

    def step(self, n: int = 1):
        if self.step_bar is not None:
            self.step_bar.update(n)

    def set_step(self, step: int):
        if self.step_bar is not None:
            self.step_bar.n = step
            self.step_bar.refresh()

    def train(self, msg: str):
        if self.train_line is not None:
            self.train_line.set_description_str(msg)
            self.train_line.refresh()

    def eval(self, msg: str):
        if not self.enabled:
            # When pretty is disabled, emit a plain line for eval on main
            if self.is_main:
                self.print(msg)
            return
        line = tqdm(
            total=0,
            position=2 + len(self._eval_lines),
            bar_format="{desc}",
            leave=True,
            dynamic_ncols=True,
            file=self.file,
            disable=not self.enabled,
        )
        line.set_description_str(msg)
        line.refresh()
        self._eval_lines.append(line)

    def print(self, msg: str):
        # Always print on main, regardless of pretty enabled
        if self.is_main:
            tqdm.write(str(msg), file=self.file)

    def close(self):
        if self.train_line is not None:
            self.train_line.close()
            self.train_line = None
        if self.step_bar is not None:
            self.step_bar.close()
            self.step_bar = None

