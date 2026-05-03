from __future__ import annotations

import io

from stok.utils.console import ConsoleLogger


def test_disabled_main_emits_eval_and_print():
    buf = io.StringIO()
    c = ConsoleLogger(total_steps=10, is_main=True, enabled=False, file=buf)
    c.eval("eval/val | step 1 | loss 0.5")
    c.print("checkpoint saved")
    c.close()
    out = buf.getvalue()
    assert "eval/val" in out
    assert "checkpoint saved" in out


def test_non_main_is_silent():
    buf = io.StringIO()
    c = ConsoleLogger(total_steps=10, is_main=False, enabled=True, file=buf)
    c.step()
    c.train("x")
    c.eval("y")
    c.print("z")
    c.close()
    assert buf.getvalue() == ""
