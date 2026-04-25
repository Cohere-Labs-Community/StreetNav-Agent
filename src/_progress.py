"""Tiny per-stage progress helper.

Each tool calls ``start(name, info)`` once on entry and ``done(summary)``
once on exit. Output is two short lines per stage so the terminal stays
responsive (no multi-minute blocks of silence) without flooding it.
"""
from __future__ import annotations

import time

_start_t: float = 0.0
_name: str = ""


def start(name: str, info: str = "") -> None:
    global _start_t, _name
    _start_t = time.perf_counter()
    _name = name
    msg = f"\u25b6 {name}"
    if info:
        msg += f"  ({info})"
    print(msg, flush=True)


def done(summary: str = "") -> None:
    dt = time.perf_counter() - _start_t
    msg = f"\u2713 {_name} done in {dt:.1f}s"
    if summary:
        msg += f"  \u2192  {summary}"
    print(msg, flush=True)


def fail(reason: str) -> None:
    dt = time.perf_counter() - _start_t
    print(f"\u2717 {_name} failed in {dt:.1f}s  \u2192  {reason}", flush=True)


def info(msg: str) -> None:
    print(f"  {msg}", flush=True)
