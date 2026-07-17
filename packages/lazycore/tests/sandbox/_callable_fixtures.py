"""Picklable module-level callables used by test_seatbelt.py's run_callable tests.

These must live in their own module (not inline in test_seatbelt.py) so
that :mod:`pickle` can reference them by ``(module, qualname)`` and the
sandboxed subprocess -- which runs a fresh Python interpreter, not the
pytest process -- can re-import this module (via ``PYTHONPATH`` pointed at
this directory) to unpickle them. This file is not itself a test module
(no ``test_`` prefix), so pytest will not try to collect it.
"""

from __future__ import annotations


def compute_answer() -> int:
    return 6 * 7


def raise_value_error() -> None:
    raise ValueError("boom from sandboxed callable")
