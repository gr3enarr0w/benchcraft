"""Regression test for a CRITICAL follow-up finding on top of Finding 2
(`lazycore.sandbox.seatbelt`'s "never pickle the caller's callable" fix).

The original Finding-2 fix replaced ``pickle.dumps(func)`` with a
verification step that called ``importlib.import_module(func.__module__)``
directly in this (trusted, host) process to confirm ``func`` re-resolves to
itself. That verification step was itself vulnerable to the same class of
bug it was meant to close: ``func.__module__`` is a plain, freely-writable
string attribute on any function object -- nothing stops a caller from
doing ``some_func.__module__ = "some/malicious/module/path"`` before
calling ``run_callable``, at which point ``importlib.import_module`` would
import and execute that module's top-level code *in the trusted host
process*, before the sandbox even starts and before the identity check has
a chance to reject anything.

The fix replaces that import-based verification with a check against
``func.__globals__`` -- the dict object that *is* the function's actual
defining namespace, and which (unlike ``__module__``/``__qualname__``)
cannot be reassigned after the fact by simple attribute assignment. No
import of the (possibly-tampered) module name ever happens on the host
side; the module is only ever imported later, inside the sandboxed child
process's runner script.

This test file deliberately does NOT invoke ``/usr/bin/sandbox-exec`` (no
macOS skip marker) -- the vulnerability and its fix both live entirely in
host-side validation logic that runs identically on every platform, before
any subprocess is spawned.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from lazycore.sandbox.seatbelt import SeatbeltSandboxExecutor

# A real, legitimately-importable-but-irrelevant stdlib module. Used as the
# "malicious" __module__ target below to prove the vulnerability is closed
# even when the tampered name points at something real and harmless (not
# just a nonexistent name, which would be a weaker proof -- a nonexistent
# name failing is not by itself evidence that import was never attempted).
_IRRELEVANT_BUT_REAL_MODULE = "colorsys"


def _victim_module_level_function() -> int:
    """A real, otherwise-ordinary module-level function (flat qualname, no
    ``<locals>``/``<lambda>``) used as the tampering target below. It must
    be defined at module level (not nested inside another function) so
    that only its ``__module__`` is tampered -- its ``__qualname__`` stays
    flat, which is what lets execution reach the new ``__globals__``-based
    check instead of being rejected earlier by the lambda/closure check.
    """
    return 1


def _make_tampered_function(monkeypatch: pytest.MonkeyPatch, tampered_module_name: str):
    """Return :func:`_victim_module_level_function` with its ``__module__``
    overwritten (via ``monkeypatch``, so it is automatically restored at
    the end of the test regardless of outcome) to point somewhere else,
    simulating an attacker (or attacker-influenced code upstream of
    ``run_callable``) tampering with the attribute after the function
    object already exists.

    This mutation is the crux of the vulnerability: ``__module__`` is just
    a plain writable string attribute, not an intrinsic, tamper-proof
    property of the function object.
    """
    monkeypatch.setattr(
        _victim_module_level_function, "__module__", tampered_module_name, raising=True
    )
    return _victim_module_level_function


def test_tampered_module_attribute_is_rejected_with_clear_value_error(monkeypatch):
    """A function whose __module__ has been reassigned since definition
    must be rejected with a clear ValueError -- __globals__["__name__"]
    (this test module's real name) will not match the tampered
    __module__ (`colorsys`), so validation fails closed.
    """
    executor = SeatbeltSandboxExecutor()
    tampered = _make_tampered_function(monkeypatch, _IRRELEVANT_BUT_REAL_MODULE)

    with pytest.raises(ValueError, match="__globals__"):
        executor._resolve_module_level_function(tampered)


def test_tampering_never_actually_imports_the_tampered_module_name(monkeypatch):
    """The critical proof: validating a tampered callable must NEVER import
    the tampered module name as a side effect. Monkeypatches
    importlib.import_module to raise AssertionError if invoked at all
    during validation, and separately confirms the tampered module name is
    absent from sys.modules before and after (belt-and-suspenders, in case
    something imports it via a path this test didn't anticipate).
    """
    assert _IRRELEVANT_BUT_REAL_MODULE not in sys.modules, (
        f"{_IRRELEVANT_BUT_REAL_MODULE!r} must not already be imported "
        "before this test runs, or the sys.modules assertion below would "
        "be meaningless."
    )

    def _boom(name, *args, **kwargs):
        raise AssertionError(
            "_resolve_module_level_function() must NEVER call "
            f"importlib.import_module (attempted with name={name!r}) -- "
            "host-side validation must reject a tampered __module__ purely "
            "via __globals__ introspection, without ever importing "
            "anything, real or fake."
        )

    monkeypatch.setattr(importlib, "import_module", _boom)

    executor = SeatbeltSandboxExecutor()
    tampered = _make_tampered_function(monkeypatch, _IRRELEVANT_BUT_REAL_MODULE)

    with pytest.raises(ValueError):
        executor._resolve_module_level_function(tampered)

    # Belt-and-suspenders: the tampered module name must never have landed
    # in sys.modules as a side effect of the validation call above.
    assert _IRRELEVANT_BUT_REAL_MODULE not in sys.modules


def test_tampered_module_pointing_at_nonexistent_module_is_also_rejected(monkeypatch):
    """Same tampering scenario, but pointing __module__ at a module name
    that does not exist at all -- confirms the fix does not depend on the
    tampered name happening to resolve to something importable; it never
    attempts to import it either way.
    """
    executor = SeatbeltSandboxExecutor()
    tampered = _make_tampered_function(
        monkeypatch, "some.totally.nonexistent.module.path"
    )

    with pytest.raises(ValueError, match="__globals__"):
        executor._resolve_module_level_function(tampered)

    assert "some.totally.nonexistent.module.path" not in sys.modules
    assert "some" not in sys.modules


def test_untampered_module_level_function_still_resolves_correctly():
    """Sanity/non-regression check: an ordinary, non-tampered module-level
    function (this test module's own top-level function, defined below)
    still resolves successfully via the new __globals__-based check.
    """
    executor = SeatbeltSandboxExecutor()
    module_name, qualname = executor._resolve_module_level_function(
        _legitimate_module_level_function
    )
    assert module_name == __name__
    assert qualname == "_legitimate_module_level_function"


def _legitimate_module_level_function() -> int:
    """A real, untampered module-level function used by
    ``test_untampered_module_level_function_still_resolves_correctly``."""
    return 42


def test_dotted_qualname_is_rejected_as_unsupported():
    """A function whose __qualname__ contains a dot (e.g. a method or a
    function nested under a class) is rejected outright -- the
    __globals__-based check can only verify flat module-level names via a
    direct dict lookup, unlike the old attribute-chasing approach which
    could (insecurely) walk a dotted path. This is a deliberate tightening:
    run_callable()'s documented contract is "module-level functions or
    functools.partial wrapping one" -- methods were never actually
    supported in a safe way.
    """

    class Container:
        def method(self) -> int:
            return 1

    executor = SeatbeltSandboxExecutor()
    with pytest.raises(ValueError, match="module-level"):
        executor._resolve_module_level_function(Container.method)
