"""Tests for lazycore.sandbox.linux_stub.LinuxNamespaceSandboxExecutor.

This backend is a documented stub (see its module docstring for why): this
development machine is macOS, so these tests confirm the stub behaves
correctly *as a stub* on this non-Linux host -- they do not, and cannot,
validate real Linux namespace isolation.
"""

from __future__ import annotations

import platform

import pytest

from lazycore.sandbox.base import SandboxBackendUnavailableError
from lazycore.sandbox.linux_stub import LinuxNamespaceSandboxExecutor


def test_is_available_is_false_on_this_non_linux_machine():
    """is_available() returns False on this non-Linux (macOS) development host, since the stub only ever reports availability on Linux."""
    assert platform.system() != "Linux", (
        "This test asserts non-Linux behavior; re-evaluate if this suite "
        "is ever run on an actual Linux host."
    )
    executor = LinuxNamespaceSandboxExecutor()
    assert executor.is_available() is False


def test_run_command_raises_documented_unavailable_error():
    """run_command() always raises SandboxBackendUnavailableError mentioning it is a documented stub, never fabricating a sandboxed run."""
    executor = LinuxNamespaceSandboxExecutor()
    with pytest.raises(SandboxBackendUnavailableError, match="documented stub"):
        executor.run_command(["echo", "hello"])


def test_run_callable_raises_documented_unavailable_error():
    """run_callable() always raises SandboxBackendUnavailableError mentioning it is a documented stub, never fabricating a sandboxed run."""
    executor = LinuxNamespaceSandboxExecutor()
    with pytest.raises(SandboxBackendUnavailableError, match="documented stub"):
        executor.run_callable(lambda: 1)


def test_stub_can_be_instantiated_without_raising():
    """Instantiating LinuxNamespaceSandboxExecutor itself never raises -- only calling run_command/run_callable does."""
    # Instantiation itself must not raise -- only actual use should.
    executor = LinuxNamespaceSandboxExecutor()
    assert isinstance(executor, LinuxNamespaceSandboxExecutor)


def test_error_message_names_the_intended_real_backend_family():
    """The SandboxBackendUnavailableError message mentions both "namespace" and "linux" so a caller understands what real backend is intended."""
    executor = LinuxNamespaceSandboxExecutor()
    with pytest.raises(SandboxBackendUnavailableError) as excinfo:
        executor.run_command(["true"])
    message = str(excinfo.value)
    assert "namespace" in message.lower()
    assert "linux" in message.lower()


def test_get_default_executor_returns_seatbelt_not_linux_stub_on_this_machine():
    """On this macOS host, get_default_executor() dispatches to SeatbeltSandboxExecutor, never to the Linux stub."""
    from lazycore.sandbox import SeatbeltSandboxExecutor, get_default_executor

    executor = get_default_executor()
    assert isinstance(executor, SeatbeltSandboxExecutor)
    assert not isinstance(executor, LinuxNamespaceSandboxExecutor)
