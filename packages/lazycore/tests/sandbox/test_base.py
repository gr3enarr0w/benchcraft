"""Tests for lazycore.sandbox.base (shared executor interface, §2.3)."""

from __future__ import annotations

import pytest

from lazycore.sandbox.base import (
    BaseSandboxExecutor,
    SandboxBackendUnavailableError,
    SandboxError,
    SandboxPolicy,
    SandboxPolicyViolationError,
    SandboxResult,
)


def test_sandbox_policy_defaults_are_maximally_restrictive():
    policy = SandboxPolicy()
    assert policy.allow_network is False
    assert policy.allowed_read_paths == ()
    assert policy.allowed_write_paths == ()
    assert policy.allowed_executables == ()
    assert policy.env == {}
    assert policy.inherit_env is False
    assert policy.timeout_seconds is None


def test_sandbox_policy_is_frozen():
    policy = SandboxPolicy()
    with pytest.raises(Exception):
        policy.allow_network = True  # type: ignore[misc]


def test_sandbox_policy_with_overrides_returns_new_instance():
    base = SandboxPolicy(allow_network=False)
    modified = base.with_overrides(allow_network=True)

    assert base.allow_network is False
    assert modified.allow_network is True
    assert modified is not base


def test_sandbox_policy_generic_enough_for_two_different_mode_configs():
    # Simulates a LazyRed-style "red-team target sandbox" policy and a
    # LazyAgent-style "benchmark task sandbox" policy both being built from
    # the exact same SandboxPolicy dataclass with different values -- per
    # §2.3's "mode-specific policy configs layered on top" framing. Neither
    # module-specific logic nor field is hardcoded into the dataclass.
    lazyred_style = SandboxPolicy(
        allow_network=False,
        allowed_read_paths=("/tmp/redteam-target",),
        allowed_write_paths=(),
        allowed_executables=("/usr/bin/python3",),
    )
    lazyagent_style = SandboxPolicy(
        allow_network=True,
        allowed_read_paths=("/tmp/benchmark-workspace",),
        allowed_write_paths=("/tmp/benchmark-workspace/output",),
        allowed_executables=(),
    )

    assert lazyred_style.allow_network is False
    assert lazyagent_style.allow_network is True
    assert type(lazyred_style) is type(lazyagent_style) is SandboxPolicy


def test_sandbox_result_succeeded_true_only_when_clean_exit():
    ok = SandboxResult(exit_code=0, stdout="hi", stderr="")
    assert ok.succeeded is True

    bad_exit = SandboxResult(exit_code=1, stdout="", stderr="boom")
    assert bad_exit.succeeded is False

    blocked_but_zero_exit = SandboxResult(
        exit_code=0, stdout="", stderr="", policy_blocked=True
    )
    assert blocked_but_zero_exit.succeeded is False


def test_sandbox_result_is_frozen():
    result = SandboxResult(exit_code=0, stdout="", stderr="")
    with pytest.raises(Exception):
        result.exit_code = 1  # type: ignore[misc]


def test_base_sandbox_executor_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseSandboxExecutor()  # type: ignore[abstract]


def test_base_sandbox_executor_subclass_must_implement_full_interface():
    class Incomplete(BaseSandboxExecutor):
        def is_available(self) -> bool:
            return True

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_minimal_concrete_executor_tracks_configured_policy():
    class NoOpExecutor(BaseSandboxExecutor):
        def is_available(self) -> bool:
            return True

        def run_command(self, command, *, policy=None):
            return SandboxResult(exit_code=0, stdout="", stderr="")

        def run_callable(self, func, *, policy=None):
            return SandboxResult(exit_code=0, stdout="", stderr="")

    default_policy = SandboxPolicy()
    executor = NoOpExecutor(default_policy)
    assert executor.policy is default_policy

    new_policy = SandboxPolicy(allow_network=True)
    executor.configure(new_policy)
    assert executor.policy is new_policy


def test_exception_hierarchy():
    assert issubclass(SandboxBackendUnavailableError, SandboxError)
    assert issubclass(SandboxPolicyViolationError, SandboxError)
    assert issubclass(SandboxError, RuntimeError)
