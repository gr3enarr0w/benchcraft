"""Tests for benchcraft_lazyagent.adapter -- the AgentAdapter/SandboxedAgentAdapter pattern.

These tests exercise the real macOS Seatbelt backend
(`lazycore.sandbox.get_default_executor()`) rather than mocking the
sandbox -- this machine is macOS, so `SeatbeltSandboxExecutor` is expected
to actually be available (see README "Sandbox wiring").
"""

from __future__ import annotations

import platform
import tempfile
from pathlib import Path

import pytest

from lazycore.sandbox import BaseSandboxExecutor, get_default_executor
from benchcraft_lazyagent.adapter import (
    AgentAction,
    AgentTrajectory,
    SandboxedAgentAdapter,
    TaskSpec,
)
from benchcraft_lazyagent.tasks import make_pass_task, rule_based_agent

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="SeatbeltSandboxExecutor is macOS-only; this suite exercises the real backend, not a mock",
)


@pytest.fixture()
def executor() -> BaseSandboxExecutor:
    ex = get_default_executor()
    assert ex.is_available(), "expected the real Seatbelt backend to be available on macOS"
    return ex


def test_sandboxed_agent_adapter_returns_trajectory_with_three_steps(executor, tmp_path: Path):
    task = make_pass_task(tmp_path)
    adapter = SandboxedAgentAdapter(rule_based_agent)

    trajectory = adapter.run_task(task, executor)

    assert isinstance(trajectory, AgentTrajectory)
    assert trajectory.task_name == task.name
    assert len(trajectory.steps) == 3
    assert trajectory.steps[0].role == "user"
    assert trajectory.steps[1].role == "assistant"
    assert trajectory.steps[2].role == "tool"


def test_sandboxed_agent_adapter_actually_executes_via_sandbox_result(executor, tmp_path: Path):
    task = make_pass_task(tmp_path)
    adapter = SandboxedAgentAdapter(rule_based_agent)

    trajectory = adapter.run_task(task, executor)

    # The sandbox result is real: exit_code/stdout/stderr came from an
    # actual subprocess.run() call inside SeatbeltSandboxExecutor, not a
    # stub -- a successful file-creation command in an allowed path exits 0.
    assert trajectory.sandbox_result.exit_code == 0
    assert trajectory.sandbox_result.policy_blocked is False
    assert Path(task.target_path).is_file()


def test_agent_fn_plug_in_point_accepts_any_matching_callable(executor, tmp_path: Path):
    """A caller-supplied callable matching AgentFn's signature works without
    subclassing anything -- this is the "bring your own agent" contract."""
    task = make_pass_task(tmp_path)

    def custom_agent(t: TaskSpec) -> AgentAction:
        # Deliberately different rationale/behavior from rule_based_agent,
        # to prove the adapter doesn't hardcode a specific agent.
        return AgentAction(command=["/bin/sh", "-c", "true"], rationale="no-op stand-in agent")

    adapter = SandboxedAgentAdapter(custom_agent)
    trajectory = adapter.run_task(task, executor)

    assert trajectory.sandbox_result.exit_code == 0
    assert "no-op stand-in agent" in trajectory.steps[1].content
