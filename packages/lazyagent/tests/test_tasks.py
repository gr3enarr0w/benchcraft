"""Tests for benchcraft_lazyagent.tasks -- the file-manipulation task family.

The key acceptance criterion these tests demonstrate: the fail-designed
(sandbox-escape-attempt) task is genuinely blocked by the real Seatbelt
sandbox -- not merely scored as a failure by convention. The forbidden
file must not exist on disk after the run.
"""

from __future__ import annotations

import platform
from pathlib import Path

import pytest

from lazycore.sandbox import get_default_executor
from benchcraft_lazyagent.adapter import SandboxedAgentAdapter
from benchcraft_lazyagent.tasks import (
    default_task_suite,
    make_fail_task,
    make_pass_task,
    rule_based_agent,
    score_file_task,
)

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="SeatbeltSandboxExecutor is macOS-only; this suite exercises the real backend, not a mock",
)


@pytest.fixture()
def executor():
    ex = get_default_executor()
    assert ex.is_available()
    return ex


def test_pass_task_creates_expected_file_and_scores_success(executor, tmp_path: Path):
    task = make_pass_task(tmp_path)
    adapter = SandboxedAgentAdapter(rule_based_agent)

    trajectory = adapter.run_task(task, executor)
    success, detail = score_file_task(task, trajectory)

    assert success is True, detail
    target = Path(task.target_path)
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == task.expected_content


def test_fail_task_is_genuinely_blocked_by_the_sandbox_not_just_scored_false(
    executor, tmp_path: Path
):
    """The escape-attempt task's target file must not exist at all -- proving
    the shared lazycore.sandbox executor's write-path allowlist actually
    prevented the write, rather than the agent simply not trying."""
    task = make_fail_task(tmp_path)
    adapter = SandboxedAgentAdapter(rule_based_agent)

    trajectory = adapter.run_task(task, executor)
    success, detail = score_file_task(task, trajectory)

    # The sandbox should report the command as blocked/failed.
    assert trajectory.sandbox_result.exit_code != 0

    # The scored outcome must be failure...
    assert success is False, detail

    # ...and the containment must be real: no file was created anywhere,
    # not even the forbidden target path.
    forbidden_target = Path(task.target_path)
    assert not forbidden_target.exists()
    assert not forbidden_target.parent.exists(), (
        "the sandbox should have blocked mkdir -p on the forbidden parent "
        "directory as well, since it is outside allowed_write_paths"
    )


def test_default_task_suite_has_one_pass_and_one_fail_designed_task(tmp_path: Path):
    suite = default_task_suite(tmp_path)

    assert len(suite) == 2
    assert sum(1 for t in suite if t.expect_success) == 1
    assert sum(1 for t in suite if not t.expect_success) == 1


def test_score_file_task_rejects_non_file_task_spec():
    from benchcraft_lazyagent.adapter import AgentTrajectory, TaskSpec
    from lazycore.sandbox import SandboxPolicy, SandboxResult

    bad_task = TaskSpec(name="x", description="x", sandbox_policy=SandboxPolicy())
    trajectory = AgentTrajectory(
        task_name="x",
        steps=(),
        sandbox_result=SandboxResult(exit_code=0, stdout="", stderr=""),
    )

    with pytest.raises(TypeError):
        score_file_task(bad_task, trajectory)
