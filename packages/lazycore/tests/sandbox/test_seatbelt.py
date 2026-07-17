"""Real tests for lazycore.sandbox.seatbelt.SeatbeltSandboxExecutor.

These tests actually invoke `/usr/bin/sandbox-exec` on the machine running
the test suite -- they are skipped (not mocked) on non-macOS hosts, since
the whole point is demonstrating real Seatbelt enforcement, not plumbing.
"""

from __future__ import annotations

import platform
import tempfile
from pathlib import Path

import pytest

from lazycore.sandbox.base import SandboxPolicy
from lazycore.sandbox.seatbelt import SeatbeltSandboxExecutor, build_sbpl_profile

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").exists(),
    reason="SeatbeltSandboxExecutor only runs on macOS with /usr/bin/sandbox-exec present",
)

_FIXTURES_DIR = str(Path(__file__).parent)


def test_is_available_on_this_macos_machine():
    """is_available() returns True on this macOS host where /usr/bin/sandbox-exec is present."""
    executor = SeatbeltSandboxExecutor()
    assert executor.is_available() is True


def test_build_sbpl_profile_contains_deny_default_baseline():
    """The generated SBPL profile for a default policy has the deny-default baseline and no network-allow rule."""
    profile = build_sbpl_profile(SandboxPolicy())
    assert "(version 1)" in profile
    assert "(deny default)" in profile
    # No allow_network -> no network* allow rule anywhere in the profile.
    assert "(allow network*)" not in profile


def test_build_sbpl_profile_never_mentions_gpu_metal_or_cocoa():
    """The generated SBPL profile never references GPU/Metal/Cocoa/MPS terms, per §2.3.1's rule that Seatbelt cannot and must not attempt to gate GPU access."""
    # Per §2.3.1: Seatbelt cannot gate GPU/Metal/Cocoa access, and this
    # backend must never pretend otherwise by emitting rules that reference
    # them.
    profile = build_sbpl_profile(
        SandboxPolicy(allow_network=True, allowed_write_paths=("/tmp",))
    )
    lowered = profile.lower()
    for forbidden_term in ("metal", "cocoa", "gpu", "mps"):
        assert forbidden_term not in lowered


def test_command_touching_allowed_write_path_succeeds():
    """A command writing inside an allowed_write_paths directory succeeds, with a clean exit and no policy_blocked flag."""
    executor = SeatbeltSandboxExecutor()
    with tempfile.TemporaryDirectory() as allowed_dir:
        target_file = str(Path(allowed_dir) / "written-by-sandbox.txt")
        policy = SandboxPolicy(allowed_write_paths=(allowed_dir,))

        result = executor.run_command(
            ["/usr/bin/touch", target_file], policy=policy
        )

        assert result.exit_code == 0, result.stderr
        assert result.policy_blocked is False
        assert Path(target_file).exists()


def test_command_writing_outside_allowed_path_is_blocked():
    """A command attempting to write to a directory outside allowed_write_paths is genuinely denied by Seatbelt: nonzero exit, file never created, policy_blocked=True."""
    executor = SeatbeltSandboxExecutor()
    with tempfile.TemporaryDirectory() as allowed_dir, tempfile.TemporaryDirectory() as other_dir:
        # allowed_dir is granted write access; other_dir is a completely
        # separate real temp directory that is NOT in allowed_write_paths.
        forbidden_target = str(Path(other_dir) / "should-not-be-created.txt")
        policy = SandboxPolicy(allowed_write_paths=(allowed_dir,))

        result = executor.run_command(
            ["/usr/bin/touch", forbidden_target], policy=policy
        )

        # This is the core enforcement assertion: sandbox-exec must have
        # actually denied the write. `touch` on a Seatbelt-denied path
        # exits non-zero and reports "Operation not permitted".
        assert result.exit_code != 0
        assert not Path(forbidden_target).exists()
        assert result.policy_blocked is True
        assert "Operation not permitted" in result.stderr


def test_default_policy_denies_all_writes_anywhere():
    """With the default (empty) allowed_write_paths, a write is denied even to a directory that would otherwise seem harmless."""
    executor = SeatbeltSandboxExecutor()
    with tempfile.TemporaryDirectory() as some_dir:
        target_file = str(Path(some_dir) / "no-write-paths-configured.txt")
        # Default SandboxPolicy() has allowed_write_paths=() -- nothing is
        # writable, not even a directory that would otherwise seem "safe".
        result = executor.run_command(["/usr/bin/touch", target_file])

        assert result.exit_code != 0
        assert not Path(target_file).exists()


def test_read_only_command_succeeds_by_default():
    """Reading a system file (/etc/hosts) succeeds under the default policy, since reads are left broadly allowed unless allowed_read_paths is set."""
    executor = SeatbeltSandboxExecutor()
    # Default policy leaves reads broadly allowed (documented in
    # build_sbpl_profile's docstring) -- only writes/network are
    # restricted by default.
    result = executor.run_command(["/bin/cat", "/etc/hosts"])
    assert result.exit_code == 0
    assert result.policy_blocked is False
    assert len(result.stdout) > 0


def test_network_egress_denied_by_default():
    """A curl request to an external host fails when allow_network=False, since the profile omits the network-allow rule."""
    executor = SeatbeltSandboxExecutor()
    policy = SandboxPolicy(allow_network=False, timeout_seconds=10)
    result = executor.run_command(
        ["/usr/bin/curl", "-s", "--max-time", "5", "https://example.com"],
        policy=policy,
    )
    assert result.exit_code != 0


def test_run_callable_executes_picklable_function_and_returns_output():
    """run_callable() marshals a module-level picklable function into a sandboxed subprocess and returns its repr()'d output via stdout."""
    # tests/sandbox has no __init__.py (matching this package's existing
    # flat tests/ convention), so pytest's rootdir-based import mode has
    # already put this directory on sys.path to import test_seatbelt.py
    # itself as a top-level module -- `import _callable_fixtures` here
    # resolves the same way, and its __module__ (used by pickle) matches
    # what the sandboxed subprocess will resolve via PYTHONPATH below.
    import _callable_fixtures  # type: ignore[import-not-found]

    executor = SeatbeltSandboxExecutor()
    policy = SandboxPolicy(inherit_env=True, env={"PYTHONPATH": _FIXTURES_DIR})

    result = executor.run_callable(_callable_fixtures.compute_answer, policy=policy)

    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "42"


def test_run_callable_surfaces_exception_as_nonzero_exit():
    """A callable that raises inside the sandboxed subprocess exits nonzero and the traceback (including the exception type/message) appears in stderr."""
    import _callable_fixtures  # type: ignore[import-not-found]

    executor = SeatbeltSandboxExecutor()
    policy = SandboxPolicy(inherit_env=True, env={"PYTHONPATH": _FIXTURES_DIR})

    result = executor.run_callable(_callable_fixtures.raise_value_error, policy=policy)

    assert result.exit_code == 1
    assert "ValueError" in result.stderr
    assert "boom from sandboxed callable" in result.stderr


def test_run_callable_rejects_unpicklable_lambda():
    """run_callable() raises ValueError immediately for a lambda, since lambdas cannot be pickled for the subprocess handoff."""
    executor = SeatbeltSandboxExecutor()
    with pytest.raises(ValueError):
        executor.run_callable(lambda: 1)


def test_get_default_executor_returns_seatbelt_on_this_machine():
    """get_default_executor() returns an available SeatbeltSandboxExecutor on this macOS host."""
    from lazycore.sandbox import get_default_executor

    executor = get_default_executor()
    assert isinstance(executor, SeatbeltSandboxExecutor)
    assert executor.is_available() is True
