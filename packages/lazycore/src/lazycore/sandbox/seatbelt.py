"""macOS Seatbelt (`sandbox-exec`) backend (architecture doc §2.3.1).

This is the v1-default backend on macOS for the shared sandbox executor.
It generates a Seatbelt Scheme-based SBPL (Sandbox Profile Language)
profile from a :class:`~lazycore.sandbox.base.SandboxPolicy` and invokes
``/usr/bin/sandbox-exec -f <profile> -- <command>`` via :mod:`subprocess`.

**Read this before touching GPU/Metal anything.** Per §2.3.1's research
findings, Seatbelt is confirmed to be a real, production-used isolation
primitive for constraining untrusted *tool-calling* processes on macOS
(used by Gemini CLI and other "agent-in-a-sandbox" projects) -- but it is
also confirmed that **Seatbelt cannot block or gate Metal/GPU/Cocoa access
even if you wanted it to**: "GPU and display passthrough flags have no
effect on macOS because Metal and Cocoa are system-level and cannot be
blocked via SBPL." Seatbelt is a process-level filesystem/network/syscall
allowlist, not a VM/kernel boundary, and it has no mechanism to mediate the
GPU driver stack at all. This module therefore does not attempt to add any
GPU-blocking rule to the generated profile -- there is no such rule to add.
Per the locked split-trust architecture, GPU-bound model inference is
expected to run *outside* this sandbox entirely; this executor only ever
constrains the CPU-bound tool-calling/code-execution layer (shell commands,
file I/O, network egress).
"""

from __future__ import annotations

import pickle
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Callable, Sequence

from lazycore.sandbox.base import (
    BaseSandboxExecutor,
    SandboxBackendUnavailableError,
    SandboxPolicy,
    SandboxResult,
)

__all__ = ["SeatbeltSandboxExecutor", "build_sbpl_profile"]

_SANDBOX_EXEC_PATH = "/usr/bin/sandbox-exec"

#: Substrings observed in stderr/output when Seatbelt denies an operation.
#: Used only as a best-effort heuristic for :attr:`SandboxResult.policy_blocked`
#: -- most Unix tools surface a Seatbelt EPERM/EACCES denial as one of these
#: standard POSIX ``strerror`` messages, but callers should not treat this
#: as a hard guarantee (see :class:`~lazycore.sandbox.base.SandboxResult`
#: docstring and :meth:`SeatbeltSandboxExecutor._classify_denial` below for
#: how false positives are reduced).
#:
#: This previously also included ``"Sandbox: "`` and ``"deny("``, which are
#: the format of macOS's Seatbelt *unified log* denial line (e.g.
#: ``Sandbox: cat(1234) deny(1) file-read-data ...``). That line is written
#: to the system unified log via ``os_log``/ASL, not to the sandboxed
#: process's own stdout/stderr -- ``subprocess.run(..., capture_output=True)``
#: never sees it, so these two markers could never actually match anything
#: this backend captures and were dead code. Capturing that log line for
#: real would require running a concurrent ``log stream``/``log show``
#: collector alongside every sandboxed invocation and correlating its
#: output back to this specific subprocess by PID and timestamp -- doable
#: in principle, but it needs extra process-management complexity, a
#: predicate stable across macOS versions, and (on some macOS versions)
#: elevated log-reading privileges, for a benefit that's marginal given the
#: precision improvements below. That's a reasonable future enhancement,
#: not something this fix takes on; the markers were simply removed rather
#: than left in as non-functional decoration.
_DENIAL_MARKERS = (
    "Operation not permitted",
    "Permission denied",
)

#: Conservative allowlist of command basenames that are read-only in every
#: normal invocation (no flag combination makes them write to the
#: filesystem). Used only to *rule out* a policy-attributed denial, never
#: to assert one -- see :meth:`SeatbeltSandboxExecutor._classify_denial`.
#: Deliberately small: an incomplete allowlist just means some false
#: positives aren't caught (the pre-existing, documented behavior), not
#: that anything is misclassified in the other direction.
_READ_ONLY_COMMAND_BASENAMES = frozenset(
    {
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "wc",
        "file",
        "stat",
        "md5",
        "md5sum",
        "shasum",
        "sha1sum",
        "sha256sum",
        "sha512sum",
        "readlink",
        "realpath",
        "od",
        "xxd",
        "hexdump",
        "strings",
    }
)

#: Filesystem locations that a resolved ``allowed_read_paths``/
#: ``allowed_write_paths`` entry must never land on exactly -- see
#: :func:`_reject_overbroad_allowed_path` (Finding 2). This is an
#: exact-match set, not a prefix/depth check: it exists to catch the
#: concrete "resolves to a filesystem root or another suspiciously broad
#: system directory" case without also rejecting legitimate, much more
#: common deep paths that happen to live a couple of components below root
#: (e.g. ``/opt/homebrew`` or ``/private/tmp``).
_FORBIDDEN_BROAD_PATHS = frozenset(
    str(Path(p))
    for p in (
        "/",
        "/Users",
        "/home",
        "/System",
        "/etc",
        "/Library",
        "/private",
        "/private/etc",
        "/private/var",
        "/var",
        "/usr",
        "/bin",
        "/sbin",
        "/opt",
        "/root",
        "/Applications",
    )
)

try:
    _HOME_DIR: Path | None = Path.home()
except Exception:  # pragma: no cover -- exotic envs with no resolvable home
    _HOME_DIR = None


def _escape_sbpl_string(value: str) -> str:
    """Escape a path for embedding in an SBPL string literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _canonical(path: str) -> str:
    """Resolve a path to its canonical, symlink-free form.

    macOS resolves several common temp-directory prefixes through symlinks
    (e.g. ``/tmp`` -> ``/private/tmp``), and Seatbelt matches against the
    canonical filesystem path, not the symlinked alias. Failing to resolve
    this is a classic source of "my allowed path doesn't work" bugs in
    hand-written Seatbelt profiles.

    This function performs *no* safety validation of its own -- it is used
    directly for ``allowed_executables`` (where bare, non-path names such
    as ``"python3"`` are a documented, legitimate value and "resolving" one
    is harmless/inert). For ``allowed_read_paths``/``allowed_write_paths``,
    :func:`_reject_overbroad_allowed_path` wraps this with the Finding-2
    anti-symlink-widening check below; use that instead for those two
    fields.
    """
    return str(Path(path).resolve())


def _reject_overbroad_allowed_path(path: str, *, kind: str) -> str:
    """Resolve an ``allowed_read_paths``/``allowed_write_paths`` entry, and
    refuse to build a profile if doing so would silently grant far broader
    access than the path string suggests.

    ``Path.resolve()`` correctly (and desirably) follows well-known, benign
    OS-provided symlinks such as ``/tmp`` -> ``/private/tmp`` -- that case
    must keep working exactly as before. The problem this guards against is
    different: if a *caller-supplied* allowed-path entry is itself a
    symlink (or resolves through one) to something the path string gives no
    hint of -- e.g. an entry that looks like a scoped project directory but
    is actually a symlink to ``/`` -- ``resolve()`` would silently
    substitute that broader target into the generated SBPL profile, and the
    resulting policy would grant access far wider than anyone reviewing the
    ``SandboxPolicy`` construction call could tell from the path string
    alone.

    This is caught here by refusing to build the profile at all if the
    *resolved* path lands exactly on a filesystem root or another
    well-known, suspiciously broad system directory (``/``, ``/Users``,
    ``/home``, ``/System``, ``/etc``, and similar -- see
    ``_FORBIDDEN_BROAD_PATHS``), or exactly on the user's home directory.

    **Honest limits.** This is a targeted check, not a general anti-symlink
    or anti-traversal mechanism:

    - It only catches resolution landing on one of a fixed set of
      well-known broad locations -- a symlink pointing at some other
      specific-but-still-wrong directory (e.g. a different user's home
      directory, or an unrelated project's data dir) will not be flagged.
      Getting this fully general would require a subjective notion of
      "how much broader is too much broader" relative to the *original*
      path string, which this function deliberately does not attempt.
    - It is a construction-time/profile-build-time check on the
      *configured* allowed-path entries. It cannot and does not attempt to
      prevent a sandboxed process from creating its own symlink at runtime
      that points from an allowed directory to somewhere outside it -- that
      is an OS-level property of how Seatbelt evaluates ``subpath`` rules
      against live symlinks, a separate, already-noted test gap, and out of
      scope for this fix.
    """
    resolved = Path(path).resolve()
    resolved_str = str(resolved)
    is_root = resolved_str == resolved.anchor
    is_forbidden_broad = resolved_str in _FORBIDDEN_BROAD_PATHS
    is_home = _HOME_DIR is not None and resolved == _HOME_DIR
    if is_root or is_forbidden_broad or is_home:
        raise ValueError(
            f"Refusing to build sandbox profile: {kind} entry {path!r} "
            f"resolves to {resolved_str!r}, which is a filesystem root or "
            "another suspiciously broad location (e.g. '/', '/Users', "
            "'/etc', '/System', or the user's home directory itself). If "
            f"{path!r} is a symlink, Path.resolve() has silently "
            "substituted its target here -- that would grant access far "
            "broader than the path string suggests. Pass a more specific, "
            "non-symlinked subdirectory instead."
        )
    return resolved_str


def build_sbpl_profile(policy: SandboxPolicy) -> str:
    """Generate an SBPL (Sandbox Profile Language) profile string for ``policy``.

    Profile structure:

    - ``(version 1)`` / ``(deny default)`` -- default-deny baseline.
    - A small fixed set of ``allow`` rules needed for *any* process to
      start and exit cleanly (fork, signal-to-self, sysctl-read,
      file-read-metadata) -- these are not policy-configurable because
      they are required for basic process lifecycle, not for accessing
      user data.
    - **Reads:** if ``policy.allowed_read_paths`` is empty (the default),
      reads are left broadly allowed (``(allow file-read*)``) -- this
      matches how most real-world agent-sandbox profiles operate in
      practice (the write and network surfaces are the actual enforcement
      points; unrestricted read of a single-user dev machine's filesystem
      by a locally-invoked tool is the accepted baseline). If
      ``allowed_read_paths`` is non-empty, reads are restricted to exactly
      those subpaths -- this is a stricter opt-in mode, and the caller is
      responsible for including any system paths a given command actually
      needs to read (dynamic linker, shared libraries, etc.); omitting
      them will cause otherwise-unrelated commands to fail to even start.
    - **Writes:** always restricted to ``policy.allowed_write_paths``
      (subpath-based). An empty tuple (the default) means no writes are
      allowed anywhere -- this is the primary enforcement demonstrated by
      this backend's tests.
    - **Network:** ``(allow network*)`` only if ``policy.allow_network`` is
      True; otherwise omitted, so the default-deny baseline blocks all
      network egress.
    - **Executables:** if ``policy.allowed_executables`` is non-empty,
      ``process-exec`` is restricted to those literal paths; otherwise
      ``process-exec`` is left unrestricted (needed for ordinary shell
      commands to exec `/bin/sh`, coreutils, `python3`, etc.).

    **Anti-widening check on read/write paths.** Each entry in
    ``allowed_read_paths``/``allowed_write_paths`` is resolved via
    :func:`_reject_overbroad_allowed_path`, not plain ``Path.resolve()``:
    this raises ``ValueError`` (refusing to build the profile at all)
    rather than silently emitting a ``subpath`` rule if an entry resolves
    to a filesystem root or another suspiciously broad system directory
    (see that function's docstring for exactly what is and is not caught --
    it is a targeted check, not a general anti-symlink/anti-traversal
    mechanism).

    Never adds, and will never add, any rule attempting to gate GPU/Metal/
    Cocoa access -- see the module docstring.

    Raises:
        ValueError: If an ``allowed_read_paths``/``allowed_write_paths``
            entry resolves to a suspiciously broad location -- see
            :func:`_reject_overbroad_allowed_path`.
    """
    lines = [
        "(version 1)",
        "(deny default)",
        "",
        "; --- baseline process lifecycle (not user-configurable) ---",
        "(allow process-fork)",
        "(allow signal (target self))",
        "(allow sysctl-read)",
        "(allow file-read-metadata)",
        "(allow mach-lookup)",
        "(allow iokit-open)",
        "(allow file-ioctl)",
    ]

    lines.append("")
    lines.append("; --- filesystem reads ---")
    if policy.allowed_read_paths:
        read_clauses = " ".join(
            f'(subpath "{_escape_sbpl_string(_reject_overbroad_allowed_path(p, kind="allowed_read_paths"))}")'
            for p in policy.allowed_read_paths
        )
        lines.append(f"(allow file-read* {read_clauses})")
    else:
        lines.append("(allow file-read*)")

    lines.append("")
    lines.append("; --- filesystem writes ---")
    if policy.allowed_write_paths:
        write_clauses = " ".join(
            f'(subpath "{_escape_sbpl_string(_reject_overbroad_allowed_path(p, kind="allowed_write_paths"))}")'
            for p in policy.allowed_write_paths
        )
        lines.append(f"(allow file-write* {write_clauses})")
    else:
        lines.append("; no allowed_write_paths configured -- all writes denied")

    lines.append("")
    lines.append("; --- network egress ---")
    if policy.allow_network:
        lines.append("(allow network*)")
    else:
        lines.append("; allow_network=False -- all network egress denied")

    lines.append("")
    lines.append("; --- process execution ---")
    if policy.allowed_executables:
        exec_clauses = " ".join(
            f'(literal "{_escape_sbpl_string(_canonical(p))}")'
            for p in policy.allowed_executables
        )
        lines.append(f"(allow process-exec {exec_clauses})")
    else:
        lines.append("(allow process-exec)")

    return "\n".join(lines) + "\n"


class SeatbeltSandboxExecutor(BaseSandboxExecutor):
    """Sandbox executor backed by macOS's `sandbox-exec`/Seatbelt.

    Restricts filesystem reads/writes, network egress, and (optionally)
    which executables may be exec'd by the tool-calling/code-execution
    layer. Never attempts, and cannot in principle (per §2.3.1), gate GPU/
    Metal/Cocoa access -- inference processes must be run outside this
    sandbox entirely, per the platform's split-trust architecture.

    **Stricter contract than the original implementation:**

    - ``SandboxPolicy.allowed_read_paths``/``allowed_write_paths``/
      ``working_directory`` must be absolute paths, enforced eagerly by
      :class:`~lazycore.sandbox.base.SandboxPolicy`'s own
      ``__post_init__`` -- a relative entry raises ``ValueError`` at
      policy-construction time, not when ``run_command()`` happens to be
      called from a different working directory later.
    - ``allowed_read_paths``/``allowed_write_paths`` entries are rejected
      with ``ValueError`` (profile build refused entirely) if they resolve
      to a filesystem root or another suspiciously broad system directory
      -- see :func:`_reject_overbroad_allowed_path`. This is a targeted
      anti-widening check, not a general anti-symlink mechanism; benign,
      well-known OS symlinks (e.g. ``/tmp`` -> ``/private/tmp``) are
      unaffected.
    - ``SandboxResult.policy_blocked`` is still a best-effort heuristic,
      never a guarantee, but is more precise than plain substring matching
      against stdout/stderr: it cross-references the specific command and
      the policy actually in effect for that call to rule out the common
      false-positive case of an ordinary POSIX/DAC permission error (e.g.
      a chmod 000 file) that has nothing to do with Seatbelt -- see
      :meth:`_classify_denial`. It still cannot see Seatbelt's own denial
      logging (written to the unified log, not to the sandboxed process's
      captured stdout/stderr), and it still cannot classify commands
      outside its small internal allowlist with the same precision.
    """

    def is_available(self) -> bool:
        return platform.system() == "Darwin" and Path(_SANDBOX_EXEC_PATH).exists()

    def _require_available(self) -> None:
        if not self.is_available():
            raise SandboxBackendUnavailableError(
                "SeatbeltSandboxExecutor requires macOS with "
                f"{_SANDBOX_EXEC_PATH} present; this host does not "
                "satisfy that (platform="
                f"{platform.system()!r})."
            )

    def _resolve_policy(self, policy: SandboxPolicy | None) -> SandboxPolicy:
        return policy if policy is not None else self._policy

    def _build_env(self, policy: SandboxPolicy) -> dict[str, str]:
        import os

        env: dict[str, str] = dict(os.environ) if policy.inherit_env else {}
        env.update(policy.env)
        return env

    def _classify_denial(
        self,
        policy: SandboxPolicy,
        command: Sequence[str],
        exit_code: int,
        stdout: str,
        stderr: str,
    ) -> bool:
        """Best-effort classification of whether Seatbelt (as opposed to an
        ordinary POSIX/DAC permission error) caused this failure.

        This is necessarily still a heuristic, not a guarantee -- see the
        :class:`~lazycore.sandbox.base.SandboxResult.policy_blocked`
        docstring. It is more precise than plain substring matching in one
        specific, provable case: :func:`build_sbpl_profile` emits an
        unconditional ``(allow file-read*)`` whenever
        ``policy.allowed_read_paths`` is empty, meaning Seatbelt places
        *zero* restriction on reads for that policy. If the failing command
        is a member of ``_READ_ONLY_COMMAND_BASENAMES`` (a conservative
        allowlist of commands that never write to the filesystem under any
        normal flag combination) and reads are unrestricted for the active
        policy, then Seatbelt structurally cannot have produced this
        denial -- the profile it generated grants unconditional read
        access -- so a "Permission denied"/"Operation not permitted" in
        this specific combination must be an ordinary DAC error (e.g. a
        chmod 000 file) unrelated to the sandbox.

        This cross-reference only rules a denial *out*; it never rules one
        *in* beyond the original marker match. It does not attempt to
        determine read/write/exec/network direction for arbitrary commands
        outside the allowlist (most real-world tool invocations), so it
        narrows -- but does not eliminate -- the false-positive surface
        described in the confirmed finding this fixes.
        """
        if exit_code == 0:
            return False
        combined = stdout + stderr
        if not any(marker in combined for marker in _DENIAL_MARKERS):
            return False

        if command:
            basename = Path(command[0]).name
            if basename in _READ_ONLY_COMMAND_BASENAMES and not policy.allowed_read_paths:
                return False

        return True

    def run_command(
        self,
        command: Sequence[str],
        *,
        policy: SandboxPolicy | None = None,
    ) -> SandboxResult:
        self._require_available()
        active_policy = self._resolve_policy(policy)
        profile = build_sbpl_profile(active_policy)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sb", delete=False
        ) as profile_file:
            profile_file.write(profile)
            profile_path = profile_file.name

        try:
            argv = [_SANDBOX_EXEC_PATH, "-f", profile_path, "--", *command]
            try:
                completed = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=active_policy.timeout_seconds,
                    env=self._build_env(active_policy),
                    cwd=active_policy.working_directory,
                )
            except subprocess.TimeoutExpired as exc:
                return SandboxResult(
                    exit_code=124,
                    stdout=exc.stdout or "" if isinstance(exc.stdout, str) else "",
                    stderr=(exc.stderr or "" if isinstance(exc.stderr, str) else "")
                    + "\n[lazycore.sandbox] command timed out",
                    policy_blocked=False,
                )

            policy_blocked = self._classify_denial(
                active_policy, command, completed.returncode, completed.stdout, completed.stderr
            )
            return SandboxResult(
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                policy_blocked=policy_blocked,
            )
        finally:
            Path(profile_path).unlink(missing_ok=True)

    def run_callable(
        self,
        func: Callable[[], object],
        *,
        policy: SandboxPolicy | None = None,
    ) -> SandboxResult:
        self._require_available()
        active_policy = self._resolve_policy(policy)

        try:
            payload = pickle.dumps(func)
        except (pickle.PicklingError, AttributeError, TypeError) as exc:
            raise ValueError(
                "run_callable() requires a picklable callable (e.g. a "
                "module-level function), because the Seatbelt backend "
                "must marshal it into a subprocess to actually sandbox "
                "it. A local closure or lambda cannot be pickled."
            ) from exc

        with tempfile.TemporaryDirectory(prefix="lazycore-sandbox-") as tmp_dir:
            pkl_path = Path(tmp_dir) / "callable.pkl"
            pkl_path.write_bytes(payload)

            runner_path = Path(tmp_dir) / "runner.py"
            runner_path.write_text(
                textwrap.dedent(
                    """\
                    import pickle
                    import sys

                    with open(sys.argv[1], "rb") as f:
                        func = pickle.load(f)
                    try:
                        result = func()
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        sys.exit(1)
                    else:
                        sys.stdout.write("" if result is None else repr(result))
                        sys.exit(0)
                    """
                )
            )

            # The harness's own temp files (script + pickled payload) must
            # be readable regardless of the caller's policy -- this is
            # internal plumbing, not user data, and mirrors how
            # run_command's own generated .sb profile file needs no
            # explicit allow-read rule (profile files are read by
            # sandbox-exec itself, before the sandbox is even active).
            effective_policy = active_policy
            if active_policy.allowed_read_paths:
                effective_policy = active_policy.with_overrides(
                    allowed_read_paths=(
                        *active_policy.allowed_read_paths,
                        tmp_dir,
                    )
                )

            return self.run_command(
                [sys.executable, str(runner_path), str(pkl_path)],
                policy=effective_policy,
            )
