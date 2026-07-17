"""Linux namespace-based sandbox backend -- DOCUMENTED STUB, not a real
implementation (architecture doc §2.3).

Per §2.3, Linux is Benchcraft's secondary v1 platform, and the "existing
options from prior research apply unchanged" there: gVisor/runsc,
Firecracker microVMs, Docker/runc, or lightweight namespace-based
approaches (the "SWE-MiniSandbox" pattern using `unshare`/`bwrap`-style
primitives) are all viable, genuinely Linux-native isolation mechanisms,
because -- unlike macOS -- Linux has no VM-boundary GPU passthrough problem
to design around in the first place (§2.3.1).

**This class is intentionally not a real backend.** The reference
development machine for this repository is macOS (Apple Silicon), per
`CLAUDE.md` and the architecture doc's stated reference hardware. A real
namespace/gVisor/Firecracker-based executor cannot be meaningfully built
*or verified* from this machine -- there is no Linux kernel here to test
namespace isolation against, and shipping untested isolation code as if it
were real would be worse than not shipping it. This class exists only to:

1. Satisfy :class:`~lazycore.sandbox.base.BaseSandboxExecutor`'s interface,
   so callers can reference it and `get_default_executor()` can dispatch to
   it on Linux without an import error.
2. Detect, at runtime, whether the host *could* plausibly support a real
   namespace-based backend (presence of `unshare`/`bwrap` or similar), via
   :meth:`is_available`.
3. Raise a clear, documented error on any actual use, rather than silently
   pretending to sandbox anything.

Building the real Linux backend is future work for whoever has an actual
Linux dev/CI environment to validate it against -- do not fill in this
stub with unverified logic just to make it "look done."
"""

from __future__ import annotations

import platform
import shutil
from typing import Callable, Sequence

from lazycore.sandbox.base import (
    BaseSandboxExecutor,
    SandboxBackendUnavailableError,
    SandboxPolicy,
    SandboxResult,
)

__all__ = ["LinuxNamespaceSandboxExecutor"]

#: Helper binaries a real namespace-based backend would depend on
#: (bubblewrap's `bwrap`, or raw `unshare` from util-linux). Their presence
#: is used only to answer "could this host plausibly run a real backend
#: some day", not to actually invoke them -- this stub never execs
#: anything.
_LINUX_SANDBOX_HELPERS = ("bwrap", "unshare")


class LinuxNamespaceSandboxExecutor(BaseSandboxExecutor):
    """Stub for the Linux-native namespace/gVisor/Firecracker backend.

    Every method that would need to actually sandbox something raises
    :class:`~lazycore.sandbox.base.SandboxBackendUnavailableError` -- this
    class never fabricates sandboxing behavior. See the module docstring
    for why this is a deliberate stub rather than a real implementation.
    """

    def is_available(self) -> bool:
        """True only if this looks like a Linux host with a plausible
        namespace-sandboxing helper installed. This does **not** mean the
        backend is implemented -- it only means the host isn't immediately
        disqualified. Always False on macOS (this development machine).
        """
        if platform.system() != "Linux":
            return False
        return any(shutil.which(tool) is not None for tool in _LINUX_SANDBOX_HELPERS)

    def _unavailable(self) -> SandboxBackendUnavailableError:
        return SandboxBackendUnavailableError(
            "LinuxNamespaceSandboxExecutor is a documented stub, not a "
            "real sandbox backend. Per architecture doc §2.3, the intended "
            "real Linux backend is a namespace-based isolation mechanism "
            "(gVisor/Firecracker/unshare+bwrap-style, per prior sandbox "
            "research) -- it was deliberately not implemented here because "
            "this repository's reference/dev environment is macOS "
            f"(detected platform: {platform.system()!r}), and namespace "
            "isolation cannot be meaningfully built or verified without a "
            "real Linux kernel to test against. Implement and validate "
            "this backend from an actual Linux environment before relying "
            "on it."
        )

    def run_command(
        self,
        command: Sequence[str],
        *,
        policy: SandboxPolicy | None = None,
    ) -> SandboxResult:
        """Always raise :class:`SandboxBackendUnavailableError`.

        This stub never executes ``command`` under any isolation -- see the
        module docstring for why a real namespace-based implementation
        isn't provided here.
        """
        raise self._unavailable()

    def run_callable(
        self,
        func: Callable[..., object],
        *,
        policy: SandboxPolicy | None = None,
    ) -> SandboxResult:
        """Always raise :class:`SandboxBackendUnavailableError`.

        This stub never executes ``func`` under any isolation -- see the
        module docstring for why a real namespace-based implementation
        isn't provided here.
        """
        raise self._unavailable()
