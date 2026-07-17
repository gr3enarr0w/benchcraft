"""OpenTelemetry GenAI semantic-convention helpers (architecture doc §2.6).

LazyCore adopts OpenTelemetry GenAI semantic conventions as the shared
trace/result schema across LazyRed's security-audit reports, LazyAgent's
execution trajectories, and the leaderboard-style results emitted by the
AutoML module, LazyForecast, LazyGraph, and LazyVision. Data is represented
as OTel spans plus a small set of custom attributes (e.g.
``security.severity``, ``owasp.mapping``, ``ml.metric.accuracy``) and span
events for conversational transcripts.

This module depends only on ``opentelemetry-api`` (never the SDK). Without
an SDK/exporter configured by the *application*, the tracer returned here is
a documented OTel no-op tracer -- spans are created and attributes/events
are accepted, but nothing is exported anywhere. That's intentional: it keeps
``lazycore`` import-safe and dependency-light for every module, while still
letting a module that *does* wire up the SDK (e.g. an OTLP exporter, or a
console exporter for local debugging) get consistently-shaped spans for
free.
"""

from __future__ import annotations

import enum
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer

__all__ = [
    "ATTR_SECURITY_SEVERITY",
    "ATTR_OWASP_MAPPING",
    "ATTR_ML_METRIC_PREFIX",
    "ATTR_ML_METRIC_ACCURACY",
    "ATTR_GENAI_EVENT_ROLE",
    "ATTR_GENAI_EVENT_CONTENT",
    "SecuritySeverity",
    "ml_metric_attribute",
    "get_tracer",
    "genai_span",
    "set_security_finding",
    "set_ml_metric",
    "add_transcript_event",
]

# Default tracer/instrumentation-scope name used when a caller doesn't
# supply their own module name.
_DEFAULT_TRACER_NAME = "benchcraft.lazycore"

# --- Shared attribute names (§2.6) -----------------------------------------

#: Severity of a security/red-team finding, one of the values in
#: :class:`SecuritySeverity`. Used by LazyRed's security-audit reports.
ATTR_SECURITY_SEVERITY = "security.severity"

#: OWASP LLM Top 10 / OWASP Agentic Top 10 / MITRE ATLAS mapping ID(s) for a
#: given finding. Used by LazyRed.
ATTR_OWASP_MAPPING = "owasp.mapping"

#: Namespace prefix for ML leaderboard metrics (accuracy, F1, PICP, MPIW,
#: etc.) reported by the AutoML module, LazyForecast, LazyGraph, and
#: LazyVision.
ATTR_ML_METRIC_PREFIX = "ml.metric."

#: The specific example attribute named in the architecture doc.
ATTR_ML_METRIC_ACCURACY = f"{ATTR_ML_METRIC_PREFIX}accuracy"

#: Span-event attribute keys for conversational transcripts (agent
#: trajectories, red-team conversations).
ATTR_GENAI_EVENT_ROLE = "gen_ai.event.role"
ATTR_GENAI_EVENT_CONTENT = "gen_ai.event.content"


class SecuritySeverity(str, enum.Enum):
    """Allowed values for :data:`ATTR_SECURITY_SEVERITY`.

    Modeled on common vulnerability-severity scales used by LazyRed's
    OWASP/MITRE ATLAS-mapped findings.
    """

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def ml_metric_attribute(metric_name: str) -> str:
    """Build a namespaced ``ml.metric.<metric_name>`` attribute key.

    Example: ``ml_metric_attribute("accuracy") == "ml.metric.accuracy"``.
    """
    if not metric_name:
        raise ValueError("metric_name must be a non-empty string")
    return f"{ATTR_ML_METRIC_PREFIX}{metric_name}"


def get_tracer(instrumenting_module_name: str | None = None) -> Tracer:
    """Return an OTel tracer scoped to ``instrumenting_module_name``.

    Thin wrapper over ``opentelemetry.trace.get_tracer`` so every module
    goes through one lazycore entrypoint rather than importing
    ``opentelemetry.trace`` directly. If the calling application has not
    configured a TracerProvider (via the SDK), this returns OTel's
    documented no-op tracer -- calls are safe, they simply don't export.
    """
    return trace.get_tracer(instrumenting_module_name or _DEFAULT_TRACER_NAME)


@contextmanager
def genai_span(
    name: str,
    *,
    tracer: Tracer | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Span]:
    """Start a span using the shared GenAI schema conventions.

    Usage::

        with genai_span("lazyred.probe.run", attributes={...}) as span:
            set_security_finding(span, severity=SecuritySeverity.HIGH,
                                  owasp_mapping=["LLM01"])
            ...

    ``tracer`` defaults to :func:`get_tracer` with the shared lazycore
    instrumentation-scope name; pass your own (e.g.
    ``get_tracer(__name__)``) if you want spans attributed to your module.
    """
    active_tracer = tracer or get_tracer()
    with active_tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span


def set_security_finding(
    span: Span,
    *,
    severity: SecuritySeverity | str,
    owasp_mapping: str | Sequence[str] | None = None,
) -> None:
    """Attach a security-audit finding's severity/OWASP mapping to ``span``.

    ``severity`` may be a :class:`SecuritySeverity` member or a raw string
    matching one of its values; anything else raises ``ValueError`` so
    inconsistent severity strings don't silently leak into telemetry.
    """
    if isinstance(severity, SecuritySeverity):
        severity_value = severity.value
    else:
        try:
            severity_value = SecuritySeverity(severity).value
        except ValueError as exc:
            valid = ", ".join(s.value for s in SecuritySeverity)
            raise ValueError(
                f"Invalid security severity {severity!r}; expected one of: {valid}"
            ) from exc

    span.set_attribute(ATTR_SECURITY_SEVERITY, severity_value)
    if owasp_mapping is not None:
        mapping_value = (
            [owasp_mapping] if isinstance(owasp_mapping, str) else list(owasp_mapping)
        )
        span.set_attribute(ATTR_OWASP_MAPPING, mapping_value)


def set_ml_metric(span: Span, metric_name: str, value: float) -> None:
    """Attach a leaderboard metric (e.g. ``accuracy``, ``picp``) to ``span``
    under the shared ``ml.metric.*`` namespace."""
    span.set_attribute(ml_metric_attribute(metric_name), value)


def add_transcript_event(
    span: Span,
    role: str,
    content: str,
    **extra_attributes: Any,
) -> None:
    """Record one turn of a conversational transcript as a span event.

    Used for agent trajectories (LazyAgent) and red-team conversations
    (LazyRed) so both modules represent multi-turn transcripts the same
    way: one span event per turn, tagged with the speaker ``role`` (e.g.
    ``"user"``, ``"assistant"``, ``"tool"``) and its ``content``.
    """
    attributes: dict[str, Any] = {
        ATTR_GENAI_EVENT_ROLE: role,
        ATTR_GENAI_EVENT_CONTENT: content,
    }
    attributes.update(extra_attributes)
    span.add_event(name="gen_ai.content", attributes=attributes)
