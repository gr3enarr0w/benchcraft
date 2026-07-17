"""Tests for lazycore.telemetry (OTel GenAI semantic-convention helpers, §2.6).

These tests deliberately do not depend on the OpenTelemetry SDK (lazycore
only depends on opentelemetry-api). Without an SDK TracerProvider
configured, spans returned by the API are non-recording no-op spans -- so
these tests assert on *behavior that doesn't raise* and on the shared
attribute-name/enum contract, rather than on exported span data.
"""

from __future__ import annotations

import pytest
from opentelemetry.trace import Span

from lazycore.telemetry import (
    ATTR_GENAI_EVENT_CONTENT,
    ATTR_GENAI_EVENT_ROLE,
    ATTR_ML_METRIC_ACCURACY,
    ATTR_ML_METRIC_PREFIX,
    ATTR_OWASP_MAPPING,
    ATTR_SECURITY_SEVERITY,
    SecuritySeverity,
    add_transcript_event,
    genai_span,
    get_tracer,
    ml_metric_attribute,
    set_ml_metric,
    set_security_finding,
)


def test_attribute_name_constants_match_architecture_doc():
    """The shared attribute-name constants match the exact strings specified in architecture doc §2.6."""
    assert ATTR_SECURITY_SEVERITY == "security.severity"
    assert ATTR_OWASP_MAPPING == "owasp.mapping"
    assert ATTR_ML_METRIC_ACCURACY == "ml.metric.accuracy"
    assert ATTR_ML_METRIC_PREFIX == "ml.metric."
    assert ATTR_GENAI_EVENT_ROLE == "gen_ai.event.role"
    assert ATTR_GENAI_EVENT_CONTENT == "gen_ai.event.content"


def test_ml_metric_attribute_namespaces_metric_names():
    """ml_metric_attribute() prefixes an arbitrary metric name with the shared "ml.metric." namespace."""
    assert ml_metric_attribute("accuracy") == "ml.metric.accuracy"
    assert ml_metric_attribute("picp") == "ml.metric.picp"


def test_ml_metric_attribute_rejects_empty_name():
    """ml_metric_attribute() raises ValueError for an empty metric name."""
    with pytest.raises(ValueError):
        ml_metric_attribute("")


def test_security_severity_enum_values():
    """SecuritySeverity members expose the expected lowercase string values."""
    assert SecuritySeverity.INFO.value == "info"
    assert SecuritySeverity.LOW.value == "low"
    assert SecuritySeverity.MEDIUM.value == "medium"
    assert SecuritySeverity.HIGH.value == "high"
    assert SecuritySeverity.CRITICAL.value == "critical"


def test_get_tracer_returns_a_tracer_like_object():
    """get_tracer() returns an object exposing the OTel Tracer API (start_as_current_span), even with no SDK configured."""
    tracer = get_tracer("test.module")
    assert hasattr(tracer, "start_as_current_span")


def test_genai_span_yields_a_span_and_does_not_raise():
    """genai_span() is a usable context manager that yields a real Span instance without raising, even as a no-op (SDK-less) span."""
    with genai_span("test.span") as span:
        assert isinstance(span, Span)


def test_set_security_finding_accepts_enum_member():
    """set_security_finding() does not raise when given a SecuritySeverity enum member plus a list of OWASP mapping IDs."""
    with genai_span("test.security") as span:
        # Should not raise for a valid enum member + owasp mapping list.
        set_security_finding(
            span, severity=SecuritySeverity.HIGH, owasp_mapping=["LLM01", "LLM06"]
        )


def test_set_security_finding_accepts_raw_string_severity():
    """set_security_finding() accepts a raw string severity (matching a SecuritySeverity value) and a single string OWASP mapping."""
    with genai_span("test.security.raw") as span:
        set_security_finding(span, severity="critical", owasp_mapping="LLM01")


def test_set_security_finding_rejects_invalid_severity():
    """set_security_finding() raises ValueError for a severity string that doesn't match any SecuritySeverity value."""
    with genai_span("test.security.invalid") as span:
        with pytest.raises(ValueError):
            set_security_finding(span, severity="apocalyptic")


def test_set_ml_metric_does_not_raise():
    """set_ml_metric() does not raise when attaching a numeric metric value to a span."""
    with genai_span("test.metric") as span:
        set_ml_metric(span, "accuracy", 0.987)


def test_add_transcript_event_does_not_raise_and_accepts_extra_attrs():
    """add_transcript_event() does not raise for multiple turns and accepts arbitrary extra keyword attributes (e.g. turn_index) alongside role/content."""
    with genai_span("test.transcript") as span:
        add_transcript_event(span, role="user", content="hello", turn_index=0)
        add_transcript_event(span, role="assistant", content="hi there", turn_index=1)
