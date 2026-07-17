"""Tests for lazycore.licensing (license-isolation + allowlist policy, §2.2/§2.10)."""

from __future__ import annotations

import pytest

from lazycore.licensing import (
    Allowlist,
    Mitigation,
    ModelLicenseEntry,
    ModelTier,
    RestrictedLicenseNotAcceptedError,
    RISK_MITIGATIONS,
    RiskType,
)


# ---------------------------------------------------------------------------
# §2.2 License Isolation Policy decision table
# ---------------------------------------------------------------------------


def test_every_risk_type_has_a_mitigation_entry():
    for risk_type in RiskType:
        assert risk_type in RISK_MITIGATIONS
        mitigation = RISK_MITIGATIONS[risk_type]
        assert isinstance(mitigation, Mitigation)
        assert mitigation.risk_type is risk_type
        assert mitigation.required_mitigation  # non-empty
        assert mitigation.notes  # non-empty


def test_agpl_gpl_runtime_internal_is_flagged_for_counsel():
    mitigation = RISK_MITIGATIONS[RiskType.AGPL_GPL_RUNTIME_INTERNAL]
    assert mitigation.flag_for_counsel is True
    assert "subprocess" in mitigation.required_mitigation.lower()


def test_agpl_gpl_network_facing_is_flagged_for_counsel():
    mitigation = RISK_MITIGATIONS[RiskType.AGPL_GPL_NETWORK_FACING]
    assert mitigation.flag_for_counsel is True


def test_gpl_build_time_link_is_not_flagged_for_counsel():
    mitigation = RISK_MITIGATIONS[RiskType.GPL_BUILD_TIME_LINK]
    assert mitigation.flag_for_counsel is False


def test_non_commercial_weights_mitigation_references_opt_in_flag():
    mitigation = RISK_MITIGATIONS[RiskType.NON_COMMERCIAL_WEIGHTS]
    assert "accept_restricted_licenses" in mitigation.required_mitigation


def test_mitigation_dataclass_is_immutable():
    mitigation = RISK_MITIGATIONS[RiskType.GPL_BUILD_TIME_LINK]
    with pytest.raises(Exception):
        mitigation.required_mitigation = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# §2.10 Model Licensing Allowlist Policy
# ---------------------------------------------------------------------------


def test_new_allowlist_starts_empty():
    allowlist = Allowlist()
    assert allowlist.list_tier(ModelTier.TIER_1) == []
    assert allowlist.list_tier(ModelTier.TIER_2) == []
    assert "anything" not in allowlist


def test_register_and_check_tier1_entry_succeeds_without_opt_in():
    allowlist = Allowlist()
    allowlist.register("timesfm-2.5", ModelTier.TIER_1, "Apache-2.0")

    entry = allowlist.check("timesfm-2.5")
    assert isinstance(entry, ModelLicenseEntry)
    assert entry.tier is ModelTier.TIER_1
    assert entry.license_identifier == "Apache-2.0"


def test_check_unknown_entry_raises_key_error():
    allowlist = Allowlist()
    with pytest.raises(KeyError):
        allowlist.check("not-registered")


def test_tier2_entry_raises_without_opt_in_flag():
    allowlist = Allowlist()
    allowlist.register(
        "moirai-1.0",
        ModelTier.TIER_2,
        "CC-BY-NC-4.0",
        notes="Salesforce MOIRAI weights are non-commercial.",
    )

    with pytest.raises(RestrictedLicenseNotAcceptedError):
        allowlist.check("moirai-1.0")

    # Explicitly-false is treated the same as omitted.
    with pytest.raises(RestrictedLicenseNotAcceptedError):
        allowlist.check("moirai-1.0", accept_restricted_licenses=False)


def test_tier2_entry_succeeds_with_explicit_opt_in():
    allowlist = Allowlist()
    allowlist.register("moirai-1.0", ModelTier.TIER_2, "CC-BY-NC-4.0")

    entry = allowlist.check("moirai-1.0", accept_restricted_licenses=True)
    assert entry.tier is ModelTier.TIER_2
    assert entry.name == "moirai-1.0"


def test_list_tier_partitions_entries_correctly():
    allowlist = Allowlist()
    allowlist.register("timesfm-2.5", ModelTier.TIER_1, "Apache-2.0")
    allowlist.register("chronos-bolt", ModelTier.TIER_1, "Apache-2.0")
    allowlist.register("moirai-1.0", ModelTier.TIER_2, "CC-BY-NC-4.0")

    tier1_names = {entry.name for entry in allowlist.list_tier(ModelTier.TIER_1)}
    tier2_names = {entry.name for entry in allowlist.list_tier(ModelTier.TIER_2)}

    assert tier1_names == {"timesfm-2.5", "chronos-bolt"}
    assert tier2_names == {"moirai-1.0"}


def test_register_overwrites_existing_entry():
    allowlist = Allowlist()
    allowlist.register("some-model", ModelTier.TIER_2, "CC-BY-NC-4.0")
    allowlist.register("some-model", ModelTier.TIER_1, "Apache-2.0")

    entry = allowlist.check("some-model")  # no opt-in needed now
    assert entry.tier is ModelTier.TIER_1
    assert entry.license_identifier == "Apache-2.0"


def test_get_returns_none_for_missing_entry_without_raising():
    allowlist = Allowlist()
    assert allowlist.get("missing") is None
