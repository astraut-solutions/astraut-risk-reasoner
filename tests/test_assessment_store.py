from pathlib import Path

from astraut_risk.assessment_store import (
    cache_key_for_assessment,
    load_cached_result,
    save_cached_result,
)
from astraut_risk.risk_engine import assess_company_risk


def test_save_and_load_cached_result(tmp_path: Path) -> None:
    company_input = "12-person SaaS startup with public API and no MFA on admin"
    assessment = assess_company_risk(company_input)

    saved = save_cached_result(
        company_description=company_input,
        model="llama-3.3-70b-versatile",
        assessment=assessment,
        llm_explanation="## Risk Rationale\nCached explanation",
        assessment_markdown="## Overall Risk Score\n28/100",
        cache_dir=str(tmp_path),
    )

    assert saved.exists()

    loaded = load_cached_result(
        company_description=company_input,
        model="llama-3.3-70b-versatile",
        assessment=assessment,
        cache_dir=str(tmp_path),
    )
    assert loaded is not None
    assert loaded["llm_explanation"].startswith("## Risk Rationale")


def test_cache_key_changes_for_different_deterministic_signals() -> None:
    a = assess_company_risk("startup with no mfa and public api")
    b = assess_company_risk("startup with mfa enabled and no public api")

    key_a = cache_key_for_assessment("A", "llama-3.3-70b-versatile", a)
    key_b = cache_key_for_assessment("B", "llama-3.3-70b-versatile", b)

    assert key_a != key_b
