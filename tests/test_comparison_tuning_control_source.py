from pathlib import Path


CONTROL = Path("frontend/src/components/ComparisonTuningControl.jsx")
API = Path("frontend/src/api.js")
APP = Path("frontend/src/App.jsx")


def test_comparison_tuning_control_is_nested_in_existing_card() -> None:
    """The tuning switch should be inserted below activation without a new card."""

    control = CONTROL.read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")

    assert "比較後のモデル" in control
    assert 'insertAdjacentElement("afterend", nextHost)' in control
    assert "ベストモデルをチューニング" in control
    assert "disabled={!activateBest}" in control
    assert "<ComparisonTuningControl />" in app


def test_enabled_control_promotes_compare_request_to_tune_best() -> None:
    """The API request should tune only when activation and the switch are enabled."""

    source = API.read_text(encoding="utf-8")

    assert "setComparisonTuneBestOverride" in source
    assert "if (!payload?.activate_best || !comparisonTuneBestOverride) return payload;" in source
    assert "return { ...payload, tune_best: true };" in source
    assert "JSON.stringify(comparisonPayload(payload))" in source
