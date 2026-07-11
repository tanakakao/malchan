from malchan.app import AppSettings, get_settings
from malchan.app.core import AppSettings as CoreAppSettings


def test_app_settings_are_available_from_app_package() -> None:
    """Application settings should be importable without FastAPI installed."""

    settings = get_settings()

    assert isinstance(settings, AppSettings)
    assert AppSettings is CoreAppSettings
    assert settings.api_prefix == "/api"
