"""Farmer alerts (src/alerts/alert_engine.py) and their English / Tamil text (src/dashboard/i18n.py)."""
import pytest

from src.alerts import alert_engine as ae


def _frame(peak=0.4, zones=(3, 7), rain=0.0, dry_share=0.0):
    n = 400
    moist = [0.3] * int(n * dry_share) + [0.9] * (n - int(n * dry_share))
    return {"field": {"what_if": {"no_spray": {"fungus_peak": peak}}, "moisture": moist, "pending_sprays": []},
            "metrics": {"harvest_t_ha": {"spray_zones": list(zones)}},
            "weather": {"precipitation_mm": rain}}


@pytest.fixture(autouse=True)
def _tmp_alert_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ae, "ALERT_DIR", tmp_path)


def test_rules():
    assert [a["key"] for a in ae.evaluate(_frame())] == ["alert_disease"]
    assert [a["key"] for a in ae.evaluate(_frame(rain=12))] == ["alert_disease", "alert_rain_spray"]
    assert ae.evaluate(_frame(peak=0.1)) == []                       # low risk: no alert
    assert ae.evaluate(_frame(zones=())) == []                       # nothing to spray: no alert
    assert [a["key"] for a in ae.evaluate(_frame(zones=(), dry_share=0.2))] == ["alert_water"]


def test_messages_in_both_languages():
    a = ae.evaluate(_frame())[0]
    assert "3, 7" in ae.message(a, "en") and "Spray" in ae.message(a, "en")
    ta = ae.message(a, "ta")
    assert "3, 7" in ta and "மண்டலம்" in ta


def test_send_logs_and_does_not_repeat(tmp_path):
    alerts = ae.evaluate(_frame())
    people = [{"name": "Muthu", "language": "ta", "channel": "log", "address": "muthu"},
              {"name": "Asha", "language": "en", "channel": "sms", "address": "+911234567890"}]
    res = ae.send(alerts, people, field_name="Plot 7")
    assert res[0]["status"] == "sent by log"
    assert res[1]["status"].startswith("not sent")                  # Twilio not configured in tests
    assert "மண்டலம்" in (tmp_path / "alerts.log").read_text(encoding="utf-8")
    again = ae.send(alerts, people[:1], field_name="Plot 7")
    assert again[0]["status"].startswith("skipped")                 # not repeated within REPEAT_HOURS
