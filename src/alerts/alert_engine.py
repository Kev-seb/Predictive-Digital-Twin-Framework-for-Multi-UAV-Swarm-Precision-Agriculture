"""
alert_engine.py - turn the live twin's state into farmer alerts, and send them.

    alerts = evaluate(frame)                    # frame = GET /ue5/frame from the twin API
    send(alerts, recipients)                    # log + email / SMS / WhatsApp, in each person's language

Channels (configured with environment variables; a channel that is not configured is skipped):
    log        always on: outputs/alerts/alerts.log
    email      SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_FROM
    sms        TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_SMS_FROM            (Twilio REST API)
    whatsapp   TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM       (e.g. whatsapp:+14155238886)

The same alert is not sent to the same person again within REPEAT_HOURS.
"""
from __future__ import annotations

import json
import os
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
ALERT_DIR = ROOT / "outputs" / "alerts"
REPEAT_HOURS = 12.0

DISEASE_PEAK = 0.25          # 7-day worst-cell fungus above this, with zones to spray -> alert
RAIN_MM = 1.0
DRY_MOISTURE = 0.45          # cell moisture below this counts as drying
DRY_SHARE = 0.10             # alert when this share of the field is drying


def evaluate(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Alerts from one /ue5/frame. Each: {id, level, key, params}; text comes from i18n in any language."""
    out: List[Dict[str, Any]] = []
    field = frame.get("field") or {}
    m = frame.get("metrics") or {}
    wi = field.get("what_if") or {}
    zones = (m.get("harvest_t_ha") or {}).get("spray_zones") or field.get("pending_sprays") or []
    peak = (wi.get("no_spray") or {}).get("fungus_peak", 0.0)
    rain = (frame.get("weather") or {}).get("precipitation_mm", 0.0) or 0.0
    if zones and peak >= DISEASE_PEAK:
        z = ", ".join(str(x) for x in zones)
        out.append({"id": f"disease:{z}", "level": "high", "key": "alert_disease", "params": {"zones": z}})
        if rain > RAIN_MM:
            out.append({"id": "rain_spray", "level": "medium", "key": "alert_rain_spray", "params": {"mm": rain}})
    moist = field.get("moisture") or []
    if moist and sum(1 for v in moist if v < DRY_MOISTURE) / len(moist) >= DRY_SHARE:
        out.append({"id": "water", "level": "medium", "key": "alert_water", "params": {}})
    return out


def message(alert: Dict[str, Any], language: str = "en") -> str:
    from src.dashboard.i18n import text_in
    return text_in(alert["key"], language, **alert.get("params", {}))


# ------------------------------------------------------------------ channels
def channels_configured() -> Dict[str, bool]:
    e = os.environ
    twilio = bool(e.get("TWILIO_ACCOUNT_SID") and e.get("TWILIO_AUTH_TOKEN"))
    return {"log": True, "email": bool(e.get("SMTP_HOST") and e.get("ALERT_EMAIL_FROM")),
            "sms": twilio and bool(e.get("TWILIO_SMS_FROM")),
            "whatsapp": twilio and bool(e.get("TWILIO_WHATSAPP_FROM"))}


def _log(text: str, who: str) -> None:
    ALERT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ALERT_DIR / "alerts.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{who}\t{text}\n")


def _email(to: str, text: str, subject: str) -> None:
    e = os.environ
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = e["ALERT_EMAIL_FROM"], to, subject
    msg.set_content(text)
    with smtplib.SMTP(e["SMTP_HOST"], int(e.get("SMTP_PORT", "587")), timeout=20) as s:
        s.starttls()
        if e.get("SMTP_USER"):
            s.login(e["SMTP_USER"], e.get("SMTP_PASSWORD", ""))
        s.send_message(msg)


def _twilio(to: str, text: str, whatsapp: bool) -> None:
    import requests
    e = os.environ
    sid = e["TWILIO_ACCOUNT_SID"]
    sender = e["TWILIO_WHATSAPP_FROM"] if whatsapp else e["TWILIO_SMS_FROM"]
    dest = (to if to.startswith("whatsapp:") else f"whatsapp:{to}") if whatsapp else to
    r = requests.post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
                      data={"From": sender, "To": dest, "Body": text},
                      auth=(sid, e["TWILIO_AUTH_TOKEN"]), timeout=20)
    r.raise_for_status()


# ------------------------------------------------------------------ sending
def _sent_state() -> Dict[str, float]:
    p = ALERT_DIR / "sent.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def send(alerts: List[Dict[str, Any]], recipients: List[Dict[str, Any]], field_name: str = "your field",
         force: bool = False) -> List[Dict[str, Any]]:
    """recipients: [{name, language, channel: email|sms|whatsapp|log, address}]. Returns one result per send."""
    state, now, results = _sent_state(), time.time(), []
    ok = channels_configured()
    for r in recipients:
        for a in alerts:
            key = f"{r.get('address')}|{a['id']}"
            if not force and now - state.get(key, 0) < REPEAT_HOURS * 3600:
                results.append({"to": r.get("name"), "alert": a["id"], "status": "skipped (sent recently)"})
                continue
            text = f"PaddyTwin · {field_name}: {message(a, r.get('language', 'en'))}"
            ch = r.get("channel", "log")
            try:
                if ch != "log" and not ok.get(ch):
                    raise RuntimeError(f"{ch} is not configured")
                if ch == "email":
                    _email(r["address"], text, f"PaddyTwin alert: {field_name}")
                elif ch in ("sms", "whatsapp"):
                    _twilio(r["address"], text, whatsapp=ch == "whatsapp")
                _log(text, f"{r.get('name')} via {ch}")
                state[key] = now
                results.append({"to": r.get("name"), "alert": a["id"], "status": f"sent by {ch}"})
            except Exception as e:
                _log(f"FAILED ({e}): {text}", f"{r.get('name')} via {ch}")
                results.append({"to": r.get("name"), "alert": a["id"], "status": f"not sent: {e}"})
    ALERT_DIR.mkdir(parents=True, exist_ok=True)
    (ALERT_DIR / "sent.json").write_text(json.dumps(state), encoding="utf-8")
    return results


def load_recipients() -> List[Dict[str, Any]]:
    p = ALERT_DIR / "recipients.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_recipients(recipients: List[Dict[str, Any]]) -> None:
    ALERT_DIR.mkdir(parents=True, exist_ok=True)
    (ALERT_DIR / "recipients.json").write_text(json.dumps(recipients, ensure_ascii=False, indent=2), encoding="utf-8")
