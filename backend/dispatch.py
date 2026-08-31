"""Send a confirmed alert to the officers responsible for that camera.

What this does
--------------
Turns an alert into a message a person can act on: which camera, which street,
what was seen, how serious, and a picture. Delivered by WhatsApp, email, or
both, to whoever covers that area.

Two registries drive it, both plain JSON you can edit by hand:

  config/cameras.json     camera id -> name, address, coordinates, area,
                          and the police post that covers it
  config/recipients.json  officers -> name, areas they cover, WhatsApp
                          number, email, and the minimum severity worth
                          waking them for

Why manual by default
---------------------
`CLAUDE.md`'s ground rules say an alert is "a prioritisation aid for a human
reviewer, never a finding, and never routed to automated action". Texting a
police officer *is* automated action, and the models here cannot see ownership,
consent or intent — a parent taking a phone from a child produces the same
alert as a thief.

So `SENTINEL_DISPATCH_MODE` defaults to `manual`: the system prepares the
message and waits for a person to press send. Setting it to `auto` is possible
and is a deliberate choice with consequences; it logs loudly every time it
fires.

Nothing sends without credentials, and with none configured every call is a
dry run that returns exactly what *would* have been sent. That is the safe
default for a demo.

Configuration
-------------
  SENTINEL_DISPATCH_MODE      manual (default) | auto | off
  SENTINEL_PUBLIC_URL         public base URL for snapshots. WhatsApp media
                              must be fetchable by the provider, so a
                              localhost address yields a text-only message.

  Email (SMTP):  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM
  WhatsApp (Twilio):  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
                      TWILIO_WHATSAPP_FROM  (e.g. "whatsapp:+14155238886")
"""
from __future__ import annotations

import json
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

SEVERITY_ORDER = ["none", "low", "medium", "high", "critical"]


def _rank(sev: str) -> int:
    try:
        return SEVERITY_ORDER.index((sev or "low").lower())
    except ValueError:
        return 1


def _load(name: str) -> dict:
    p = CONFIG_DIR / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        # A malformed registry must not silently mean "nobody is on call".
        print(f"[dispatch] {name} is not valid JSON: {exc}", flush=True)
        return {}


def cameras() -> dict:
    return _load("cameras.json")


def recipients() -> list[dict]:
    data = _load("recipients.json")
    return data.get("recipients", []) if isinstance(data, dict) else data


def save_recipients(items: list[dict]) -> dict:
    """Overwrite the recipient registry from the UI.

    Numbers are stored exactly as typed. Guessing a country code is worse
    than refusing: "0700123456" could be Uganda or Kenya, and a silently
    rewritten number fails as a delivery error nobody connects to the typo.
    So a number that is not in +country-code form is stored and reported
    back as a warning the operator can see and fix.
    """
    cleaned: list[dict] = []
    warnings: list[str] = []
    for r in items or []:
        name = (r.get("name") or "").strip()
        if not name:
            continue
        wa = (r.get("whatsapp") or "").strip()
        email = (r.get("email") or "").strip()
        if not wa and not email:
            warnings.append(f"{name}: no WhatsApp number and no email — cannot be reached")
        if wa and not wa.startswith("+"):
            warnings.append(f"{name}: \"{wa}\" needs a country code, e.g. +256{wa.lstrip('0')}")
        sev = (r.get("min_severity") or "high").lower()
        if sev not in SEVERITY_ORDER:
            warnings.append(f"{name}: unknown severity \"{sev}\", using high")
            sev = "high"
        cleaned.append({
            "name": name,
            "areas": [a.strip() for a in (r.get("areas") or []) if a.strip()],
            "email": email,
            "whatsapp": wa,
            "min_severity": sev,
        })

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "recipients.json").write_text(
        json.dumps({
            "_comment": "Officers who receive alerts. 'areas' empty means every "
                        "camera. 'min_severity' is the lowest level worth "
                        "sending: low, medium, high, critical.",
            "recipients": cleaned,
        }, indent=2) + "\n",
        encoding="utf-8")
    return {"recipients": cleaned, "warnings": warnings}


def resolve_camera(camera_id: str) -> dict:
    """Location for a camera id, or a clearly-marked unknown."""
    cam = cameras().get(camera_id or "")
    if cam:
        return {"id": camera_id, **cam}
    return {"id": camera_id or "unknown", "name": camera_id or "Unknown camera",
            "address": "location not registered", "area": None,
            "lat": None, "lng": None, "police_post": None}


def for_alert(alert: dict, camera_id: str) -> list[dict]:
    """Officers who cover this camera's area and want this severity."""
    cam = resolve_camera(camera_id)
    sev = _rank(alert.get("severity", "low"))
    out = []
    for r in recipients():
        areas = [str(a).strip().lower() for a in (r.get("areas") or []) if str(a).strip()]
        # An empty area list means "everywhere" — useful for a control room.
        # So does the word people actually type when they mean it. The field
        # says "blank = all", so "all" is the obvious thing to write, and it
        # used to match no camera at all: the recipient saved fine and then
        # silently received nothing, which is the worst way for this to fail.
        everywhere = not areas or bool({"all", "any", "*"} & set(areas))
        if not everywhere and (cam.get("area") or "").lower() not in areas:
            continue
        if sev < _rank(r.get("min_severity", "high")):
            continue
        out.append(r)
    return out


def compose(alert: dict, camera_id: str) -> dict:
    """Build the human-readable message. Pure — sends nothing."""
    cam = resolve_camera(camera_id)
    sev = (alert.get("severity") or "low").upper()
    when = alert.get("timecode") or alert.get("timestamp", "")
    where = cam["name"]
    if cam.get("address"):
        where += f", {cam['address']}"

    lines = [
        f"{sev}: {alert.get('label') or alert.get('type', 'Incident')}",
        f"Camera : {where}",
        f"Time   : {when}",
    ]
    if cam.get("lat") is not None and cam.get("lng") is not None:
        lines.append(f"Map    : https://maps.google.com/?q={cam['lat']},{cam['lng']}")
    if alert.get("detail"):
        lines.append(f"Seen   : {alert['detail']}")
    lines.append(f"Confidence: {alert.get('score', 0)}")
    lines.append("")
    # Every message says what the system is and is not. An officer acting on
    # this must know a machine flagged a movement, not that a crime is proven.
    lines.append("Automated flag for human review — not a confirmed crime. "
                 "Sentinel detects movement, not intent.")

    snap_url = None
    name = alert.get("snapshot")
    if name:
        base = os.environ.get("SENTINEL_PUBLIC_URL", "").rstrip("/")
        if base:
            snap_url = f"{base}/api/alerts/snapshot/{Path(name).name}"

    return {
        "subject": f"[Sentinel {sev}] {alert.get('label', 'Incident')} — {cam['name']}",
        "text": "\n".join(lines),
        "camera": cam,
        "snapshot_name": Path(name).name if name else None,
        "snapshot_url": snap_url,
    }


def _snapshot_path(name: str | None) -> Path | None:
    if not name:
        return None
    for cand in (ROOT / "logs" / "snapshots" / name,
                 ROOT / "logs" / "snapshots" / "alerts" / name):
        if cand.exists():
            return cand
    return None


def send_email(to: str, msg: dict) -> dict:
    host = os.environ.get("SMTP_HOST", "").strip()
    if not host:
        return {"channel": "email", "to": to, "sent": False, "reason": "SMTP not configured"}
    try:
        em = EmailMessage()
        em["Subject"] = msg["subject"]
        em["From"] = os.environ.get("SMTP_FROM", os.environ.get("SMTP_USER", ""))
        em["To"] = to
        em.set_content(msg["text"])
        # Email can carry the picture itself, so it does not need the server
        # to be reachable from the internet the way WhatsApp media does.
        snap = _snapshot_path(msg.get("snapshot_name"))
        if snap:
            em.add_attachment(snap.read_bytes(), maintype="image",
                              subtype="jpeg", filename=snap.name)
        port = int(os.environ.get("SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=20) as s:
            s.starttls(context=ssl.create_default_context())
            user = os.environ.get("SMTP_USER")
            if user:
                s.login(user, os.environ.get("SMTP_PASSWORD", ""))
            s.send_message(em)
        return {"channel": "email", "to": to, "sent": True,
                "attached": bool(snap)}
    except Exception as exc:
        return {"channel": "email", "to": to, "sent": False,
                "reason": f"{type(exc).__name__}: {exc}"}


def send_whatsapp(to: str, msg: dict) -> dict:
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    frm = os.environ.get("TWILIO_WHATSAPP_FROM", "").strip()
    if not (sid and token and frm):
        return {"channel": "whatsapp", "to": to, "sent": False,
                "reason": "Twilio not configured"}
    if not to.replace("whatsapp:", "").startswith("+"):
        # Caught here rather than at Twilio, whose error for this is a bare
        # code 21211 that gives an operator nothing to act on.
        return {"channel": "whatsapp", "to": to, "sent": False,
                "reason": f"{to} needs a country code, e.g. +256..."}
    try:
        import requests
        data = {"From": frm,
                "To": to if to.startswith("whatsapp:") else f"whatsapp:{to}",
                "Body": msg["text"]}
        # Twilio fetches media over the public internet, so a snapshot only
        # attaches when SENTINEL_PUBLIC_URL points somewhere reachable. On
        # localhost the message still goes, text-only.
        if msg.get("snapshot_url"):
            data["MediaUrl"] = msg["snapshot_url"]
        r = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
            data=data, auth=(sid, token), timeout=20)
        ok = r.status_code < 300
        return {"channel": "whatsapp", "to": to, "sent": ok,
                "media": bool(msg.get("snapshot_url")),
                "reason": None if ok else f"HTTP {r.status_code}: {r.text[:160]}"}
    except Exception as exc:
        return {"channel": "whatsapp", "to": to, "sent": False,
                "reason": f"{type(exc).__name__}: {exc}"}


def dispatch(alert: dict, camera_id: str, confirmed_by: str | None = None,
             force: bool = False) -> dict:
    """Send an alert to everyone on call for that camera.

    `force=True` is what a human pressing "Send to police" supplies. Without
    it, dispatch only proceeds in `auto` mode — so the default build cannot
    message anyone without a person in the loop.
    """
    mode = os.environ.get("SENTINEL_DISPATCH_MODE", "manual").lower()
    msg = compose(alert, camera_id)
    people = for_alert(alert, camera_id)

    if mode == "off":
        return {"mode": mode, "sent": False, "reason": "dispatch disabled",
                "preview": msg, "recipients": people}
    if mode != "auto" and not force:
        # The normal path: everything is ready, nothing has been sent, and a
        # person decides.
        return {"mode": mode, "sent": False, "reason": "awaiting human confirmation",
                "preview": msg, "recipients": people}
    if mode == "auto" and not force:
        print(f"[dispatch] AUTO-SENDING without human review: "
              f"{alert.get('severity')} {alert.get('label')} @ {camera_id}",
              flush=True)

    results = []
    for r in people:
        if r.get("email"):
            results.append(send_email(r["email"], msg))
        if r.get("whatsapp"):
            results.append(send_whatsapp(r["whatsapp"], msg))

    delivered = sum(1 for x in results if x.get("sent"))
    print(f"[dispatch] {alert.get('severity')} {alert.get('label')} @ "
          f"{camera_id}: {delivered}/{len(results)} delivered"
          f"{' (confirmed by ' + confirmed_by + ')' if confirmed_by else ''}",
          flush=True)
    return {"mode": mode, "sent": delivered > 0, "delivered": delivered,
            "attempted": len(results), "results": results,
            "preview": msg, "recipients": people,
            "confirmed_by": confirmed_by}
