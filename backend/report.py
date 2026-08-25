"""Incident reporting — turn an analysis run into something a person can act on.

The report is built deterministically from the event log. An LLM narrative is
added when one is reachable, but it is decoration: every fact in the report —
timecodes, counts, severities, entity links, recommended actions — comes from
the structured data, so a report is still complete and correct with no model
running. That ordering is deliberate. A report whose contents depend on a
1.5B model's mood is not evidence of anything.

Two ideas shape the output:

**Clustering.** Forty `loitering` events over three minutes is one observation,
not forty. Consecutive events of the same type are merged into an incident with
a first and last timecode and an occurrence count, so the timeline reads like a
sequence of things that happened.

**Provenance.** Every incident states which tier produced it. A `weapon_brandished`
grounded in a detector box, a wrist keypoint and a VLM confirmation is a
different object from a `theft_reported` that is one regex hit on a caption, and
the report never lets those two look alike.
"""
from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from backend.identity import format_timecode
from backend.threat import SEVERITY_ORDER
from backend.weapons import weapon_model_status

# Event types whose only evidence is a regex hit on a VLM caption.
CAPTION_ONLY_TYPES = {
    "weapon_reported", "shooting_reported", "violence_reported", "robbery_reported",
    "theft_reported", "burglary_reported", "vandalism_reported", "fire_reported",
    "intrusion_reported", "person_down_reported", "pursuit_reported",
    "crowd_panic_reported", "narcotics_reported", "collision_reported",
}

# What a reviewer should do about each event type. Phrased as verification
# steps, because none of these are findings.
ACTION_TEMPLATES: dict[str, str] = {
    "weapon_brandished": "Review {tc} — subject appears to be holding a weapon. Confirm visually before any dispatch decision.",
    "weapon_carried": "Review {tc} — possible weapon on a person, not clearly in hand. Confirm.",
    "weapon_near_person": "Review {tc} — weapon-class object near a person; pose did not place it in a hand. Low priority unless corroborated.",
    "weapon_visible": "Review {tc} — weapon-class object with nobody nearby. Likely a misclassification; confirm.",
    "crime_tool_held": "Review {tc} — subject holding a tool associated with forced entry. Check what they do next.",
    "crime_tool_carried": "Review {tc} — possible entry tool on a person. Confirm.",
    "violence": "Review {tc} — pose model reports an altercation. Check for injuries and whether anyone left the scene.",
    "person_down": "Review {tc} — a person may be injured or unwell. Treat as a welfare check first.",
    "vehicle_collision": "Review {tc} — probable vehicle collision. Identify both vehicles and check for casualties.",
    "vehicle_pedestrian_collision": "Review {tc} — a vehicle appears to have struck a pedestrian. Treat as urgent; preserve this segment.",
    "vehicle_left_scene": "Review {tc} — a vehicle involved in a collision left the frame. Check whether anyone stopped, and preserve plate-legible frames.",
    "zone_intrusion": "Review {tc} — person inside a restricted zone.",
    "loitering": "Review {tc} — prolonged presence. Corroborate against other signals before acting.",
    "unattended_object": "Review {tc} — object left unattended. Follow your site's procedure.",
    "crowd_dispersal": "Review {tc} — sudden dispersal often follows an incident. Check the preceding 30 seconds.",
    "crowd_surge": "Review {tc} — rapid crowd build-up.",
    "after_hours_presence": "Review {tc} — presence outside permitted hours.",
}

_DEFAULT_ACTION = "Review {tc} — automated signal raised; verify against the footage."


# ── Clustering ────────────────────────────────────────────────────────────

def cluster_events(events: list[dict], gap_s: float = 20.0) -> list[dict]:
    """Merge repeats of the same event type into single incidents.

    Events without a video clock (a live session) fall back to `monotonic`,
    so clustering behaves the same on both paths.
    """
    def when(e: dict) -> float:
        t = e.get("video_time_s")
        return float(t) if t is not None else float(e.get("monotonic") or 0.0)

    ordered = sorted(events, key=when)
    clusters: list[dict] = []
    open_by_type: dict[str, dict] = {}

    for ev in ordered:
        t = when(ev)
        etype = ev["type"]
        cur = open_by_type.get(etype)
        if cur is not None and t - cur["_last_t"] <= gap_s:
            cur["count"] += 1
            cur["_last_t"] = t
            cur["last_time_s"] = round(t, 2)
            cur["last_timecode"] = ev.get("timecode") or format_timecode(t)
            cur["peak_score"] = max(cur["peak_score"], ev["score"])
            if SEVERITY_ORDER.get(ev["severity"], 0) > SEVERITY_ORDER.get(cur["severity"], 0):
                cur["severity"] = ev["severity"]
                cur["detail"] = ev["detail"]
            cur["event_ids"].append(ev["id"])
            if ev.get("snapshot") and len(cur["snapshots"]) < 5:
                cur["snapshots"].append(ev["snapshot"])
            for link in ev.get("entities", []):
                if link["entity_id"] not in {e["entity_id"] for e in cur["entities"]}:
                    cur["entities"].append(link)
            continue

        cluster = {
            "type": etype,
            "label": ev["label"],
            "category": ev.get("category"),
            "severity": ev["severity"],
            "rule": ev.get("rule"),
            "detail": ev["detail"],
            "count": 1,
            "first_time_s": round(t, 2),
            "first_timecode": ev.get("timecode") or format_timecode(t),
            "last_time_s": round(t, 2),
            "last_timecode": ev.get("timecode") or format_timecode(t),
            "peak_score": ev["score"],
            "frame": ev.get("frame"),
            "event_ids": [ev["id"]],
            "snapshots": [ev["snapshot"]] if ev.get("snapshot") else [],
            "entities": list(ev.get("entities", [])),
            "evidence_tier": "caption" if etype in CAPTION_ONLY_TYPES else "detector",
            "verification": ev.get("verification"),
            "repeat_subject": ev.get("repeat_subject", False),
            "_last_t": t,
        }
        clusters.append(cluster)
        open_by_type[etype] = cluster

    for c in clusters:
        c.pop("_last_t", None)
    clusters.sort(key=lambda c: (-SEVERITY_ORDER.get(c["severity"], 0),
                                 c["first_time_s"]))
    return clusters


# ── Report construction ───────────────────────────────────────────────────

def build_report(job: Any, cluster_gap_s: float = 20.0,
                 narrative: bool = True,
                 ollama_url: str | None = None,
                 model: str = "deepseek-r1:1.5b") -> dict:
    """Build the full report for a completed `AnalysisJob`.

    `job` is duck-typed so live sessions can pass a lightweight stand-in with
    the same attributes.
    """
    events = list(getattr(job, "events", []))
    captions = list(getattr(job, "captions", []))
    entities = list(getattr(job, "entities", []))
    video = dict(getattr(job, "video", {}) or {})
    options = getattr(job, "options", None)

    incidents = cluster_events(events, gap_s=cluster_gap_s)
    by_sev = Counter(e["severity"] for e in events)
    by_type = Counter(e["type"] for e in events)

    vehicles = [e for e in entities if e["kind"] == "vehicle"]
    persons = [e for e in entities if e["kind"] == "person"]

    # A subject seen in an earlier incident is the single most report-worthy
    # fact the identity registry produces, so it gets its own section.
    repeats = [e for e in entities if e.get("incident_count", 0) > 1]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_id": getattr(job, "job_id", None),
        "source": {
            "video": getattr(job, "video_name", None),
            "path": getattr(job, "video_path", None),
            "duration_s": video.get("duration_s"),
            "duration_timecode": format_timecode(video.get("duration_s")),
            "fps": video.get("fps"),
            "resolution": (f"{video.get('width')}x{video.get('height')}"
                           if video.get("width") else None),
            "analysed_frames": (getattr(job, "progress", {}) or {}).get("frames_processed"),
            "status": getattr(job, "status", None),
        },
        "summary": _summary(incidents, by_sev, by_type, video, entities),
        "incidents": incidents,
        "suspects": _people_section(persons),
        "vehicles": _vehicle_section(vehicles),
        "repeat_subjects": [
            {
                "entity_id": e["entity_id"], "kind": e["kind"], "label": e["label"],
                "incident_count": e["incident_count"],
                "incidents": [
                    {"type": i.get("type"), "timecode": i.get("timecode"),
                     "source": i.get("source"), "severity": i.get("severity")}
                    for i in e.get("incidents", [])
                ],
            }
            for e in repeats
        ],
        "observations": [
            {"timecode": c["timecode"], "video_time_s": c["video_time_s"],
             "text": c["text"]}
            for c in captions
            if not c["text"].lower().startswith("no incident observed")
        ][:40],
        "recommended_actions": _actions(incidents),
        "caveats": _caveats(incidents, options, captions,
                            dict(getattr(job, "counters", {}) or {})),
        "counts": {
            "events": len(events),
            "incidents": len(incidents),
            "by_severity": dict(by_sev),
            "by_type": dict(by_type),
            "captions": len(captions),
            "entities": len(entities),
        },
        "narrative": None,
    }

    if narrative:
        report["narrative"] = _narrative(report, ollama_url=ollama_url, model=model)
    return report


def _summary(incidents, by_sev, by_type, video, entities) -> dict:
    critical = [i for i in incidents if i["severity"] == "critical"]
    high = [i for i in incidents if i["severity"] == "high"]
    headline = critical or high

    if not incidents:
        text = ("No incidents were raised. This means no rule fired and no caption "
                "matched — not that nothing happened.")
    else:
        lead = headline[0] if headline else incidents[0]
        text = (f"{len(incidents)} distinct incident(s) across "
                f"{format_timecode(video.get('duration_s'))} of footage. "
                f"Highest-priority: {lead['label']} at {lead['first_timecode']} "
                f"({lead['severity']}).")

    return {
        "headline": text,
        "critical_count": len(critical),
        "high_count": len(high),
        "first_incident_timecode": (min(incidents, key=lambda i: i["first_time_s"])
                                    ["first_timecode"] if incidents else None),
        "subjects_recorded": len(entities),
        "dominant_types": [t for t, _ in by_type.most_common(5)],
    }


def _people_section(persons: list[dict]) -> list[dict]:
    """Persons linked to incidents. Deliberately not called 'identification'."""
    out = []
    for p in persons:
        times = [s.get("timecode") for s in p.get("sightings", []) if s.get("timecode")]
        incidents = p.get("incidents", [])
        # "Involved" means a rule named this person's box. "Present" means they
        # were merely in frame when a caption-tier signal fired. Collapsing the
        # two would turn bystanders into suspects.
        involved = [i for i in incidents if i.get("role") == "involved"]
        out.append({
            "entity_id": p["entity_id"],
            "description": p["label"],
            "appearances": times[:20],
            "appearance_count": p.get("sighting_count", len(times)),
            "incident_count": p.get("incident_count", 0),
            "involved_count": len(involved),
            "present_only_count": len(incidents) - len(involved),
            "incidents": [
                {"type": i.get("type"), "timecode": i.get("timecode"),
                 "severity": i.get("severity"), "role": i.get("role")}
                for i in p.get("incidents", [])
            ],
            "basis": "clothing colour signature — an investigative lead, not an identification",
        })
    out.sort(key=lambda p: (-p["involved_count"], -p["incident_count"]))
    return out


def _vehicle_section(vehicles: list[dict]) -> list[dict]:
    out = []
    for v in vehicles:
        incidents = v.get("incidents", [])
        collision = [i for i in incidents
                     if i.get("type") in ("vehicle_collision",
                                          "vehicle_pedestrian_collision",
                                          "vehicle_left_scene")]
        out.append({
            "entity_id": v["entity_id"],
            "description": v["label"],
            "plate": v.get("plate"),
            "appearances": [s.get("timecode") for s in v.get("sightings", [])
                            if s.get("timecode")][:20],
            "incident_count": v.get("incident_count", 0),
            "involved_in_collision": bool(collision),
            "collision_events": [
                {"type": i.get("type"), "timecode": i.get("timecode"),
                 "role": i.get("role"), "severity": i.get("severity"),
                 "source": i.get("source")}
                for i in collision
            ],
            "left_scene": any(i.get("type") == "vehicle_left_scene" for i in incidents),
            "basis": "colour and silhouette signature — matches on re-appearance, not a plate read",
        })
    out.sort(key=lambda v: (not v["involved_in_collision"], -v["incident_count"]))
    return out


def _actions(incidents: list[dict]) -> list[dict]:
    actions = []
    for inc in incidents:
        template = ACTION_TEMPLATES.get(inc["type"], _DEFAULT_ACTION)
        text = template.format(tc=inc["first_timecode"])
        if inc["count"] > 1:
            text += f" ({inc['count']} occurrences to {inc['last_timecode']})"
        if inc.get("repeat_subject"):
            text += " Subject matches a previously recorded entity — see Repeat Subjects."
        actions.append({
            "priority": inc["severity"],
            "timecode": inc["first_timecode"],
            "video_time_s": inc["first_time_s"],
            "action": text,
            "evidence_tier": inc["evidence_tier"],
        })
    actions.sort(key=lambda a: (-SEVERITY_ORDER.get(a["priority"], 0),
                                a["video_time_s"]))
    return actions


def _caveats(incidents, options, captions, counters) -> list[str]:
    out: list[str] = [
        "Nothing in this report establishes that a crime occurred. Every entry is "
        "an automated signal for human review.",
    ]

    # A caption tier that was requested but produced nothing is the most
    # dangerous failure here: the report would otherwise look like a clean scan.
    if options is not None and getattr(options, "enable_vlm", False) and not captions:
        reason = counters.get("last_vlm_error") or "the model returned no text"
        out.append(
            f"**Caption screening was enabled but produced no captions** ({reason}). "
            "Open-vocabulary coverage was therefore absent from this run — an empty "
            "timeline here is not evidence that nothing happened."
        )
    elif counters.get("vlm_errors") or counters.get("vlm_empty"):
        out.append(
            f"The captioning model failed on "
            f"{counters.get('vlm_errors', 0) + counters.get('vlm_empty', 0)} sampled "
            f"frame(s) ({counters.get('last_vlm_error', 'no detail')}); coverage is "
            "patchier than the sampling interval implies."
        )

    wm = weapon_model_status()
    if not wm["loaded"]:
        out.append(
            "**No weapon-trained detector is loaded.** The detection tier can only "
            "produce `knife`, `baseball bat` and `scissors` — those are the only "
            "weapon-adjacent classes in COCO. Firearms, machetes/pangas, clubs, "
            "crowbars and every other implement are invisible to it regardless of "
            "confidence threshold, because the model has no output for them. An "
            "absence of weapon events in this report is therefore not evidence that "
            "no weapon was present. The captioning model is the only remaining "
            "signal, and on low-resolution or night footage it typically reports a "
            "held object without identifying it (observed: a machete described as "
            "'a black bag')."
        )

    caption_only = [i for i in incidents if i["evidence_tier"] == "caption"]
    if caption_only:
        types = ", ".join(sorted({i["type"] for i in caption_only}))
        out.append(
            f"{len(caption_only)} incident(s) rest solely on a keyword match against "
            f"a 0.5B captioning model ({types}). Treat these as prompts to look, not findings."
        )

    if options is not None:
        if not getattr(options, "enable_vlm", True):
            out.append("Caption screening was disabled, so open-vocabulary categories "
                       "(theft, robbery, arson, and anything else without a rule) were not covered.")
        if not getattr(options, "enable_pose", True):
            out.append("Pose was disabled, so grip reasoning and action recognition did not run; "
                       "weapon events fall back to proximity, which is markedly weaker.")
        stride = getattr(options, "stride", 1)
        if stride and stride > 4:
            out.append(f"Every {stride}th frame was analysed; events shorter than "
                       f"roughly {stride / 25:.2f}s may have been missed entirely.")

    if any(i["type"] in ("violence", "person_down") for i in incidents):
        out.append("Action recognition uses NTU-60, recorded indoors at close range with "
                   "actors facing the camera. On CCTV geometry it both misses and false-alarms.")

    if any(i["type"].startswith("vehicle_") for i in incidents):
        out.append("Collision detection infers impact from box overlap plus abrupt speed "
                   "change in the image plane. A camera angle where vehicles pass behind "
                   "one another produces false contacts; tune collision_min_speed and "
                   "collision_iou against your own footage.")

    if any(i.get("entities") for i in incidents):
        out.append("Subject re-identification is colour-based. It will confuse two similar "
                   "vehicles and will fail to match one vehicle across a large lighting "
                   "change. Every match carries a similarity score — check it.")

    out.append("Snapshots and identity records are personal data. Retention, signage and "
               "lawful basis are the operator's responsibility.")
    return out


# ── Narrative (optional) ──────────────────────────────────────────────────

def _narrative(report: dict, ollama_url: str | None = None,
               model: str = "deepseek-r1:1.5b") -> dict | None:
    """Ask a local LLM to write the prose summary. Never required."""
    url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
    incidents = report["incidents"][:15]
    if not incidents:
        return None

    lines = "\n".join(
        f"- [{i['first_timecode']}–{i['last_timecode']}] {i['severity'].upper()} "
        f"{i['label']} ×{i['count']} (score {i['peak_score']}, "
        f"evidence: {i['evidence_tier']}) — {i['detail']}"
        for i in incidents
    )
    subjects = "\n".join(
        f"- {v['description']} ({v['entity_id']}): "
        f"{'collision-involved' if v['involved_in_collision'] else 'seen'}, "
        f"{v['incident_count']} incident(s)"
        for v in report["vehicles"][:10]
    ) or "(none recorded)"

    prompt = (
        "You are writing the narrative section of a video review report for a human "
        "analyst. Below is the structured incident list from an automated system.\n\n"
        "Write 3–5 sentences describing what the footage appears to show, in "
        "chronological order, citing timecodes. Rules: do not assert that a crime "
        "occurred; do not invent details absent from the list; say explicitly when a "
        "signal is weak. Do not repeat the list verbatim.\n\n"
        f"## Incidents\n{lines}\n\n## Vehicles\n{subjects}\n\n## Narrative\n"
    )

    try:
        from openai import OpenAI
        client = OpenAI(base_url=f"{url.rstrip('/')}/v1", api_key="ollama", timeout=90)
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=400,
        )
        text = (resp.choices[0].message.content or "").strip()
        return {"model": model, "text": text} if text else None
    except Exception as exc:
        # The report is complete without this; record why it is absent.
        return {"model": model, "text": None,
                "error": f"{type(exc).__name__}: {exc}",
                "note": "Narrative unavailable — the structured report above is complete."}


# ── Markdown rendering ────────────────────────────────────────────────────

_SEV_MARK = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "info": "⚪"}


def render_markdown(report: dict) -> str:
    """Render the report as Markdown. Same facts, human layout."""
    src = report["source"]
    s = report["summary"]
    out: list[str] = []

    out.append(f"# Incident Report — {src.get('video') or 'live session'}")
    out.append("")
    out.append(f"*Generated {report['generated_at']} · job `{report.get('job_id')}`*")
    out.append("")
    out.append(f"**Source** · {src.get('video')} · {src.get('duration_timecode')} "
               f"· {src.get('resolution') or 'unknown resolution'} @ {src.get('fps')}fps "
               f"· {src.get('analysed_frames') or '?'} frames analysed")
    out.append("")

    out.append("## Summary")
    out.append("")
    out.append(s["headline"])
    out.append("")
    counts = report["counts"]["by_severity"]
    if counts:
        out.append("| Severity | Events |")
        out.append("|---|---|")
        for sev in ("critical", "high", "medium", "low", "info"):
            if counts.get(sev):
                out.append(f"| {_SEV_MARK[sev]} {sev} | {counts[sev]} |")
        out.append("")

    out.append("## Incident timeline")
    out.append("")
    if not report["incidents"]:
        out.append("_No incidents raised._")
    else:
        out.append("| Time in video | Severity | Incident | × | Score | Evidence | Detail |")
        out.append("|---|---|---|---|---|---|---|")
        for i in sorted(report["incidents"], key=lambda x: x["first_time_s"]):
            span = i["first_timecode"]
            if i["count"] > 1 and i["last_timecode"] != i["first_timecode"]:
                span += f" → {i['last_timecode']}"
            detail = i["detail"].replace("|", "\\|")
            out.append(
                f"| `{span}` | {_SEV_MARK.get(i['severity'], '')} {i['severity']} | "
                f"{i['label']} | {i['count']} | {i['peak_score']} | {i['evidence_tier']} | {detail} |"
            )
    out.append("")

    if report["suspects"]:
        out.append("## Persons of interest")
        out.append("")
        out.append("_Appearance-based grouping only. Not an identification of anyone. "
                   "'Present' means in frame when a signal fired — not participation._")
        out.append("")
        for p in report["suspects"]:
            times = ", ".join(f"`{t}`" for t in p["appearances"][:10]) or "—"
            roles = f"{p['involved_count']} involved, {p['present_only_count']} present"
            out.append(f"- **{p['description']}** (`{p['entity_id']}`) — "
                       f"{roles}; seen at {times}")
        out.append("")

    if report["vehicles"]:
        out.append("## Vehicles")
        out.append("")
        for v in report["vehicles"]:
            flag = ""
            if v["involved_in_collision"]:
                flag = " — **involved in a collision**"
            if v["left_scene"]:
                flag += " — **left the scene afterwards**"
            times = ", ".join(f"`{t}`" for t in v["appearances"][:10]) or "—"
            plate = f" · plate `{v['plate']}`" if v.get("plate") else ""
            out.append(f"- **{v['description']}** (`{v['entity_id']}`){plate}{flag}")
            out.append(f"  - seen at {times}")
            for c in v["collision_events"]:
                out.append(f"  - `{c['timecode']}` {c['type']} ({c['role']})")
        out.append("")

    if report["repeat_subjects"]:
        out.append("## Repeat subjects")
        out.append("")
        out.append("_Matched to a subject recorded in an earlier incident. "
                   "Verify the match before relying on it._")
        out.append("")
        for r in report["repeat_subjects"]:
            out.append(f"- **{r['label']}** (`{r['entity_id']}`) — "
                       f"{r['incident_count']} incidents:")
            for i in r["incidents"]:
                out.append(f"  - `{i['timecode']}` {i['type']} "
                           f"({i.get('source') or 'unknown source'})")
        out.append("")

    if report["observations"]:
        out.append("## Scene descriptions")
        out.append("")
        out.append("_Captioning-model output, sampled through the clip. Unverified._")
        out.append("")
        for o in report["observations"][:20]:
            out.append(f"- `{o['timecode']}` — {o['text']}")
        out.append("")

    if report["recommended_actions"]:
        out.append("## Recommended actions")
        out.append("")
        for a in report["recommended_actions"]:
            out.append(f"{_SEV_MARK.get(a['priority'], '')} **{a['priority']}** — {a['action']}")
        out.append("")

    narrative = report.get("narrative")
    if narrative and narrative.get("text"):
        out.append("## Narrative")
        out.append("")
        out.append(narrative["text"])
        out.append("")
        out.append(f"_Written by {narrative['model']}; the structured sections above "
                   f"are the authoritative record._")
        out.append("")

    out.append("## Confidence and caveats")
    out.append("")
    for c in report["caveats"]:
        out.append(f"- {c}")
    out.append("")
    return "\n".join(out)
