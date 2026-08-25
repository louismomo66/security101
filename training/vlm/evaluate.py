"""Score a fine-tuned VLM checkpoint on THIS project's own labelled footage.

The number a fine-tune ships with is accuracy on its own held-out split of
the training distribution — for training/vlm/colab_train_lfm_ucf.ipynb,
that's UCF Crime's own held-out frames. That number does not say anything
about Uganda street footage, resolution, camera angle, or the specific
INCIDENT: vocabulary this project's threat engine actually screens for. So
the only question this script asks is the one that decides whether a
checkpoint is worth wiring into SENTINEL_VLM_BACKEND=lfm:

    On this project's own labelled spans, does the INCIDENT: line say
    something for real incidents and NONE for the confirmed-normal span?

Reuses the exact same regexes backend/threat.py::ingest_caption() screens
with (INCIDENT_LINE_RE, NO_INCIDENT_RE), so a "pass" here means the
production parsing path would actually behave correctly, not just that the
text looks right to a human reading it.

Usage
-----
    python -m training.vlm.evaluate --model checkpoints/lfm_ucf.gguf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.threat import INCIDENT_LINE_RE, NO_INCIDENT_RE  # noqa: E402
from training.annotations import load  # noqa: E402


def sample_frames(video_path: Path, start: float, end: float, every: float = 1.0):
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    t = start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        t += every
        if not ok or frame is None:
            continue
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()


def flagged(text: str) -> bool | None:
    """True = flagged an incident, False = said NONE, None = didn't follow format."""
    m = INCIDENT_LINE_RE.search(text)
    if not m:
        return None
    answer = m.group(1).strip()
    return not bool(NO_INCIDENT_RE.match(answer))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--mmproj", default=None)
    p.add_argument("--annotations", default="training/data/annotations.csv")
    p.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames")
    p.add_argument("--max-frames-per-span", type=int, default=15)
    args = p.parse_args()

    import os
    os.environ["SENTINEL_LFM_MODEL"] = args.model
    if args.mmproj:
        os.environ["SENTINEL_LFM_MMPROJ"] = args.mmproj
    from backend.lfm_vlm import run_vlm_lfm
    from backend.threat import ThreatEngine

    prompt = ThreatEngine().threat_prompt()

    ann = load(ROOT / args.annotations)
    print(f"model: {args.model}")
    print(f"{len(ann.spans)} labelled spans in {args.annotations}\n")

    results = []  # (label, expected_incident, flagged, malformed)
    for span in ann.spans:
        video_path = ROOT / span.video
        if not video_path.exists():
            print(f"  -- missing, skipped: {span.video}")
            continue
        expected_incident = span.label.lower() != "normal"
        n = 0
        hits = misses = malformed = 0
        for frame in sample_frames(video_path, span.start, span.end, args.every):
            if n >= args.max_frames_per_span:
                break
            n += 1
            out = run_vlm_lfm(frame, prompt=prompt, max_tokens=64)
            text = out.get("text", "")
            f = flagged(text)
            if f is None:
                malformed += 1
                continue
            correct = (f == expected_incident)
            hits += int(correct)
            misses += int(not correct)
        total = hits + misses
        acc = 100.0 * hits / total if total else 0.0
        print(f"{span.video.name[:50]:50s} {span.label:15s} "
              f"{hits}/{total} correct ({acc:.0f}%), {malformed} malformed")
        results.append((span.label, expected_incident, hits, total, malformed))

    print("\n-- SUMMARY --")
    normal = [r for r in results if not r[1]]
    incident = [r for r in results if r[1]]
    for name, rows in (("normal spans (want: NONE)", normal),
                       ("incident spans (want: flagged)", incident)):
        h = sum(r[2] for r in rows)
        t = sum(r[3] for r in rows)
        print(f"  {name}: {h}/{t} correct ({100.0*h/t:.0f}%)" if t else f"  {name}: no data")

    print("\nA checkpoint that flags the normal span heavily is not usable "
          "regardless of incident recall — false alarms on ordinary footage "
          "are the failure mode this project has repeatedly measured other "
          "models out on (see training/weapons/evaluate.py's own framing).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
