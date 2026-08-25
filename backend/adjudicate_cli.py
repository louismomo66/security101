"""CLI for VLM adjudication — measure it against footage you have labelled.

Single window (the case you care about):

    python -m backend.adjudicate_cli \
        --video "videos/clip.mp4" --start 78 --end 82

Sweep a whole clip in overlapping windows, and score against known incidents:

    python -m backend.adjudicate_cli \
        --video "videos/clip.mp4" --sweep --window 3 --step 1.5 \
        --truth 79.0 --tolerance 2.0

`--truth` accepts one or more timestamps where an incident actually occurs.
The sweep then reports hits, misses and false positives, which is the only way
to know whether the adjudicator is worth its runtime cost.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.adjudicate import (  # noqa: E402
    EVENT_TYPES, QUESTIONS, adjudicate_video,
)


_DEFAULT_MODEL_PATH = os.environ.get(
    "VLM_MODEL_PATH",
    str(Path(__file__).resolve().parent.parent
        / "checkpoints" / "llava-fastvithd_0.5b_stage3"),
)


def get_ask(model_path: str | None):
    """Return the VLM callable, loading the model *now* so failure is loud.

    Nothing else in this process calls `configure_vlm` — that happens in
    server.py's startup hook, which the CLI never runs. An unconfigured VLM
    answers every question with `{"text": "", "error": ...}`, every answer
    parses as unusable, and the run ends with a null verdict that looks like
    "the VLM saw nothing" rather than "the VLM was never loaded".
    """
    path = model_path or _DEFAULT_MODEL_PATH
    if not Path(path).exists():
        raise SystemExit(f"VLM checkpoint not found: {path}")
    import backend.models as M
    M.configure_vlm(path)
    if not M.init_vlm():
        raise SystemExit(f"could not load VLM from {path}")
    return M.run_vlm


def fmt(t: float) -> str:
    m, s = divmod(max(t, 0), 60)
    return f"{int(m):02d}:{s:05.2f}"


def main() -> int:
    p = argparse.ArgumentParser(description="Adjudicate video windows with the VLM")
    p.add_argument("--video", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=0.0)
    p.add_argument("--sweep", action="store_true", help="scan the whole clip")
    p.add_argument("--window", type=float, default=3.0, help="sweep window seconds")
    p.add_argument("--step", type=float, default=1.5, help="sweep step seconds")
    p.add_argument("--frames", type=int, default=5, help="frames sampled per window")
    p.add_argument("--model-path", default=None,
                   help="VLM checkpoint (defaults to the configured one)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="minimum verdict score to report")
    p.add_argument("--truth", type=float, nargs="*", default=[],
                   help="timestamps where an incident actually occurs")
    p.add_argument("--tolerance", type=float, default=2.0,
                   help="seconds a detection may be off and still count as a hit")
    p.add_argument("--json", default="", help="write full results here")
    p.add_argument("--no-verify", action="store_true",
                   help="skip opposite-polarity control questions (halves the "
                        "VLM calls; restores the older credulous scoring)")
    args = p.parse_args()

    ask = get_ask(args.model_path)
    opts = {"verify_polarity": not args.no_verify}
    results: list[dict] = []

    if args.sweep:
        import cv2
        cap = cv2.VideoCapture(args.video)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        dur = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
        cap.release()
        print(f"Sweeping {dur:.1f}s in {args.window}s windows every {args.step}s "
              f"({args.frames} frames each, {len(QUESTIONS)} questions per frame)")
        print(f"That is ~{int(dur / args.step) * args.frames * len(QUESTIONS)} "
              f"VLM calls — this is not cheap.\n")

        t = 0.0
        while t + args.window <= dur:
            r = adjudicate_video(args.video, t, t + args.window, ask,
                                 max_frames=args.frames, **opts)
            r["t"] = t
            results.append(r)
            v, s = r.get("verdict"), r.get("score", 0)
            if v and s >= args.threshold:
                print(f"  {fmt(t)}-{fmt(t + args.window)}  {v:<13} {s:.2f}")
            t += args.step
    else:
        end = args.end or (args.start + args.window)
        r = adjudicate_video(args.video, args.start, end, ask,
                             max_frames=args.frames, **opts)
        r["t"] = args.start
        results.append(r)
        if "error" in r:
            print("error:", r["error"])
            return 1
        print(f"Window {fmt(args.start)}-{fmt(end)}  "
              f"({r['window']['frames_decoded']} frames, "
              f"{r['frames_used']} sampled)\n")
        gated = set(r.get("gated_questions", []))
        print("observations:")
        for q in QUESTIONS:
            v = r["observations"].get(q.key)
            mark = "yes" if v == 1.0 else "no " if v == 0.0 else "???"
            note = "  (gated: no second person confirmed)" if q.key in gated else ""
            print(f"   {mark}  {q.key}{note}")
        if r["unusable_answers"]:
            inc = r.get("inconsistent_answers", 0)
            print(f"\n   {r['unusable_answers']}/{r['questions_asked']} answers "
                  f"unusable — of those {inc} were self-contradictory")
            print("   (the model affirmed both a claim and its negation)")
        print("\nscores:")
        for e in EVENT_TYPES:
            print(f"   {e:<14} {r['scores'][e]:.3f}")
        if r.get("insufficient_evidence"):
            print("\nverdict: none — insufficient evidence "
                  "(too few answers survived verification)")
        else:
            print(f"\nverdict: {r.get('verdict') or 'none'}  "
                  f"({r.get('score', 0):.3f})")

    # Scoring against ground truth
    if args.truth:
        hits, fps_, matched = [], [], set()
        for r in results:
            if not r.get("verdict") or r.get("score", 0) < args.threshold:
                continue
            centre = r["t"] + args.window / 2
            near = [g for g in args.truth if abs(centre - g) <= args.tolerance]
            if near:
                hits.append((r["t"], r["verdict"], r["score"]))
                matched.update(near)
            else:
                fps_.append((r["t"], r["verdict"], r["score"]))
        missed = [g for g in args.truth if g not in matched]
        print("\n── scored against ground truth ──")
        print(f"  incidents      : {len(args.truth)}")
        print(f"  detected       : {len(matched)}")
        print(f"  missed         : {len(missed)}  {[fmt(m) for m in missed]}")
        print(f"  false positives: {len(fps_)}")
        for t, v, s in fps_[:10]:
            print(f"      {fmt(t)}  {v} {s:.2f}")
        if len(matched) + len(fps_):
            prec = len(matched) / (len(matched) + len(fps_))
            rec = len(matched) / max(len(args.truth), 1)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            print(f"  precision {prec:.2f}  recall {rec:.2f}  F1 {f1:.2f}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
