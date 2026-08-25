"""Annotation schema: labelled time spans over video files.

CSV format (header required), one row per labelled span:

    video,start,end,label,split,notes
    videos/kampala_street.mp4,00:01:12,00:01:16,phone_snatch,train,lady in blue
    videos/kampala_street.mp4,00:00:00,00:01:10,normal,train,
    videos/market.mp4,142.5,148.0,pickpocket,val,

`start` / `end` accept either seconds (float) or HH:MM:SS[.ms] / MM:SS.
`split` is one of train / val / test. Blank defaults to train.
`notes` is free text, ignored by the pipeline.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from .labels import CRIME_ACTIONS, resolve

VALID_SPLITS = {"train", "val", "test"}

_TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")


def parse_time(value: str) -> float:
    """Parse seconds, MM:SS, or HH:MM:SS[.ms] into float seconds."""
    value = str(value).strip()
    if not value:
        raise ValueError("empty timestamp")
    m = _TIME_RE.match(value)
    if m:
        hours = float(m.group(1) or 0)
        return hours * 3600 + float(m.group(2)) * 60 + float(m.group(3))
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Cannot parse timestamp {value!r}") from exc


def format_time(seconds: float) -> str:
    h, rem = divmod(max(seconds, 0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


@dataclass
class Span:
    """One labelled time range within one video."""

    video: Path
    start: float
    end: float
    label: str
    split: str = "train"
    notes: str = ""

    @property
    def label_index(self) -> int:
        return resolve(self.label)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class AnnotationSet:
    spans: list[Span] = field(default_factory=list)

    def by_split(self, split: str) -> list[Span]:
        return [s for s in self.spans if s.split == split]

    def class_counts(self) -> dict[str, int]:
        counts = {c: 0 for c in CRIME_ACTIONS}
        for s in self.spans:
            counts[s.label] = counts.get(s.label, 0) + 1
        return counts

    def summary(self) -> str:
        lines = [f"{len(self.spans)} spans across "
                 f"{len({s.video for s in self.spans})} videos"]
        for split in ("train", "val", "test"):
            sub = self.by_split(split)
            if sub:
                secs = sum(s.duration for s in sub)
                lines.append(f"  {split:5s}: {len(sub):4d} spans, {secs / 60:.1f} min")
        lines.append("  class distribution:")
        for name, n in self.class_counts().items():
            if n:
                lines.append(f"    {name:16s} {n:4d}")
        return "\n".join(lines)


def load(csv_path: str | Path, root: str | Path | None = None) -> AnnotationSet:
    """Read an annotation CSV, validating every row.

    Validation is strict and fails loudly: a typo'd label or a reversed time
    range that slips through silently costs a whole training run.
    """
    csv_path = Path(csv_path)
    root = Path(root) if root else csv_path.parent
    spans: list[Span] = []
    problems: list[str] = []

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"video", "start", "end", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{csv_path}: missing required column(s): {', '.join(sorted(missing))}"
            )

        for lineno, row in enumerate(reader, start=2):
            if not (row.get("video") or "").strip():
                continue  # blank line
            try:
                video = Path(row["video"].strip())
                if not video.is_absolute():
                    video = root / video

                start = parse_time(row["start"])
                end = parse_time(row["end"])
                if end <= start:
                    raise ValueError(f"end ({end}) must be after start ({start})")

                label = row["label"].strip().lower()
                if not label:
                    raise ValueError(
                        "blank label — this is an unlabelled candidate row from "
                        "scan_candidates. Watch the span and fill in the label, "
                        "or delete the row."
                    )
                resolve(label)  # raises on unknown

                split = (row.get("split") or "train").strip().lower() or "train"
                if split not in VALID_SPLITS:
                    raise ValueError(
                        f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}"
                    )

                spans.append(Span(video, start, end, label, split,
                                  (row.get("notes") or "").strip()))
            except Exception as exc:
                problems.append(f"  line {lineno}: {exc}")

    if problems:
        raise ValueError(
            f"{csv_path}: {len(problems)} invalid row(s):\n" + "\n".join(problems)
        )
    return AnnotationSet(spans)


def write_template(path: str | Path) -> Path:
    """Write a starter CSV with the header and a couple of example rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "start", "end", "label", "split", "notes"])
        w.writerow(["videos/example.mp4", "00:01:12", "00:01:16",
                    "phone_snatch", "train", "describe the incident"])
        w.writerow(["videos/example.mp4", "00:00:00", "00:01:10",
                    "normal", "train", "baseline street activity"])
    return path
