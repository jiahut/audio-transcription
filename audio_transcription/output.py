from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def _format_srt_time(seconds: float) -> str:
    td = timedelta(seconds=max(0.0, float(seconds)))
    total_ms = int(td.total_seconds() * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _format_vtt_time(seconds: float) -> str:
    return _format_srt_time(seconds).replace(",", ".")


@dataclass(frozen=True)
class TextLine:
    start: float | None
    end: float | None
    speaker: str | None
    text: str


def iter_lines_from_result(result: dict[str, Any]) -> Iterable[TextLine]:
    for seg in result.get("segments", []) or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        yield TextLine(
            start=seg.get("start"),
            end=seg.get("end"),
            speaker=seg.get("speaker"),
            text=text,
        )


def write_txt(path: Path, lines: Iterable[TextLine], with_timestamps: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            speaker = line.speaker or "UNKNOWN"
            if with_timestamps and line.start is not None and line.end is not None:
                f.write(f"[{line.start:.2f}s - {line.end:.2f}s] {speaker}: {line.text}\n")
            else:
                f.write(f"{speaker}: {line.text}\n")


def write_srt(path: Path, lines: Iterable[TextLine]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, line in enumerate(lines, start=1):
            start = 0.0 if line.start is None else float(line.start)
            end = start if line.end is None else float(line.end)
            speaker = line.speaker or "UNKNOWN"
            f.write(f"{idx}\n")
            f.write(f"{_format_srt_time(start)} --> {_format_srt_time(end)}\n")
            f.write(f"{speaker}: {line.text}\n\n")


def write_vtt(path: Path, lines: Iterable[TextLine]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for line in lines:
            start = 0.0 if line.start is None else float(line.start)
            end = start if line.end is None else float(line.end)
            speaker = line.speaker or "UNKNOWN"
            f.write(f"{_format_vtt_time(start)} --> {_format_vtt_time(end)}\n")
            f.write(f"{speaker}: {line.text}\n\n")

