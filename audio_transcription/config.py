from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Device = Literal["auto", "cuda", "cpu"]
VadMethod = Literal["silero", "pyannote"]
OutputFormat = Literal["json", "txt", "srt", "vtt"]


@dataclass(frozen=True)
class AppConfig:
    # Core
    device: Device = "auto"
    device_index: int = 0
    model: str = "large-v3"
    compute_type: str = "float16"
    batch_size: int = 16
    language: str | None = None  # None => auto-detect
    task: Literal["transcribe", "translate"] = "transcribe"
    threads: int = 4
    download_root: str | None = None
    local_files_only: bool = False

    # VAD (used by WhisperX pipeline)
    vad_method: VadMethod = "silero"
    vad_options: dict[str, Any] = field(default_factory=dict)

    # Alignment
    align: bool = True
    align_model: str | None = None
    interpolate_method: str = "nearest"

    # Diarization
    diarize: bool = True
    diarize_model: str | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None

    # Auth
    hf_token: str | None = None

    # Output
    output_dir: Path = Path("./outputs")
    formats: tuple[OutputFormat, ...] = ("json", "txt")
    print_stdout: bool = True

    # Advanced passthroughs
    asr_options: dict[str, Any] = field(default_factory=dict)


def normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    lang = language.strip()
    if not lang or lang.lower() == "auto":
        return None
    return lang


def coerce_path(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    return Path(str(value))


def config_from_dict(data: dict[str, Any]) -> AppConfig:
    data = dict(data)
    if "output_dir" in data:
        data["output_dir"] = coerce_path(data["output_dir"])
    if "formats" in data:
        formats = data["formats"]
        if isinstance(formats, str):
            formats = [f.strip() for f in formats.split(",") if f.strip()]
        data["formats"] = tuple(formats)
    if "language" in data:
        data["language"] = normalize_language(data["language"])
    return AppConfig(**data)


def config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    d = dict(cfg.__dict__)
    d["output_dir"] = str(cfg.output_dir)
    d["formats"] = list(cfg.formats)
    d["language"] = cfg.language or "auto"
    return d

