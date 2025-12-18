from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig


def _configure_torch_safe_globals() -> None:
    try:
        import torch
    except Exception:
        return

    if not hasattr(torch.serialization, "add_safe_globals"):
        return
    try:
        from omegaconf import DictConfig, ListConfig

        torch.serialization.add_safe_globals([DictConfig, ListConfig])
    except Exception:
        return


def _resolve_device(cfg: AppConfig) -> str:
    if cfg.device != "auto":
        return cfg.device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass(frozen=True)
class RunResult:
    audio_path: str
    started_at: float
    finished_at: float
    device: str
    result: dict[str, Any]
    diarize_segments: Any | None = None


def run_one(audio_path: Path, cfg: AppConfig) -> RunResult:
    _configure_torch_safe_globals()
    started_at = time.time()

    import whisperx
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    device = _resolve_device(cfg)

    audio = whisperx.load_audio(str(audio_path))

    model = whisperx.load_model(
        cfg.model,
        device,
        device_index=cfg.device_index,
        compute_type=cfg.compute_type,
        asr_options=cfg.asr_options or None,
        language=cfg.language,
        vad_method=cfg.vad_method,
        vad_options=cfg.vad_options or None,
        task=cfg.task,
        download_root=cfg.download_root,
        local_files_only=cfg.local_files_only,
        threads=cfg.threads,
    )
    result: dict[str, Any] = model.transcribe(audio, batch_size=cfg.batch_size)

    if cfg.align:
        align_model, metadata = whisperx.load_align_model(
            language_code=result.get("language") or (cfg.language or "en"),
            device=device,
            model_name=cfg.align_model,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            interpolate_method=cfg.interpolate_method,
            return_char_alignments=False,
        )

    diarize_segments = None
    if cfg.diarize:
        diarize_model = DiarizationPipeline(
            model_name=cfg.diarize_model,
            use_auth_token=cfg.hf_token,
            device=device,
        )
        diarized = diarize_model(
            audio,
            num_speakers=cfg.num_speakers,
            min_speakers=cfg.min_speakers,
            max_speakers=cfg.max_speakers,
            return_embeddings=False,
        )
        diarize_segments = diarized
        result = assign_word_speakers(diarize_segments, result)

    finished_at = time.time()

    try:
        del model
    except Exception:
        pass
    gc.collect()
    if device == "cuda":
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    return RunResult(
        audio_path=str(audio_path),
        started_at=started_at,
        finished_at=finished_at,
        device=device,
        result=result,
        diarize_segments=diarize_segments,
    )

