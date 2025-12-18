from __future__ import annotations

import gc
import os
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
        import collections
        import typing

        from omegaconf import DictConfig, ListConfig
        from omegaconf import nodes as omegaconf_nodes
        from omegaconf.base import ContainerMetadata, Metadata
        from pyannote.audio.core.model import Introspection

        torch.serialization.add_safe_globals(
            [
                DictConfig,
                ListConfig,
                ContainerMetadata,
                Metadata,
                typing.Any,
                type(None),
                list,
                dict,
                tuple,
                set,
                collections.defaultdict,
                collections.OrderedDict,
                int,
                float,
                str,
                bool,
                torch.torch_version.TorchVersion,
                Introspection,
                omegaconf_nodes.Node,
                omegaconf_nodes.ValueNode,
                omegaconf_nodes.AnyNode,
                omegaconf_nodes.StringNode,
                omegaconf_nodes.IntegerNode,
                omegaconf_nodes.FloatNode,
                omegaconf_nodes.BooleanNode,
                omegaconf_nodes.BytesNode,
                omegaconf_nodes.EnumNode,
                omegaconf_nodes.PathNode,
                omegaconf_nodes.InterpolationResultNode,
            ]
        )
    except Exception:
        return


def _configure_checkpoint_loading(trust_checkpoints: bool) -> None:
    if not trust_checkpoints:
        _configure_torch_safe_globals()
        return
    try:
        import lightning_fabric.utilities.cloud_io as cloud_io
    except Exception:
        return

    if getattr(cloud_io._load, "_audio_transcription_patched", False):
        return

    original_load = cloud_io._load

    def patched_load(path_or_url, map_location=None, weights_only=None):
        if weights_only is None:
            weights_only = False
        return original_load(path_or_url, map_location=map_location, weights_only=weights_only)

    patched_load._audio_transcription_patched = True  # type: ignore[attr-defined]
    cloud_io._load = patched_load  # type: ignore[assignment]


def _resolve_device(cfg: AppConfig) -> str:
    if cfg.device != "auto":
        return cfg.device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _maybe_configure_certifi_ca_bundle() -> None:
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE"):
        return
    try:
        import certifi
    except Exception:
        return
    ca_bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
    os.environ.setdefault("CURL_CA_BUNDLE", ca_bundle)


def _default_cache_dir() -> Path:
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return base / "audio-transcription"


def _nltk_data_dir(cfg: AppConfig) -> Path:
    if cfg.download_root:
        return Path(cfg.download_root) / "nltk_data"
    return _default_cache_dir() / "nltk_data"


def _ensure_nltk_punkt(cfg: AppConfig) -> None:
    try:
        import nltk
    except Exception:
        return

    data_dir = _nltk_data_dir(cfg)
    data_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("NLTK_DATA", str(data_dir))
    if str(data_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(data_dir))

    def has(resource: str) -> bool:
        try:
            nltk.data.find(resource)
            return True
        except LookupError:
            return False

    missing: list[str] = []
    if not has("tokenizers/punkt/english.pickle"):
        missing.append("punkt")
    if not has("tokenizers/punkt_tab/english/"):
        missing.append("punkt_tab")
    if not missing:
        return

    _maybe_configure_certifi_ca_bundle()

    for pkg in missing:
        try:
            nltk.download(pkg, download_dir=str(data_dir), quiet=True)
        except Exception:
            # We'll re-check below and raise a clear error if still missing.
            pass

    still_missing: list[str] = []
    if not has("tokenizers/punkt/english.pickle"):
        still_missing.append("punkt")
    if not has("tokenizers/punkt_tab/english/"):
        still_missing.append("punkt_tab")
    if still_missing:
        raise RuntimeError(
            "WhisperX alignment requires NLTK tokenizer data but it is missing: "
            f"{', '.join(still_missing)}. "
            f"Expected under NLTK_DATA={data_dir}. "
            "Fix by running `python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab')\"`, "
            "or point NLTK_DATA at a pre-downloaded nltk_data directory. "
            "If you are behind a custom CA/proxy, pass `--ca-bundle /path/to/ca.pem`. "
            "To bypass this step, disable alignment with `--no-align`."
        )


@dataclass(frozen=True)
class RunResult:
    audio_path: str
    started_at: float
    finished_at: float
    device: str
    result: dict[str, Any]
    diarize_segments: Any | None = None


def run_one(audio_path: Path, cfg: AppConfig) -> RunResult:
    _configure_checkpoint_loading(cfg.trust_checkpoints)
    started_at = time.time()

    import whisperx
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    device = _resolve_device(cfg)

    audio = whisperx.load_audio(str(audio_path))

    try:
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
    except Exception as e:
        message = str(e)
        if cfg.vad_method == "silero" and (
            "CERTIFICATE_VERIFY_FAILED" in message
            or "snakers4/silero-vad" in message
            or "torch/hub" in message
            or "torch.hub" in message
        ):
            raise RuntimeError(
                "Silero VAD is loaded via torch.hub and requires GitHub access (or a populated torch hub cache). "
                "Try `--vad-method pyannote` to avoid GitHub downloads, or fix system CA certificates."
            ) from e
        raise
    result: dict[str, Any] = model.transcribe(audio, batch_size=cfg.batch_size)

    if cfg.align:
        _ensure_nltk_punkt(cfg)
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
