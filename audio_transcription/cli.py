from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from .config import AppConfig, OutputFormat, config_from_dict, config_to_dict, normalize_language
from .io_config import dump_config_file, load_config_file
from .output import iter_lines_from_result, write_json, write_srt, write_txt, write_vtt
from .pipeline import run_one


def _parse_kv(values: list[str] | None) -> dict[str, Any]:
    if not values:
        return {}

    def parse_value(raw: str) -> Any:
        s = raw.strip()
        if not s:
            return ""
        if s.lower() in {"true", "false"}:
            return s.lower() == "true"
        try:
            if "." in s:
                return float(s)
            return int(s)
        except Exception:
            pass
        # Allow JSON literals (objects/arrays/strings/numbers/bools/null)
        try:
            import json

            return json.loads(s)
        except Exception:
            return s

    out: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Invalid KEY=VALUE pair: {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise argparse.ArgumentTypeError(f"Invalid KEY=VALUE pair: {item!r}")
        out[k] = parse_value(v)
    return out


def _parse_formats(items: list[str] | None) -> tuple[OutputFormat, ...] | None:
    if not items:
        return None
    normalized: list[str] = []
    for x in items:
        parts = [p.strip() for p in x.split(",") if p.strip()]
        normalized.extend(parts)
    allowed: set[str] = {"json", "txt", "srt", "vtt"}
    for f in normalized:
        if f not in allowed:
            raise argparse.ArgumentTypeError(f"Unsupported format: {f!r} (choose from {sorted(allowed)})")
    return tuple(normalized)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audio-transcribe", description="WhisperX + Pyannote transcription + diarization")
    p.add_argument("audio", nargs="+", help="Input audio file(s)")
    p.add_argument("--config", type=Path, help="YAML/JSON config file (CLI flags override it)")
    p.add_argument("--dump-config", type=Path, help="Write resolved config to YAML/JSON and exit")
    p.add_argument("--dry-run", action="store_true", help="Validate config and inputs, then exit")

    # Core
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    p.add_argument("--device-index", type=int, default=None)
    p.add_argument("--model", default=None, help="Whisper model name (e.g. large-v3, medium, tiny)")
    p.add_argument("--compute-type", default=None, help="float16/float32/int8 (depends on device)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--language", default=None, help="Language code (e.g. en, zh) or 'auto'")
    p.add_argument("--task", choices=["transcribe", "translate"], default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--download-root", default=None)
    p.add_argument("--local-files-only", action="store_true", default=None)

    # VAD
    p.add_argument("--vad-method", choices=["silero", "pyannote"], default=None)
    p.add_argument("--vad-option", action="append", default=None, metavar="KEY=VALUE", help="Override VAD option")

    # Alignment
    p.add_argument("--align", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--align-model", default=None)
    p.add_argument("--interpolate-method", default=None)

    # Diarization
    p.add_argument("--diarize", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--diarize-model", default=None)
    p.add_argument("--num-speakers", type=int, default=None)
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--hf-token", default=None, help="HuggingFace token (or set HF_TOKEN env var)")

    # ASR extra options
    p.add_argument("--asr-option", action="append", default=None, metavar="KEY=VALUE", help="Pass-through ASR option")

    # Output
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--format", action="append", default=None, help="Output format(s): json,txt,srt,vtt (repeatable)")
    p.add_argument("--print", dest="print_stdout", action=argparse.BooleanOptionalAction, default=None)

    return p


def _resolve_config(args: argparse.Namespace) -> AppConfig:
    base = AppConfig()
    data: dict[str, Any] = config_to_dict(base)

    if args.config:
        file_data = load_config_file(args.config)
        data.update(file_data)

    # CLI overrides (None => not provided)
    overrides: dict[str, Any] = {}
    for key in [
        "device",
        "device_index",
        "model",
        "compute_type",
        "batch_size",
        "task",
        "threads",
        "download_root",
        "align",
        "align_model",
        "interpolate_method",
        "diarize",
        "diarize_model",
        "num_speakers",
        "min_speakers",
        "max_speakers",
        "output_dir",
        "print_stdout",
    ]:
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value

    if args.local_files_only is True:
        overrides["local_files_only"] = True

    if args.language is not None:
        overrides["language"] = normalize_language(args.language)

    if args.vad_method is not None:
        overrides["vad_method"] = args.vad_method
    if args.vad_option is not None:
        overrides["vad_options"] = {**(data.get("vad_options") or {}), **_parse_kv(args.vad_option)}

    if args.asr_option is not None:
        overrides["asr_options"] = {**(data.get("asr_options") or {}), **_parse_kv(args.asr_option)}

    fmt = _parse_formats(args.format)
    if fmt is not None:
        overrides["formats"] = fmt

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        overrides["hf_token"] = token

    data.update(overrides)
    return config_from_dict(data)


def _validate_inputs(audio_paths: list[Path]) -> None:
    missing = [p for p in audio_paths if not p.exists()]
    if missing:
        raise SystemExit(f"Input audio not found: {', '.join(str(p) for p in missing)}")
    not_files = [p for p in audio_paths if not p.is_file()]
    if not_files:
        raise SystemExit(f"Input must be file(s): {', '.join(str(p) for p in not_files)}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    audio_paths = [Path(a) for a in args.audio]
    _validate_inputs(audio_paths)

    cfg = _resolve_config(args)

    if args.dump_config:
        dump_config_file(args.dump_config, config_to_dict(cfg))
        return 0

    if args.dry_run:
        return 0

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for audio_path in audio_paths:
        run = run_one(audio_path, cfg)

        stem = audio_path.stem
        payload = {
            "audio_path": run.audio_path,
            "device": run.device,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "config": config_to_dict(cfg),
            "result": run.result,
        }

        if "json" in cfg.formats:
            write_json(cfg.output_dir / f"{stem}.json", payload)

        lines = list(iter_lines_from_result(run.result))
        if "txt" in cfg.formats:
            write_txt(cfg.output_dir / f"{stem}.txt", lines)
        if "srt" in cfg.formats:
            write_srt(cfg.output_dir / f"{stem}.srt", lines)
        if "vtt" in cfg.formats:
            write_vtt(cfg.output_dir / f"{stem}.vtt", lines)

        if cfg.print_stdout:
            for line in lines:
                speaker = line.speaker or "UNKNOWN"
                if line.start is not None and line.end is not None:
                    print(f"[{line.start:.2f}s - {line.end:.2f}s] {speaker}: {line.text}")
                else:
                    print(f"{speaker}: {line.text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

