"""
Legacy entrypoint (kept for convenience).

Prefer the real CLI:

  audio-transcribe <audio> --help

or:

  python -m audio_transcription <audio> --help
"""

from audio_transcription.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

