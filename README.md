# AI Clip Studio — Local MVP

A local-first video clipping prototype. It currently supports direct video upload, FFprobe validation, faster-whisper transcription, heuristic highlight candidates, vertical FFmpeg rendering, and MP4 downloads. URL import is deliberately a visible placeholder until the upload pipeline is stable and a compliant source-acquisition method is selected.

## Requirements
- Python 3.10+
- FFmpeg and FFprobe available on PATH
- Recommended: 8 GB RAM or more for the `small` Whisper model; CPU-only operation is supported but may be slow

## Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The first transcription run downloads the selected faster-whisper model (`small` by default). This is open-source software, but requires an internet connection for the initial model download and uses local compute/storage. Set `WHISPER_MODEL=base` for a lighter model or `WHISPER_MODEL=medium` for higher accuracy with greater resource use.

## Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000.

## Current limitations
- Job state is held in memory; restarting the server loses job history.
- Highlight detection is a transparent heuristic baseline, not a trained engagement predictor.
- The initial renderer uses center cropping to 9:16; subject-aware reframing is not implemented.
- Subtitle generation/rendering and an interactive timeline are not yet implemented.
- Reference-video style analysis is not yet implemented.
- YouTube URL import is not enabled. Use files you own or are authorized to process.
- No deployment is configured; this is a local MVP.
- Output contains no application-added watermark.

## Troubleshooting
- `FFmpeg/FFprobe not found`: install FFmpeg and ensure both executables are on PATH.
- `faster-whisper not installed`: activate the virtual environment and run `pip install -r requirements.txt`.
- Slow first run: model weights may be downloading. CPU inference can take longer than the source video.
- Upload rejected: use a supported video format and keep the file under 1 GB.

## Tests
Run `python -m unittest discover -s tests -v` for lightweight unit tests. End-to-end processing additionally requires FFmpeg, FFprobe, faster-whisper, and a suitable sample video.
