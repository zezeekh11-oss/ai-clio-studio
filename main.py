from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
STORAGE = ROOT / "storage"
UPLOADS = STORAGE / "uploads"
OUTPUTS = STORAGE / "outputs"
TEMP = STORAGE / "temp"
for directory in (UPLOADS, OUTPUTS, TEMP):
    directory.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 1_000_000_000  # 1 GB local MVP limit
ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
JOBS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()

app = FastAPI(title="AI Clip Studio", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def run_command(args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    """Run a program without invoking a shell; user input is never interpolated into a shell."""
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=True)


def probe_video(path: Path) -> dict[str, Any]:
    try:
        result = run_command([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,width,height,codec_name",
            "-of", "json", str(path)
        ], timeout=60)
        data = json.loads(result.stdout)
    except FileNotFoundError as exc:
        raise RuntimeError("FFprobe belum terpasang atau tidak ada di PATH.") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise ValueError("File tidak terbaca sebagai video yang valid.") from exc
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video:
        raise ValueError("File tidak memiliki video stream.")
    duration = float(data.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise ValueError("Durasi video tidak valid.")
    return {
        "duration": duration,
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "video_codec": video.get("codec_name", "unknown"),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def transcribe(path: Path) -> list[dict[str, Any]]:
    """Load faster-whisper lazily so upload/validation works before model setup."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper belum terpasang. Jalankan pip install -r requirements.txt.") from exc
    model_name = os.getenv("WHISPER_MODEL", "small")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(path), vad_filter=True, word_timestamps=True)
    output: list[dict[str, Any]] = []
    for segment in segments:
        words = []
        for word in (segment.words or []):
            words.append({"start": float(word.start), "end": float(word.end), "text": word.word})
        output.append({
            "start": float(segment.start), "end": float(segment.end),
            "text": segment.text.strip(), "words": words,
        })
    return output


HIGHLIGHT_TERMS = {
    "id": ["ternyata", "rahasia", "mengejutkan", "kesalahan", "penting", "akhirnya", "jangan", "kenapa", "bagaimana", "terbaik"],
    "en": ["secret", "surprising", "mistake", "important", "finally", "never", "why", "how", "best", "imagine"],
}


def detect_highlights(transcript: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    candidates = []
    for segment in transcript:
        text = segment["text"].strip()
        if not text:
            continue
        lower = text.lower()
        score = 1.0 + min(len(text) / 120, 1.5)
        score += sum(0.6 for term in HIGHLIGHT_TERMS["id"] + HIGHLIGHT_TERMS["en"] if term in lower)
        if "?" in text or "!" in text:
            score += 0.4
        start = max(0.0, float(segment["start"]) - 2.0)
        end = min(duration, float(segment["end"]) + 3.0)
        if end - start < 8:
            end = min(duration, start + 8)
        if end - start > 60:
            end = start + 60
        candidates.append({
            "start": round(start, 2), "end": round(end, 2),
            "title": (text[:64] + "…") if len(text) > 64 else text,
            "reason": "Skor heuristik dari panjang ujaran, kata pemicu, dan tanda baca; perlu tinjauan manusia.",
            "score": round(score, 2), "transcript_excerpt": text,
        })
    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected = []
    for item in candidates:
        if all(item["end"] <= old["start"] or item["start"] >= old["end"] for old in selected):
            selected.append(item)
        if len(selected) >= 5:
            break
    return sorted(selected, key=lambda x: x["start"])


def render_clip(source: Path, clip: dict[str, Any], output: Path, subtitles: list[dict[str, Any]]) -> None:
    start, end = float(clip["start"]), float(clip["end"])
    if start < 0 or end <= start:
        raise ValueError("Batas clip tidak valid.")
    # Scale to fill 1080x1920 then crop centrally; never stretch the source.
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1"
    try:
        run_command([
            "ffmpeg", "-y", "-ss", str(start), "-i", str(source), "-t", str(end - start),
            "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(output)
        ], timeout=3600)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg belum terpasang atau tidak ada di PATH.") from exc


def update_job(job_id: str, **values: Any) -> None:
    with LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(values)


def process_job(job_id: str, source: Path) -> None:
    try:
        update_job(job_id, status="processing", stage="Membaca metadata", progress=10)
        metadata = probe_video(source)
        update_job(job_id, metadata=metadata, stage="Transkripsi faster-whisper", progress=20)
        transcript = transcribe(source) if metadata["has_audio"] else []
        update_job(job_id, transcript=transcript, stage="Mendeteksi highlight", progress=55)
        highlights = detect_highlights(transcript, metadata["duration"])
        if not highlights:
            # Honest fallback: no speech candidate found; do not fabricate AI highlights.
            update_job(job_id, highlights=[], stage="Tidak ada highlight otomatis", progress=65)
        rendered = []
        for index, clip in enumerate(highlights, start=1):
            output = OUTPUTS / f"{job_id}-clip-{index}.mp4"
            update_job(job_id, stage=f"Rendering clip {index}/{len(highlights)}", progress=65 + int(index / max(len(highlights), 1) * 30))
            render_clip(source, clip, output, transcript)
            # Verify output is a readable video before publishing it.
            result = probe_video(output)
            if result["width"] != 1080 or result["height"] != 1920:
                raise RuntimeError("Quality check gagal: resolusi output bukan 1080x1920.")
            rendered.append({"title": clip["title"], "start": clip["start"], "end": clip["end"], "url": f"/download/{output.name}"})
        update_job(job_id, status="completed", stage="Selesai", progress=100, clips=rendered,
                   note="Highlight memakai skor heuristik awal dan perlu ditinjau. Subtitle otomatis serta editor timeline lanjutan belum termasuk MVP awal.")
    except Exception as exc:  # keep job state visible; details go to local console only
        update_job(job_id, status="failed", stage="Gagal", error=str(exc), progress=0)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": "AI Clip Studio", "version": "0.1.0"}


@app.post("/api/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, "Format belum didukung. Gunakan MP4, MOV, MKV, WEBM, atau M4V.")
    job_id = uuid.uuid4().hex[:12]
    safe_path = UPLOADS / f"{job_id}{suffix}"
    size = 0
    try:
        with safe_path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Ukuran video melebihi batas MVP 1 GB.")
                target.write(chunk)
    except Exception:
        safe_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    try:
        metadata = probe_video(safe_path)
    except (ValueError, RuntimeError) as exc:
        safe_path.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    with LOCK:
        JOBS[job_id] = {
            "id": job_id, "status": "queued", "stage": "Menunggu pemrosesan", "progress": 0,
            "filename": file.filename, "metadata": metadata, "created_at": datetime.now(timezone.utc).isoformat(),
            "transcript": [], "highlights": [], "clips": [],
        }
    background_tasks.add_task(process_job, job_id, safe_path)
    return {"job_id": job_id, "status": "queued", "metadata": metadata}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Proyek tidak ditemukan atau server baru saja dimulai ulang.")
        return job


@app.post("/api/import-url")
def import_url(payload: dict[str, str]) -> dict[str, str]:
    url = (payload.get("url") or "").strip()
    if not re.match(r"^https://(www\.)?(youtube\.com|youtu\.be)/", url, re.I):
        raise HTTPException(400, "Masukkan URL YouTube HTTPS yang valid.")
    return {"status": "not_enabled", "message": "Import URL belum diaktifkan pada build awal. Upload video yang kamu punya hak untuk gunakan; integrasi URL akan ditambahkan setelah pipeline upload stabil."}


@app.get("/download/{filename}")
def download(filename: str) -> FileResponse:
    if not re.fullmatch(r"[a-f0-9]{12}-clip-[1-9][0-9]*\.mp4", filename):
        raise HTTPException(400, "Nama file tidak valid.")
    path = OUTPUTS / filename
    if not path.is_file():
        raise HTTPException(404, "File hasil tidak ditemukan.")
    return FileResponse(path, media_type="video/mp4", filename=filename)
