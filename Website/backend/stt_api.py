# backend/stt_api.py
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from dotenv import load_dotenv

from security import require_auth, require_ip_allowlist

# Local dev env load
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(env_path)

router = APIRouter(tags=["stt"])

# STT settings
STT_MODEL = os.getenv("STT_MODEL", "base")  # tiny/base/small/medium/large-v3
STT_DEVICE = os.getenv("STT_DEVICE", "auto")  # auto/cpu/cuda
STT_COMPUTE = os.getenv("STT_COMPUTE_TYPE", "int8")  # int8/float16/float32
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")  # blank = auto detect

_whisper_model = None

def _get_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError(
            "Missing faster-whisper. Install: pip install faster-whisper"
        ) from e

    device = STT_DEVICE
    if device == "auto":
        # Let faster-whisper decide; you can override with env
        device = "cuda" if os.getenv("USE_CUDA", "0") == "1" else "cpu"

    _whisper_model = WhisperModel(STT_MODEL, device=device, compute_type=STT_COMPUTE)
    return _whisper_model


@router.post("/api/stt/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...)):
    require_ip_allowlist(request)
    require_auth(request)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    # Save to temp file; faster-whisper expects a filename
    suffix = Path(file.filename or "").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        model = _get_model()

        segments, info = model.transcribe(
            tmp_path,
            language=(STT_LANGUAGE or None),
            vad_filter=True,
        )

        text_out = "".join([seg.text for seg in segments]).strip()

        return {
            "text": text_out,
            "language": getattr(info, "language", None),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass