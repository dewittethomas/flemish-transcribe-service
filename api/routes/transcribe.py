import io
import subprocess
from typing import List
from fastapi import APIRouter, UploadFile
from services import transcribe_multiple

router = APIRouter()

@router.post("/transcribe")
async def upload_files(files: List[UploadFile]):
    audio_buffers = []

    for file in files:
        content = await file.read()
        audio_result = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-vn", "-acodec", "pcm_s16le", "-f", "wav", "pipe:1"],
            input=content,
            capture_output=True
        )
        audio_buffer = io.BytesIO(audio_result.stdout)
        audio_buffer.name = file.filename
        audio_buffers.append(audio_buffer)

    return transcribe_multiple(audio_buffers)