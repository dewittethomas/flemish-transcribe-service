import subprocess as sp
import ffmpeg
from namedpipe import NPopen
from fastapi import FastAPI, UploadFile
from services.decode_with_vad import transcribe, transcribe_multiple

app = FastAPI()

