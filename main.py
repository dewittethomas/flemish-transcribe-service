import logging
import warnings
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routes.transcribe import router
from config import settings
from services.model_loader import load_models, cleanup_models

# Suppress unnecessary logging
logging.getLogger().setLevel(logging.CRITICAL)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set the number of threads
torch.set_num_threads(settings.num_threads)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    cleanup_models()

app = FastAPI(lifespan=lifespan)
app.include_router(router)

@app.get("/")
async def root():
    return {"status": "OK"}