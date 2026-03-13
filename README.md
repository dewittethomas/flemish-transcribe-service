# NeLF ASR Transcription Service

## About

This is a FastAPI-based REST API wrapper around the NeLF ASR model for Flemish speech recognition. It exposes the NeLF speech-to-text model via HTTP, accepting video or audio file uploads and returning transcriptions.

Built on top of the original [NeLF ASR codebase](https://huggingface.co/nelfproject/NeLF_S2T_Pytorch).

For more information about the NeLF project or contact details, visit: https://nelfproject.be

---

## Project Structure

```
NeLF/
├── main.py                         # FastAPI entry point
├── config.py                       # Pydantic settings
├── .env                            # Environment-specific values (not committed)
├── requirements.txt
├── api/
│   ├── __init__.py
│   └── routes/
│       ├── __init__.py
│       └── transcribe.py           # POST /transcribe route
├── services/
│   ├── __init__.py
│   ├── model_loader.py             # Model loading and cleanup
│   ├── decode_with_vad.py          # Transcription logic with VAD
│   └── utils/
│       ├── __init__.py
│       ├── espnet_model.py
│       ├── vad_model.py
│       └── postprocessing.py
└── model/                          # Downloaded model files
```

---

## Setup

### 1. Recreate Python environment

Python version: 3.12

```bash
pip install -r requirements.txt
```

### 2. Download models

Download the pretrained NeLF model from HuggingFace:

```
https://huggingface.co/nelfproject/NeLF_S2T_Pytorch
```

Place the downloaded model files in the `model/` directory.

### 3. Configure environment

Copy the example environment file and set your local paths:

```bash
cp .env.example .env
```

Main options in `.env`:

```bash
DEVICE=cpu                      # cuda or cpu
NUM_THREADS=8                   # number of CPU threads for PyTorch
MODEL_DIR=./model               # path to downloaded model
VAD_MODEL_DIR=./model/VAD       # path to VAD model

# Select desired model outputs
ENCODER_OUTPUTS=false
VERBATIM_DECODER_OUTPUTS=false
SUBTITLE_DECODER_OUTPUTS=true
```

### 4. Run the API

```bash
fastapi dev         # development
fastapi run         # production
```

The API will be available at `http://localhost:8000`. Interactive documentation is available at `http://localhost:8000/docs`.

---

## API Usage

### `POST /transcribe`

Accepts one or more video or audio files and returns transcriptions.

**Request:** `multipart/form-data` with one or more files

**Example using curl:**

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "files=@video1.mp4" \
  -F "files=@video2.mp4"
```

**Response:**

```json
{
  "video1.mp4": {
    "subtitle": ["[00:00:01 - 00:00:05] Transcribed text here..."]
  },
  "video2.mp4": {
    "subtitle": ["[00:00:01 - 00:00:03] More transcribed text..."]
  }
}
```

---

## License

This codebase and the related models are provided under a Creative Commons Non-Commercial license.

---

## Research Paper

If you use this code and models, please consider citing the original research paper:

```bibtex
@article{poncelet2024,
    author = "Poncelet, Jakob and Van hamme, Hugo",
    title = "Leveraging Broadcast Media Subtitle Transcripts for Automatic Speech Recognition and Subtitling",
    year={2024},
    journal={arXiv preprint arXiv:2502.03212},
    url = {https://arxiv.org/abs/2502.03212}
}
```