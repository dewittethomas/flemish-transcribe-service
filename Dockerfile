# Install NVIDIA Container Toolkit for LXC's,
# in case of problems, recommend to edit following file:
# /etc/nvidia-container-runtime/config.toml
# and add this line: no-cgroups = false

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ffmpeg

WORKDIR /app

RUN git clone https://github.com/dewittethomas/flemish-transcribe-service.git

WORKDIR /app/flemish-transcribe-service

RUN pip install --no-cache-dir -r requirements.txt

RUN pip cache purge

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]