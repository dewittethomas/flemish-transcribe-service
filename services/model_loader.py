from config import settings, decode_conf
from services.utils.espnet_model import load_nelf_model
from services.utils.vad_model import load_vad
import torch

# Global model variables
model = None
vad_model = None

def load_models():
    global model, vad_model

    print('# Loading Speech-to-Text Model')
    model, args = load_nelf_model(
        model_dir=settings.model_dir,
        ctc=settings.encoder_outputs,
        subtitle_ctc=settings.encoder_outputs,
        verbatim=settings.verbatim_decoder_outputs,
        subtitle=settings.subtitle_decoder_outputs,
        device=settings.device
    )

    # Initialize beam search for decoding with decoders
    if settings.verbatim_decoder_outputs or settings.subtitle_decoder_outputs:
        print('# Preparing decoders')
        model.prepare_for_beam_search(**decode_conf)

    # Set inference mode
    model.eval()
    torch.set_grad_enabled(False)

    print('# Loading VAD model')
    vad_model = load_vad(
        settings.vad_model_dir,
        onnx=settings.vad_use_onnx,
        device=settings.device
    )

    print('# Models loaded successfully')

def cleanup_models():
    global model, vad_model

    print('# Cleaning up models')

    if model is not None:
        del model
        model = None

    if vad_model is not None:
        del vad_model
        vad_model = None

    # Clear PyTorch CUDA cache if using GPU
    if settings.device == "cuda":
        torch.cuda.empty_cache()
        print('# Cleared CUDA cache')

    print('# Cleanup complete')