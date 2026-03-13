from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Device
    device: str = "cpu"
    num_threads: int = 8

    # Model directories
    model_dir: str = "model"
    vad_model_dir: str = "model/VAD"
    vad_use_onnx: bool = False

    # Model outputs
    encoder_outputs: bool = False
    verbatim_decoder_outputs: bool = False
    subtitle_decoder_outputs: bool = True

    # Decoder beam search
    nbest: int = 1
    beam_size: int = 20
    minlenratio: float = 0.2
    subtitle_length_penalty: float = 0.1
    normalize_length_subtitle: bool = True

    # Decoding
    batch_decode: bool = False

    # VAD settings
    max_segment_length: float = 15.0
    min_segment_length: float = 3.0
    max_pause: float = 2.0
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 50
    speech_pad_ms: int = 60
    vad_threshold: float = 0.5

    # Output
    output_add_timing: bool = True
    verbose_output: bool = True

    class Config:
        env_file = ".env"

settings = Settings()

# Derived config that needs to stay as a dict
decode_conf = {
    "nbest": settings.nbest,
    "beam_size": settings.beam_size,
    "minlenratio": settings.minlenratio,
    "subtitle_length_penalty": settings.subtitle_length_penalty,
    "normalize_length_subtitle": settings.normalize_length_subtitle,
}