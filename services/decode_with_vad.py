from services.utils.vad_model import load_audio
from services.utils.postprocessing import format_output, merge_batch_outputs
import services.model_loader as model_loader
import torch
import time
from typing import BinaryIO
from config import settings

# Sampling rate (do not change!)
SR = 16000

# Loop over audio files
def transcribe_multiple(audio_files: list[BinaryIO]) -> dict:
    results = {}

    for audio in audio_files:
        out = transcribe(audio)
        results[audio.name] = out

    return results

# Transcribe audio file
def transcribe(audio: BinaryIO) -> dict:
    model = model_loader.model
    vad_model = model_loader.vad_model

    start_time = time.time()

    # load audio file
    audio_file = load_audio(audio, sampling_rate=SR, device=settings.device)
    #wav, lens = load_wav(os.path.join(wav_dir, wav_file), device=device)
    
    # apply VAD
    speech_timestamps = vad_model.get_speech_timestamps(
        audio_file,
        min_speech_duration_ms=settings.min_speech_duration_ms,
        min_silence_duration_ms=settings.min_silence_duration_ms,
        speech_pad_ms=settings.speech_pad_ms,
        threshold=settings.vad_threshold,
    )

    # merge VAD segments
    speech_segments = vad_model.merge_segments(
        speech_timestamps,
        settings.max_pause,
        settings.max_segment_length,
        settings.min_segment_length,
        mode="hierarchical",
    )

    # decode speech segments to text
    if settings.batch_decode:
        # combine all segments and decode simultaneously
        audio_batch, audio_lens = [], []
        for seg in speech_segments:
            audio_seg = audio_file[int(seg['start']*SR):int(seg['end']*SR)].to(device=settings.device)
            audio_len = torch.tensor([audio_seg.shape[0],], dtype=torch.long, device=settings.device)
            audio_batch.append(audio_seg)
            audio_lens.append(audio_len)
    
        audio_batch = torch.nn.utils.rnn.pad_sequence(audio_batch, batch_first=True)
        audio_lens = torch.cat(audio_lens)
        res = model.decode(audio_batch, audio_lens, ctc=settings.encoder_outputs, verbatim=settings.verbatim_decoder_outputs, subtitle=settings.subtitle_decoder_outputs)
    
        out = format_output(res, speech_segments, add_timing=settings.output_add_timing, batch=True, verbose=settings.verbose_output)
    else:
        # decode segment by segment
        batch_out = []
        for seg in speech_segments:
            audio_seg = audio_file[int(seg['start']*SR):int(seg['end']*SR)].unsqueeze(0).to(device=settings.device)
            audio_len = torch.tensor([audio_seg.shape[1],], dtype=torch.long, device=settings.device)
            res = model.decode(audio_seg, audio_len, ctc=settings.encoder_outputs, verbatim=settings.verbatim_decoder_outputs, subtitle=settings.subtitle_decoder_outputs)
            
            out = format_output(res, [seg], add_timing=settings.output_add_timing, verbose=settings.verbose_output)
            batch_out.append(out)
        out = merge_batch_outputs(batch_out)

    end_time = time.time()
    print('    -- Processing %.2f seconds of audio took %.2f seconds' % (len(audio_file)/SR, (end_time - start_time)))

    return out