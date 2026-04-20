import numpy as np


_silero_model = None


def vad_silero(audio, samplerate, umbral=0.4, duracion_minima=0.25, duracion_silencio_minima=0.3):
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps

    if samplerate != 16000:
        raise ValueError(
            f"VAD Silero requiere audio a 16 kHz, se obtuvo {samplerate} Hz."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[INFO] VAD Silero (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print(f"[INFO] VAD Silero (CPU)")

    global _silero_model
    if _silero_model is None:
        _silero_model = load_silero_vad()
        _silero_model = _silero_model.to(device)

    audio_mono = audio[:, 0] if audio.ndim > 1 else audio
    tensor = torch.from_numpy(audio_mono.astype(np.float32)).to(device)

    timestamps = get_speech_timestamps(
        tensor,
        _silero_model,
        threshold=umbral,
        min_speech_duration_ms=int(duracion_minima * 1000),
        min_silence_duration_ms=int(duracion_silencio_minima * 1000),
        sampling_rate=samplerate,
        return_seconds=False,
    )

    segmentos = []
    for seg in timestamps:
        inicio = round(seg["start"] / samplerate, 3)
        fin = round(seg["end"] / samplerate, 3)
        if (fin - inicio) >= duracion_minima:
            segmentos.append((inicio, fin))
    
    print(f"[INFO] {len(segmentos)} segmentos detectados")

    return segmentos
