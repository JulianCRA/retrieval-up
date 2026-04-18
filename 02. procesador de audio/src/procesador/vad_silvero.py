import numpy as np


_silero_model = None


def vad_silvero(audio, samplerate, umbral=0.5, duracion_minima=0.25, duracion_silencio_minima=0.3):
    print(f"[INFO] Aplicando VAD Silero...")
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps

    if samplerate != 16000:
        raise ValueError(
            f"VAD Silero requiere audio a 16 kHz, se obtuvo {samplerate} Hz."
        )

    global _silero_model
    if _silero_model is None:
        _silero_model = load_silero_vad()

    audio_mono = audio[:, 0] if audio.ndim > 1 else audio
    tensor = torch.from_numpy(audio_mono.astype(np.float32))

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
            print(f"Voz detectada: Inicio = {inicio:.2f} s, Fin = {fin:.2f} s")

    return segmentos
