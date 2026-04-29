import os
import numpy as np
import psutil


_silero_model = None  # per-worker global


def _init_worker():
    global _silero_model
    import torch
    from silero_vad import load_silero_vad
    torch.set_num_threads(1)
    _silero_model = load_silero_vad()


def _procesar_chunk(args):
    global _silero_model
    chunk, samplerate, umbral, duracion_minima, duracion_silencio_minima, offset = args
    import torch
    from silero_vad import get_speech_timestamps

    with torch.inference_mode():
        timestamps = get_speech_timestamps(
            torch.from_numpy(chunk),
            _silero_model,
            threshold=umbral,
            min_speech_duration_ms=int(duracion_minima * 1000),
            min_silence_duration_ms=int(duracion_silencio_minima * 1000),
            sampling_rate=samplerate,
            return_seconds=False,
        )

    return [{"start": t["start"] + offset, "end": t["end"] + offset} for t in timestamps]


def vad_silero(audio, samplerate, umbral=0.4, duracion_minima=0.25, duracion_silencio_minima=0.3, n_workers=None):
    from concurrent.futures import ProcessPoolExecutor

    if samplerate != 16000:
        raise ValueError(f"VAD Silero requiere audio a 16 kHz, se obtuvo {samplerate} Hz.")

    audio_mono = (audio[:, 0] if audio.ndim > 1 else audio).astype(np.float32)
    
    real_cpu_count = psutil.cpu_count(logical=False)
    if n_workers is None:
        n_workers = real_cpu_count
    elif n_workers > real_cpu_count:
        print(f"[WARNING] Se solicitaron {n_workers} workers, pero solo se detectaron {real_cpu_count} CPUs físicas. Usando {real_cpu_count} workers.")
        n_workers = real_cpu_count
    
    total = len(audio_mono)
    chunk_size = total // n_workers
    overlap = samplerate  # 1s overlap at boundaries

    print(f"[INFO] Procesando con {n_workers} workers (chunks de ~{chunk_size / samplerate:.0f}s + 1s overlap)...")

    args_list = [
        (
            audio_mono[max(0, i * chunk_size - overlap) : total if i == n_workers - 1 else (i + 1) * chunk_size + overlap],
            samplerate, umbral, duracion_minima, duracion_silencio_minima,
            max(0, i * chunk_size - overlap),
        )
        for i in range(n_workers)
    ]

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as executor:
        results = list(executor.map(_procesar_chunk, args_list))

    timestamps = sorted(
        (t for chunk_result in results for t in chunk_result),
        key=lambda t: t["start"],
    )

    segmentos = [
        (round(t["start"] / samplerate, 3), round(t["end"] / samplerate, 3))
        for t in timestamps
        if (t["end"] - t["start"]) / samplerate >= duracion_minima
    ]

    print(f"[INFO] {len(segmentos)} segmentos detectados")
    return segmentos