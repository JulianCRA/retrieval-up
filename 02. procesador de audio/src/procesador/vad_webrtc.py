import numpy as np


def vad_webrtc(audio, samplerate, agresividad=3, duracion_minima=0.25, duracion_silencio_minima=0.3):
    print(f"[INFO] VAD WebRTC (CPU)")
    import webrtcvad

    if samplerate not in (8000, 16000, 32000, 48000):
        raise ValueError(
            f"VAD WebRTC requiere 8000, 16000, 32000 o 48000 Hz, se obtuvo {samplerate} Hz."
        )

    vad = webrtcvad.Vad(agresividad)

    audio_mono = audio[:, 0] if audio.ndim > 1 else audio
    pcm = (audio_mono * 32767).astype(np.int16)

    # WebRTC solo acepta frames de 10, 20 o 30 ms
    duracion_frame_ms = 30
    frame_muestras = int(samplerate * duracion_frame_ms / 1000)

    frames = [
        pcm[i:i + frame_muestras]
        for i in range(0, len(pcm) - frame_muestras + 1, frame_muestras)
    ]

    # Agrupar frames activos en segmentos, fusionando silencios cortos
    segmentos = []
    inicio_actual = None
    fin_ultimo_activo = None

    for idx, frame in enumerate(frames):
        if len(frame) < frame_muestras:
            break
        activo = vad.is_speech(frame.tobytes(), samplerate)
        tiempo_inicio_frame = idx * duracion_frame_ms / 1000.0
        tiempo_fin_frame = tiempo_inicio_frame + duracion_frame_ms / 1000.0

        if activo:
            if inicio_actual is None:
                inicio_actual = tiempo_inicio_frame
            fin_ultimo_activo = tiempo_fin_frame
        else:
            if inicio_actual is not None:
                silencio = tiempo_inicio_frame - fin_ultimo_activo
                if silencio >= duracion_silencio_minima:
                    duracion = fin_ultimo_activo - inicio_actual
                    if duracion >= duracion_minima:
                        segmentos.append((round(inicio_actual, 3), round(fin_ultimo_activo, 3)))
                    inicio_actual = None
                    fin_ultimo_activo = None

    if inicio_actual is not None:
        duracion = fin_ultimo_activo - inicio_actual
        if duracion >= duracion_minima:
            segmentos.append((round(inicio_actual, 3), round(fin_ultimo_activo, 3)))

    print(f"[INFO] {len(segmentos)} segmentos detectados")

    return segmentos
