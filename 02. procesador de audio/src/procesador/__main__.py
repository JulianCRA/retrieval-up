import argparse
import sys
from pathlib import Path

import soundfile as sf
import noisereduce as nr
import numpy as np

# from descargador.__main__ import procesar_archivo

def main():
    parser = argparse.ArgumentParser(
        prog = "procesador",
        description = "Procesa archivos de audio para remover ruido de fondo, mejorar calidad, etc."
    )

    parser.add_argument(
        "-i", "--input",
        help = "Archivo de audio o directorio a procesar",
        required = True
    )

    parser.add_argument(
        "-m", "--metodo",
        help = "Escoger el método de detección de voz (VAD) para eliminar silencios [energia|silvero|wbrtc]",
        choices = ["energia", "silvero", "wbrtc"],
    )

    args = parser.parse_args()

    if not (Path(args.input).is_file() and Path(args.input).suffix.lower() == ".wav"):
        print(f"Error: La ruta '{args.input}' no es un archivo WAV válido.")
        sys.exit(1)

    procesar_archivo(args)

def procesar_archivo(args):
    ruta = args.input
    metodo = args.metodo or None
    audio, samplerate = sf.read(ruta)
    duracion = len(audio) / samplerate

    print(f"Procesando '{ruta}' - Duración: {duracion:.2f} segundos ({int(duracion / 3600)}:{int((duracion % 3600) / 60):02d}:{int(duracion % 60):02d}:{int((duracion * 1000) % 1000):03d})")

    audio = reducir_ruido(audio, samplerate)
    audio = normalizar_picos(audio)
    if metodo is not None:
        print(f"[INFO] Aplicando VAD '{metodo}' para eliminar silencios...")
        audio = vad(audio, samplerate, metodo=metodo)

    ruta_nueva = Path(ruta).with_name(Path(ruta).stem + "_limpio.wav")
    sf.write(ruta_nueva, audio, samplerate)
    print(f"Archivo procesado y guardado: '{ruta_nueva}'")

def reducir_ruido(audio, samplerate):
    print(f"[INFO] Aplicando reducción de ruido...")
    # Aplicar reducción de ruido no estacionario con una reducción del 50%
    audio = nr.reduce_noise(
        y=audio, 
        sr=samplerate,
        stationary=False,
        prop_decrease=0.5
    )

    # Aplicar una reduccion de ruido estacionario al 80% para eliminar el ruido de fondo constante
    audio = nr.reduce_noise(
        y=audio, 
        sr=samplerate,
        stationary=True,
        prop_decrease=0.8
    )

    return audio

def normalizar_picos(audio):
    print(f"[INFO] Normalizando picos de audio...")
    # ajustar el audio para que el pico máximo esté a -1 dBFS
    decibeles_objetivo = -1.0 
    picos = np.max(np.abs(audio))
    if picos < 1e-8:
        return audio
    target_linear = 10 ** (decibeles_objetivo / 20.0)

    return audio * (target_linear / picos)

def vad(audio, samplerate, metodo="energia"):
    if metodo == "energia":
        return vad_energia(audio, samplerate)
    elif metodo == "silvero":
        return vad_silvero(audio, samplerate)
    elif metodo == "wbrtc":
        return vad_wbrtc(audio, samplerate)
    else:
        print(f"[WARNING] Método VAD '{metodo}' no reconocido.")
        return audio


# deteccion basada en energ´ıa mediante el calculo de la amplitud cuadratica media (RMS) por ventana temporal
def vad_energia(audio, samplerate, umbral_db=-40.0, duracion_minima=0.5):
    print(f"[INFO] Detectando silencios en el audio...")
    ventana_duracion = 0.02
    ventana_muestras = int(ventana_duracion * samplerate)
    umbral_lineal = 10 ** (umbral_db / 20.0)

    silencio = []
    en_silencio = False
    inicio_silencio = 0

    for i in range(0, len(audio), ventana_muestras):
        ventana = audio[i:i + ventana_muestras]
        rms = np.sqrt(np.mean(ventana**2))

        if rms < umbral_lineal:
            if not en_silencio:
                en_silencio = True
                inicio_silencio = i / samplerate
        else:
            if en_silencio:
                en_silencio = False
                duracion_silencio = (i / samplerate) - inicio_silencio
                if duracion_silencio >= duracion_minima:
                    silencio.append((inicio_silencio, duracion_silencio))

    if en_silencio:
        duracion_silencio = (len(audio) / samplerate) - inicio_silencio
        if duracion_silencio >= duracion_minima:
            silencio.append((inicio_silencio, duracion_silencio))

    for inicio, duracion in silencio:
        print(f"Silencio detectado: Inicio = {inicio:.2f} s, Fin = {inicio + duracion:.2f} s")

    return silencio    

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

def vad_wbrtc(audio, samplerate, agresividad=3, duracion_minima=0.25, duracion_silencio_minima=0.3):
    print(f"[INFO] Aplicando VAD WebRTC...")
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
                        print(f"Voz detectada: Inicio = {inicio_actual:.2f} s, Fin = {fin_ultimo_activo:.2f} s")
                    inicio_actual = None
                    fin_ultimo_activo = None

    if inicio_actual is not None:
        duracion = fin_ultimo_activo - inicio_actual
        if duracion >= duracion_minima:
            segmentos.append((round(inicio_actual, 3), round(fin_ultimo_activo, 3)))
            print(f"Voz detectada: Inicio = {inicio_actual:.2f} s, Fin = {fin_ultimo_activo:.2f} s")

    return segmentos

if __name__ == "__main__":
    main()