import argparse
import sys
from pathlib import Path

import soundfile as sf
import noisereduce as nr
import numpy as np

from descargador.__main__ import procesar_archivo

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

    args = parser.parse_args()

    if not (Path(args.input).is_file() and Path(args.input).suffix.lower() == ".wav"):
        print(f"Error: La ruta '{args.input}' no es un archivo WAV válido.")
        sys.exit(1)

    procesar_archivo(args.input)

def procesar_archivo(ruta):
    audio, samplerate = sf.read(ruta)
    duracion = len(audio) / samplerate

    print(f"Procesando '{ruta}' - Duración: {duracion:.2f} segundos ({int(duracion / 3600)}:{int((duracion % 3600) / 60):02d}:{int(duracion % 60):02d}:{int((duracion * 1000) % 1000):03d})")

    audio = reducir_ruido(audio, samplerate)
    audio = normalizar_picos(audio)
    silencios = detectar_silencio(audio, samplerate)

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

# deteccion basada en energ´ıa mediante el calculo de la amplitud cuadratica media (RMS) por ventana temporal
def detectar_silencio(audio, samplerate, umbral_db=-40.0, duracion_minima=0.5):
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

    return silencio    

if __name__ == "__main__":
    main()