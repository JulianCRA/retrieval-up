import argparse
import sys
from pathlib import Path

from compartido import json_utils as ju
from compartido.rutas import DESCARGAS_DIR

import soundfile as sf
import noisereduce as nr
import numpy as np

from procesador.vad_energia import vad_energia
from procesador.vad_silero import vad_silero
from procesador.vad_webrtc import vad_webrtc

# from descargador.__main__ import procesar_archivo

def main():
    info_vad = """
Métodos de detección de voz (VAD):

  energia:
    Detecta voz midiendo la energía de la señal en ventanas de tiempo. Es el método
    más simple y rápido, sin dependencias adicionales. Su rendimiento depende en gran
    medida de la calidad del audio: funciona bien en entornos silenciosos, pero el
    ruido de fondo puede confundirse con voz y requiere ajuste manual de umbrales.
    Recursos: mínimo 1 núcleo de CPU y ~50 MB de RAM. No requiere GPU. Recomendado
    cualquier CPU moderna con al menos 512 MB de RAM disponible.

  silero:
    Utiliza un modelo de red neuronal preentrenado (Silero VAD) para identificar
    segmentos de voz. Es el método más preciso y robusto ante ruido de fondo y
    variaciones de volumen, sin necesidad de ajuste manual. Requiere la dependencia
    'silero-vad' (torch) y es considerablemente más lento que los otros métodos.
    Recursos: mínimo 2 núcleos de CPU y ~500 MB de RAM. Recomendado 4 núcleos y
    2 GB de RAM. Una GPU compatible con CUDA acelera significativamente el proceso,
    aunque no es imprescindible.

  webrtc:
    Implementa el VAD incluido en el proyecto WebRTC, diseñado para comunicaciones
    en tiempo real. Ofrece un equilibrio entre velocidad y precisión. Solo admite
    tasas de muestreo de 8000, 16000, 32000 o 48000 Hz, y su precisión disminuye
    en grabaciones con ruido de fondo elevado.
    Recursos: mínimo 1 núcleo de CPU y ~100 MB de RAM. No requiere GPU. Recomendado
    cualquier CPU moderna con al menos 512 MB de RAM disponible.
"""

    parser = argparse.ArgumentParser(
        prog = "procesador",
        description = "Procesa archivos de audio para remover ruido de fondo, mejorar calidad, etc.",
        epilog = info_vad,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--hash",
        required = True,
        help = "Buscar contenido a partir del hash generado en el proceso de descarga",
    )

    parser.add_argument(
        "-m", "--metodo",
        help = "Escoger el método de detección de voz (VAD) para eliminar silencios [energia|silero|webrtc]",
        choices = ["energia", "silero", "webrtc"],
    )

    args = parser.parse_args()

    procesar_hash(args.hash, args.metodo)

def procesar_hash(hash, metodo=None):
    info = ju.cargar_nodo(DESCARGAS_DIR, hash)
    if info is None:
        print(f"[ERROR] No se encontró información para el hash '{hash}'.")
        sys.exit(1)
    procesar_archivo(info["archivo"], metodo=metodo)

def procesar_archivo(ruta, metodo=None):
    audio, samplerate = sf.read(ruta)
    duracion = len(audio) / samplerate

    print(f"Procesando '{ruta}' - Duración: {duracion:.2f} segundos ({int(duracion / 3600)}:{int((duracion % 3600) / 60):02d}:{int(duracion % 60):02d}:{int((duracion * 1000) % 1000):03d})")

    audio = reducir_ruido(audio, samplerate)
    audio = normalizar_picos(audio)
    audio = normalizar_volumen(audio)
    if metodo is not None:
        print(f"[INFO] Aplicando VAD '{metodo}' para eliminar silencios...")
        segmentos = vad(audio, samplerate, metodo=metodo)
        segmentos = procesar_segmentos(segmentos)
        
        ## generar audio procesado con solo los segmentos de voz detectados
        # audio_procesado = np.zeros_like(audio)
        # for inicio, fin in segmentos:
        #     inicio_muestra = int(inicio * samplerate)
        #     fin_muestra = int(fin * samplerate)
        #     audio_procesado[inicio_muestra:fin_muestra] = audio[inicio_muestra:fin_muestra]
        # ruta_procesada = Path(ruta).with_name(Path(ruta).stem + "_procesado.wav")
        # sf.write(ruta_procesada, audio_procesado, samplerate)
        

    ruta_nueva = Path(ruta).with_name(Path(ruta).stem + "_limpio.wav")
    sf.write(ruta_nueva, audio, samplerate)
    print(f"Archivo procesado y guardado: '{ruta_nueva}'")

    #TODO: actualizar el nodo con la información del procesamiento (segmentos detectados, ruta del archivo procesado, etc.)

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

def normalizar_volumen(audio, umbral_db=-20.0):
    print(f"[INFO] Normalizando volumen del audio...")
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-8:
        return audio
    umbral_lineal = 10 ** (umbral_db / 20.0)
    factor_normalizacion = umbral_lineal / rms
    return audio * factor_normalizacion

def vad(audio, samplerate, metodo="energia"):
    if metodo == "energia":
        return vad_energia(audio, samplerate)
    elif metodo == "silero":
        return vad_silero(audio, samplerate)
    elif metodo == "webrtc":
        return vad_webrtc(audio, samplerate)
    else:
        print(f"[WARNING] Método VAD '{metodo}' no reconocido.")
        return None


def procesar_segmentos(segmentos, margen=0.2):
    # anadir margen a los segmentos para evitar cortar palabras al inicio o final
    segmentos_procesados = []
    for inicio, fin in segmentos:
        inicio_procesado = max(0, inicio - margen)
        fin_procesado = fin + margen
        segmentos_procesados.append((round(inicio_procesado, 3), round(fin_procesado, 3)))

    # fusionar segmentos que se solapan
    segmentos_fusionados = []
    for inicio, fin in sorted(segmentos_procesados):
        if not segmentos_fusionados or inicio > segmentos_fusionados[-1][1]:
            segmentos_fusionados.append((inicio, fin))
        else:
            segmentos_fusionados[-1] = (segmentos_fusionados[-1][0], max(segmentos_fusionados[-1][1], fin))

    return segmentos_fusionados

if __name__ == "__main__":
    main()