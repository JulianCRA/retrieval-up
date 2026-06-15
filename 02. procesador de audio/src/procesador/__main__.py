import argparse
import gc
import sys
from pathlib import Path

from compartido import json_utils as ju
from compartido.rutas import DESCARGAS_DIR
from compartido.utils import cronometrar, crear_perfil_hardware, cronometro_activo, medir

import soundfile as sf
import noisereduce as nr
import numpy as np
from tqdm import tqdm

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
    2 GB de RAM. En esta implementación el uso de múltiples núcleos es mucho mas eficiente que el uso de GPU, por lo que se recomienda usar CPU incluso si se dispone de GPU.

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
        required=True,
        action="append",
        dest="hashes",
        metavar="HASH",
        help="Hash a procesar. Repetir para procesar varios en un solo comando.",
    )

    parser.add_argument(
        "-m", "--metodo",
        help = "Escoger el método de detección de voz (VAD) para eliminar silencios [energia|silero|webrtc]",
        choices = ["energia", "silero", "webrtc"],
    )

    args = parser.parse_args()

    procesar(args.hashes, args.metodo)

def procesar(hashes: list[str], metodo=None):
    from concurrent.futures import ProcessPoolExecutor, BrokenExecutor as BrokenProcessPool
    from procesador.vad_silero import _init_worker

    perfil = crear_perfil_hardware(forzado={"device": "cpu"})
    executor = None
    n_workers = None
    if metodo == "silero":
        import psutil
        n_workers = psutil.cpu_count(logical=False)
        executor = ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker)
        print(f"[INFO] Pool compartido de {n_workers} workers Silero creado para {len(hashes)} hash(es).")

    fallos: list[str] = []
    total = len(hashes)
    try:
        for hash in tqdm(hashes, desc="Procesando", unit="hash"):
            try:
                procesar_hash(hash, metodo, perfil=perfil, executor=executor)
            except BrokenProcessPool as e:
                print(f"[ERROR] Hash '{hash}': {e}")
                fallos.append(hash)
                if metodo == "silero":
                    try:
                        executor.shutdown(wait=False)
                    except Exception:
                        pass
                    executor = ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker)
                    print(f"[INFO] Pool recreado tras fallo de worker.")
            except Exception as e:
                print(f"[ERROR] Hash '{hash}': {e}")
                fallos.append(hash)
            finally:
                gc.collect()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if fallos:
        print(f"[ERROR] {len(fallos)} hash(es) fallaron: {', '.join(fallos)}")
        sys.exit(1)

def procesar_hash(hash, metodo=None, perfil=None, executor=None):
    folder = DESCARGAS_DIR / hash
    ruta_info = folder / "info.json"
    info = ju.cargar_archivo(ruta_info)
    if info is None:
        raise RuntimeError(f"No se encontró información para el hash '{hash}'.")
    procesar_archivo(info["descarga"]["archivo_descargado"], metodo=metodo, folder=folder, perfil=perfil, executor=executor)

def procesar_archivo(ruta, metodo=None, folder=None, perfil=None, executor=None):
    with cronometro_activo() as crono:
        with medir("lectura_audio"):
            audio, samplerate = sf.read(ruta)
        duracion = len(audio) / samplerate

        print(f"Procesando '{ruta}' - Duracion: {duracion:.2f} segundos ({int(duracion / 3600)}:{int((duracion % 3600) / 60):02d}:{int(duracion % 60):02d}:{int((duracion * 1000) % 1000):03d})")

        audio = normalizar_volumen(audio)
        audio = reducir_ruido(audio, samplerate, perfil=perfil)
        audio = normalizar_picos(audio)

        segmentos = None

        if metodo is not None:
            print(f"[INFO] Aplicando VAD '{metodo}' para eliminar silencios...")
            segmentos = vad(audio, samplerate, metodo=metodo, executor=executor)
            segmentos = procesar_segmentos(segmentos, min_gap=0.3)
            with medir("audio_prueba"):
                generar_audio_de_prueba(audio, samplerate, segmentos, folder)

        ruta_nueva = folder / "audio_procesado.wav"
        with medir("escritura_audio"):
            sf.write(ruta_nueva, audio, samplerate)
        print(f"Archivo procesado y guardado: '{ruta_nueva}'")

        if folder is not None:
            guardar_datos_procesamiento(folder, metodo, segmentos, crono.resumen(), ruta_nueva)


@cronometrar(etiqueta="reduccion_ruido")
def reducir_ruido(audio, samplerate, perfil=None):
    print(f"[INFO] Aplicando reducción de ruido...")
    if perfil is None:
        perfil = crear_perfil_hardware(forzado={"device": "cpu"})

    n = samplerate  # 1 segundo por muestra
    total = len(audio)
    # tomar muestras de ruido cada 5% del audio para obtener una representación representativa del ruido de fondo
    puntos = [int(total * p) for p in np.arange(0.05, 1.0, 0.05)]  # 5%, 10%, ..., 95%
    muestras = [audio[p:p + n] for p in puntos if p + n <= total]
    muestra_ruido = np.concatenate(muestras)
    
    print(f"[INFO] Usando PyTorch ({perfil['device']}) para reducción de ruido.")
    audio = nr.reduce_noise(
        y=audio,
        y_noise=muestra_ruido,
        sr=samplerate,
        stationary=True,
        prop_decrease=0.8,
        n_fft=1024,
        hop_length=512,
        n_jobs=1,  # Disabled to avoid joblib OS permission errors
        use_torch=True,
        device=perfil["device"]
    )
        
    return audio

@cronometrar(etiqueta="normalizacion_picos")
def normalizar_picos(audio):
    print(f"[INFO] Normalizando picos de audio...")
    # ajustar el audio para que el pico máximo esté a -1 dBFS
    decibeles_objetivo = -1.0 
    picos = np.max(np.abs(audio))
    if picos < 1e-8:
        return audio
    target_linear = 10 ** (decibeles_objetivo / 20.0)

    return audio * (target_linear / picos)

@cronometrar(etiqueta="normalizacion_volumen")
def normalizar_volumen(audio, umbral_db=-20.0):
    print(f"[INFO] Normalizando volumen del audio...")
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-8:
        return audio
    umbral_lineal = 10 ** (umbral_db / 20.0)
    factor_normalizacion = umbral_lineal / rms
    return audio * factor_normalizacion

@cronometrar(etiqueta="vad")
def vad(audio, samplerate, metodo="energia", executor=None):
    if metodo == "energia":
        return vad_energia(audio, samplerate)
    elif metodo == "silero":
        return vad_silero(audio, samplerate, executor=executor)
    elif metodo == "webrtc":
        return vad_webrtc(audio, samplerate)
    else:
        print(f"[WARNING] Método VAD '{metodo}' no reconocido.")
        return None

def procesar_segmentos(segmentos, min_gap=0.5):
    if not segmentos:
        return []

    segmentos = sorted(segmentos, key=lambda segmento: (segmento[0], segmento[1]))
    fusionados = [list(segmentos[0])]
    for inicio, fin in segmentos[1:]:
        if (inicio - fusionados[-1][1]) <= min_gap:
            fusionados[-1][1] = max(fusionados[-1][1], fin)
        else:
            fusionados.append([inicio, fin])

    resultado = [(round(i, 3), round(f, 3)) for i, f in fusionados]
    print(f"[INFO] {len(resultado)} segmentos luego de fusionar")
    return resultado

def generar_audio_de_prueba(audio, samplerate, segmentos, folder):
    t = np.arange(len(audio)) / samplerate
    beep_mono = (np.sin(2 * np.pi * 1000 * t) * 0.1).astype(audio.dtype)
    beep = np.stack([beep_mono] * audio.shape[1], axis=1) if audio.ndim == 2 else beep_mono

    audio_procesado = beep.copy()
    for inicio, fin in segmentos:
        inicio_muestra = int(inicio * samplerate)
        fin_muestra = int(fin * samplerate)
        audio_procesado[inicio_muestra:fin_muestra] = audio[inicio_muestra:fin_muestra]
    ruta_procesada = folder / "segmentos_con_silencio.wav"
    sf.write(ruta_procesada, audio_procesado, samplerate)

def guardar_datos_procesamiento(folder, metodo, segmentos, tiempos, ruta_nueva):
    ruta_info = folder / "info.json"
    data = {
        "archivo_procesado": str(ruta_nueva),
        "metodo_vad": metodo,
        "cantidad_segmentos": len(segmentos) if segmentos else 0,
        "archivo_segmentos": str(folder / "segmentos.json") if segmentos else None,
        "tiempos": tiempos,
    }

    ok = ju.guardar_nodo(ruta_info, "procesamiento", data)

    ok = ok and ju.guardar_archivo(folder / "segmentos.json", {
        "segmentos": segmentos
    })

    if ok:
        ju.guardar_nodo(ruta_info, "status", 2)
        ju.guardar_registro("status", 2, ruta=(folder.name,))

if __name__ == "__main__":
    main()