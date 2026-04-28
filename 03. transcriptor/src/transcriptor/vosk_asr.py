import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

from compartido.rutas import DESCARGAS_DIR
from compartido.json_utils import cargar_archivo
from .chunks import obtener_fragmentos_asr

import wave
import vosk

MODELOS_DIR = DESCARGAS_DIR / "modelos" / "vosk"

MODELO_ES = "vosk-model-es-0.42" # opcion: vosk-model-small-es-0.42 (más ligero pero menos preciso)
MODELO_ES_URL = f"https://alphacephei.com/vosk/models/{MODELO_ES}.zip"

PERFIL_VOSK = {
    "padding": 0.18,
    "join_gap": 0.50,
    "duracion_minima": 1.20,
    "duracion_target": 6.00,
    "duracion_maxima": 10.00,
    "overlap": 0.15,
}


def _obtener_modelo_es() -> Path:
    """Devuelve la ruta al modelo en español, descargándolo si no existe."""
    ruta_modelo = MODELOS_DIR / MODELO_ES
    if ruta_modelo.exists():
        return ruta_modelo

    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = MODELOS_DIR / f"{MODELO_ES}.zip"

    print(f"[INFO] Modelo Vosk en español no encontrado. Descargando desde:\n       {MODELO_ES_URL}")

    def _progreso(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size * 100 // total_size, 100)
            print(f"\r       {pct}% descargado...", end="", flush=True)

    try:
        urlretrieve(MODELO_ES_URL, zip_path, reporthook=_progreso)
        print() 
    except Exception as e:
        print(f"\n[ERROR] Fallo al descargar el modelo: {e}")
        if zip_path.exists():
            zip_path.unlink()
        raise

    try:
        print(f"[INFO] Extrayendo modelo en '{MODELOS_DIR}'...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(MODELOS_DIR)
    except Exception as e:
        print(f"[ERROR] Fallo al extraer el modelo: {e}")
        if zip_path.exists():
            zip_path.unlink()
        raise

    zip_path.unlink()
    print(f"[INFO] Modelo listo en '{ruta_modelo}'.")
    return ruta_modelo


def _transcribir_segmento(modelo, audio_path, inicio, fin):
    """Worker: abre su propio file handle y crea su propio KaldiRecognizer."""
    with wave.open(str(audio_path), "rb") as audio:
        sample_rate = audio.getframerate()

        rec = vosk.KaldiRecognizer(modelo, sample_rate)

        frame_inicio = int(inicio * sample_rate)
        num_frames = int((fin - inicio) * sample_rate)
        audio.setpos(frame_inicio)
        segmento = audio.readframes(num_frames)

    print(f"[DEBUG] Procesando segmento {inicio:.2f}s - {fin:.2f}s <{fin - inicio:.2f}s - {len(segmento)/1024:.2f} KB>")
    rec.AcceptWaveform(segmento)
    return inicio, fin, rec.FinalResult()


def transcribir_vosk(audio_path, paths, num_workers=6) -> None:
    print(f"[INFO] Transcribiendo '{audio_path}' con Vosk...")

    segmentos = cargar_archivo(paths["segmentos"]).get("segmentos", [])
    segmentos = obtener_fragmentos_asr(segmentos, PERFIL_VOSK)
    if not segmentos:
        print("[INFO] No hay fragmentos ASR para transcribir.")
        return

    ruta_modelo = _obtener_modelo_es()
    print("[INFO] Cargando modelo Vosk en español (una sola vez)...")
    modelo = vosk.Model(str(ruta_modelo))
    print(f"[INFO] Modelo listo. Procesando con {num_workers} workers en paralelo...")

    resultados = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futuros = {
            executor.submit(_transcribir_segmento, modelo, audio_path, inicio, fin): (inicio, fin)
            for inicio, fin in segmentos
        }
        for futuro in as_completed(futuros):
            inicio, fin, resultado = futuro.result()
            print(f"[DEBUG] Resultado Vosk {inicio:.2f}s - {fin:.2f}s: {resultado}")
            resultados[(inicio, fin)] = resultado

    for inicio, fin in segmentos:
        print(f"[INFO] {inicio:.2f}s - {fin:.2f}s → {resultados[(inicio, fin)]}")