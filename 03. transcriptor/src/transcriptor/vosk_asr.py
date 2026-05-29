import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

from compartido.rutas import DESCARGAS_DIR
from compartido.json_utils import cargar_archivo, guardar_archivo
from compartido.utils import crear_perfil_hardware, cronometrar
from .chunks import obtener_fragmentos_asr

import wave
import vosk
from tqdm import tqdm

MODELOS_DIR = DESCARGAS_DIR / "modelos" / "vosk"

MODELO_ES = "vosk-model-es-0.42" # opcion: vosk-model-small-es-0.42 (más ligero pero menos preciso)
MODELO_ES_URL = f"https://alphacephei.com/vosk/models/{MODELO_ES}.zip"

PERFIL_VOSK = {
    "padding": 0.18,
    "join_gap": 0.50,
    "duracion_minima": 5.00,
    "duracion_target": 20.00,
    "duracion_maxima": 24.00,
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


def _transcribir_segmento(modelo, audio_bytes, sample_rate, bytes_per_frame, inicio, fin):
    byte_start = int(inicio * sample_rate) * bytes_per_frame
    byte_end   = int(fin   * sample_rate) * bytes_per_frame
    chunk = audio_bytes[byte_start:byte_end]

    rec = vosk.KaldiRecognizer(modelo, sample_rate)
    rec.AcceptWaveform(chunk)
    return inicio, fin, rec.FinalResult()

@cronometrar(etiqueta="Transcripción total")
def transcribir_vosk(paths):
    modelo_path = _obtener_modelo_es()
    vosk.SetLogLevel(-1)
    modelo = vosk.Model(str(modelo_path))

    segmentos = cargar_archivo(paths["segmentos"])
    if segmentos is None:
        print(f"[ERROR] No se pudo cargar los segmentos desde '{paths['segmentos']}'.")
        return
    segmentos = obtener_fragmentos_asr(segmentos["segmentos"], PERFIL_VOSK)

    transcripciones = []
    num_workers = crear_perfil_hardware()["cpu_physical_cores"]
    # num_workers = 4

    with wave.open(str(paths["audio"]), "rb") as wf:
        sample_rate = wf.getframerate()
        bytes_per_frame = wf.getnchannels() * wf.getsampwidth()
        audio_bytes = wf.readframes(wf.getnframes())

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        print(f"[INFO] Transcribiendo {len(segmentos)} segmentos con {num_workers} workers...")
        futures = {
            executor.submit(_transcribir_segmento, modelo, audio_bytes, sample_rate, bytes_per_frame, seg[0], seg[1]): (seg[0], seg[1])
            for seg in segmentos
        }
        # for future in as_completed(futures):
        for future in tqdm(as_completed(futures), total=len(futures), desc="Transcribiendo", unit="segmento"):
            try:
                inicio, fin, resultado = future.result()
                transcripciones.append({
                    "inicio": inicio,
                    "fin": fin,
                    "duracion": round(fin - inicio, 2),
                    "texto": json.loads(resultado).get("text", "")
                })
                # print(f"[INFO] {inicio:.2f}s - {fin:.2f}s: '{transcripciones[-1]['texto']}'")

            except Exception as e:
                print(f"[ERROR] Fallo al transcribir un segmento: {e}")

    transcripciones.sort(key=lambda x: x["inicio"])
    full_text = " ".join([t["texto"] for t in transcripciones])

    data = {
        "modelo": "Vosk SPA (full)",
        "perfil": PERFIL_VOSK,
        "num_workers": num_workers,
        "tiempo_transcripcion": None,
        "speed_up": None,
        "rt_factor": None,
        "num_segmentos": len(transcripciones),
        "duracion_promedio_segmento": round(sum(t["duracion"] for t in transcripciones) / len(transcripciones), 2) if transcripciones else 0,
        "transcripciones": transcripciones,
        "texto": full_text
    }
    guardar_archivo(paths["transcripciones"], data)

    