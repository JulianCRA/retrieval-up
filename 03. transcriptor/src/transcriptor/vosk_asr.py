import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from compartido.rutas import DESCARGAS_DIR

MODELOS_DIR = DESCARGAS_DIR / "modelos" / "vosk"

MODELO_ES = "vosk-model-es-0.42" # opcion: vosk-model-small-es-0.42 (más ligero pero menos preciso)
MODELO_ES_URL = f"https://alphacephei.com/vosk/models/{MODELO_ES}.zip"


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

    urlretrieve(MODELO_ES_URL, zip_path, reporthook=_progreso)
    print()  # salto de línea tras la barra de progreso

    print(f"[INFO] Extrayendo modelo en '{MODELOS_DIR}'...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(MODELOS_DIR)

    zip_path.unlink()
    print(f"[INFO] Modelo listo en '{ruta_modelo}'.")
    return ruta_modelo


def transcribir_vosk(audio_path: Path, folder: Path) -> None:
    _obtener_modelo_es()
    print(f"[INFO] Transcribiendo '{audio_path}' con Vosk...")
