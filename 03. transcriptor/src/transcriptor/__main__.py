import argparse
import sys
from pathlib import Path

from compartido import json_utils as ju
from compartido.rutas import DESCARGAS_DIR
from compartido.utils import cronometrar

from .texto import INFO

def main():
    info_asr = INFO["INFO_ASR"]

    parser = argparse.ArgumentParser(
        prog = "transcriptor",
        description = "Procesa segementosde audio para transcribirlos a texto utilizando modelos de ASR.",
        epilog = info_asr,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--hash",
        required = True,
        help = "Obtener segmentos de audio a partir del hash generado en el proceso de descarga",
    )

    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument(
        "-m", "--modelo",
        help = "Escoger el modelo de transcripción automática (ASR) para convertir audio a texto [vosk|wac2vec|cohere|whisper|qwen]",
        choices = ["vosk", "wac2vec", "cohere", "whisper", "qwen"],
    )

    grupo.add_argument(
        "-i", "--info",
        help = "Mostrar información detallada sobre los modelos de ASR disponibles",
        choices=["vosk", "wac2vec", "cohere", "whisper", "qwen"],
    )

    args = parser.parse_args()

    if args.info:
        print(INFO[f"INFO_{args.info.upper()}"])
        sys.exit(0)

    procesar_hash(args.hash, args.modelo)

def procesar_hash(hash, modelo="vosk"):
    folder = DESCARGAS_DIR / hash
    info = ju.cargar_archivo(folder / "info.json")
    if info is None:
        print(f"[ERROR] No se encontró información para el hash '{hash}'.")
        sys.exit(1)

    audio_path = folder / info["descarga"]["archivo_descargado"]
    if not audio_path.exists():
        print(f"[ERROR] No se encontró el archivo de audio '{audio_path}'.")
        sys.exit(1)

    segmentos_path = folder / "segmentos.json"
    if not segmentos_path.exists():
        print(f"[ERROR] No se encontró el archivo de segmentos '{segmentos_path}'.")
        sys.exit(1)

    cargar_modelo(audio_path, modelo=modelo, folder=folder)

@cronometrar
def cargar_modelo(audio_path, modelo="vosk", folder=None):
    print(f"Cargando modelo '{modelo}' para el archivo '{folder / audio_path}'...")
