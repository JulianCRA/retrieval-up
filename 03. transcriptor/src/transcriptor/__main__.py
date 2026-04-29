import argparse
import sys

from compartido import json_utils as ju
from compartido.rutas import DESCARGAS_DIR
from compartido.utils import cronometrar, crear_perfil_hardware

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
    elif args.modelo is None:
        args.modelo = "vosk"
    procesar_hash(args.hash, args.modelo)

def procesar_hash(hash, modelo="vosk"):
    folder = DESCARGAS_DIR / hash
    info = ju.cargar_archivo(folder / "info.json")
    if info is None:
        print(f"[ERROR] No se encontró información para el hash '{hash}'.")
        sys.exit(1)

    audio_path = folder / info["procesamiento"]["archivo_procesado"]
    if not audio_path.exists():
        print(f"[ERROR] No se encontró el archivo de audio '{audio_path}'.")
        sys.exit(1)

    segmentos_path = folder / "segmentos.json"
    if not segmentos_path.exists():
        print(f"[ERROR] No se encontró el archivo de segmentos '{segmentos_path}'.")
        sys.exit(1)

    transcripciones_path = folder / "transcripciones.json"

    obtener_transcripcion(audio_path, segmentos_path, transcripciones_path, modelo=modelo)

def obtener_transcripcion(audio_path, segmentos_path, transcripciones_path, modelo="vosk"):
    print(f"[INFO] Transcribiendo '{audio_path.name}' usando el modelo '{modelo}'...")
    paths = {
        "audio": audio_path,
        "segmentos": segmentos_path,
        "transcripciones": transcripciones_path
    }

    if modelo == "vosk":
        from .vosk_asr import transcribir_vosk
        transcribir_vosk(paths)
        tiempo_transcripcion = round(transcribir_vosk.elapsed, 2)
        duracion = audio_path.stat().st_size / (16000 * 2)  
        rt_factor = round(tiempo_transcripcion / duracion, 2) if duracion > 0 else None
        speed_up = round(1 / rt_factor, 2) if rt_factor > 0 else None
        ju.guardar_nodos(paths["transcripciones"], {
            "tiempo_transcripcion": tiempo_transcripcion,
            "rt_factor": rt_factor,
            "speed_up": str(speed_up) + "x" if speed_up is not None else None
        })
