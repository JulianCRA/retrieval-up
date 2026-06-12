import argparse
import sys

from compartido import json_utils as ju
from compartido.rutas import DESCARGAS_DIR
from compartido.utils import crear_perfil_hardware, cronometro_activo

from .texto import INFO

def main():
    info_asr = INFO["INFO_ASR"]

    parser = argparse.ArgumentParser(
        prog = "transcriptor",
        description = "Procesa segmentos de audio para transcribirlos a texto utilizando modelos de ASR.",
        epilog = info_asr,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--hash",
        required=True,
        action="append",
        dest="hashes",
        metavar="HASH",
        help="Hash a transcribir. Repetir para procesar varios en un solo comando.",
    )

    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument(
        "-m", "--modelo",
        help = "Escoger el modelo de transcripción automática (ASR) para convertir audio a texto [vosk|cohere|whisper:small|whisper:base|whisper:turbo]",
        choices = ["vosk", "cohere", "whisper:small", "whisper:base", "whisper:turbo"],
    )

    grupo.add_argument(
        "-i", "--info",
        help = "Mostrar información detallada sobre los modelos de ASR disponibles",
        choices=["vosk", "cohere", "whisper"],
    )

    parser.add_argument(
        "--forzar-cpu",
        action="store_true",
        dest="forzar_cpu",
        help="Forzar uso de CPU aunque haya GPU disponible.",
    )

    args = parser.parse_args()

    if args.info:
        print(INFO[f"INFO_{args.info.upper()}"])
        sys.exit(0)
    elif args.modelo is None:
        args.modelo = "vosk"
    procesar(args.hashes, args.modelo, forzar_cpu=args.forzar_cpu)

def procesar(hashes: list[str], modelo="vosk", forzar_cpu: bool = False):
    forzado = {"device": "cpu"} if forzar_cpu else None
    perfil = crear_perfil_hardware(forzado=forzado)
    fallos: list[str] = []
    for hash in hashes:
        try:
            procesar_hash(hash, modelo, perfil=perfil)
        except Exception as e:
            print(f"[ERROR] Hash '{hash}': {e}")
            fallos.append(hash)
    if fallos:
        print(f"[ERROR] {len(fallos)} hash(es) fallaron: {', '.join(fallos)}")
        sys.exit(1)

def procesar_hash(hash, modelo="vosk", perfil=None):
    folder = DESCARGAS_DIR / hash
    info = ju.cargar_archivo(folder / "info.json")
    if info is None:
        raise RuntimeError(f"No se encontró información para el hash '{hash}'.")

    audio_path = folder / info["procesamiento"]["archivo_procesado"]
    if not audio_path.exists():
        raise RuntimeError(f"No se encontró el archivo de audio '{audio_path}'.")

    segmentos_path = folder / "segmentos.json"
    if not segmentos_path.exists():
        raise RuntimeError(f"No se encontró el archivo de segmentos '{segmentos_path}'.")

    transcripciones_path = folder / "transcripciones.json"

    obtener_transcripcion(audio_path, segmentos_path, transcripciones_path, modelo=modelo, perfil=perfil)

def obtener_transcripcion(audio_path, segmentos_path, transcripciones_path, modelo="vosk", perfil=None):
    print(f"[INFO] Transcribiendo '{audio_path.name}' usando el modelo '{modelo}'...")
    paths = {
        "audio": audio_path,
        "segmentos": segmentos_path,
        "transcripciones": transcripciones_path
    }

    with cronometro_activo() as crono:
        if modelo == "vosk":
            from .vosk_asr import transcribir_vosk
            transcribir_vosk(paths, perfil=perfil)
        elif modelo.startswith("whisper:"):
            variante = modelo.split(":", 1)[1]
            from .whisper_asr import transcribir_whisper
            transcribir_whisper(paths, modelo=variante, perfil=perfil)
        elif modelo == "cohere":
            from .cohere_asr import transcribir_cohere
            transcribir_cohere(paths, perfil=perfil)
        else:
            raise ValueError(f"Modelo '{modelo}' no soportado.")

        duracion = audio_path.stat().st_size / (16000 * 2)
        tiempos = crono.resumen()
        rt_factor = round(tiempos["_total"] / duracion, 3) if duracion > 0 else None
        speed_up = round(1 / rt_factor, 2) if rt_factor and rt_factor > 0 else None
        ju.guardar_nodos(paths["transcripciones"], {
            "tiempos": tiempos,
            "duracion_audio": round(duracion, 2),
            "rt_factor": rt_factor,
            "speed_up": f"{speed_up}x" if speed_up is not None else None,
        })

if __name__ == "__main__":
    main()
