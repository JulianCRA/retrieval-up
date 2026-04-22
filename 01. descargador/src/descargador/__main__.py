import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse
import time

import compartido.json_utils as ju
from compartido.rutas import DESCARGAS_DIR
from compartido.utils import obtener_identificador

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, ExtractorError

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".wma", ".aac"}
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv"}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS

OPTS = {
        "format": "bestaudio/best",
        "outtmpl": str(DESCARGAS_DIR / "%(title)s - %(id)s.%(ext)s"),
        "prefer_ffmpeg": True,
        # TODO : se puede aceptar playlists pero hay que manejar el caso de múltiples archivos descargados. por ahora se ignoora ese caso
        "noplaylist": True,
        "geo_bypass": True,
        "no_warnings": True,
        "quiet": False,
        "writethumbnail": False,
        "enable_file_urls": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "0"}
        ],
        # "postprocessor_args": ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"]
        "postprocessor_args": {
            "ffmpegextractaudio": ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"]
        }
    }

def main():
    parser = argparse.ArgumentParser(
        prog = "descargador",
        description = "Carga archivos de audio y video desde fuentes locales o remotas"
    )

    parser.add_argument(
        "-s", "--source",
        help = "Fuente del archivo de audio o video a cargar, lista de archivos o directorio local",
        required=True,
    )

    args = parser.parse_args()

    if args.source:
        determinar_fuente(args.source)


def determinar_fuente(ruta):
    path = Path(ruta)

    if urlparse(ruta).scheme in ("http", "https"):
        procesar_recurso(ruta)
    elif not path.exists():
        print(f"[ERROR] La ruta '{ruta}' no existe.")
        sys.exit(1)
    elif path.is_file():
        if path.suffix.lower() in MEDIA_EXTS:
            procesar_recurso(path.as_uri())
        else:
            #leer archivo de texto
            procesar_batch( leer_archivo_de_texto(path) )

    elif path.is_dir():
        procesar_batch( leer_directorio(path))
                
    else:
        print(f"[ERROR] La ruta '{ruta}' no es un archivo ni un directorio válido.")
        sys.exit(1)

def procesar_recurso(uri):
    try:
        with YoutubeDL(OPTS) as ydl:
            info = ydl.extract_info(uri, download=False)
            actual_uri = info.get("webpage_url", uri)
            r_id = obtener_identificador(actual_uri)

            reg = ju.cargar_registro(r_id)
            if reg and reg.get("status", 0) >= 1:
                print(f"[INFO] El recurso '{actual_uri}' ya ha sido procesado previamente. Saltando descarga.")
                return

            carpeta = DESCARGAS_DIR / r_id
            carpeta.mkdir(exist_ok=True)

            opts = {**OPTS, "outtmpl": str(carpeta / "audio_original.%(ext)s")}
            with YoutubeDL(opts) as ydl2:
                inicio = time.perf_counter()
                info = ydl2.extract_info(actual_uri, download=True)
                total = time.perf_counter() - inicio
                info['tiempo_descarga'] = total
                
                print(f"[INFO] Descarga completada en {total:.2f} segundos.")
                archivo_descargado = str(Path(ydl2.prepare_filename(info)).with_suffix(".wav"))

                if registrar_descarga(r_id, actual_uri, info):
                    registrar_detalles(r_id, actual_uri, info, archivo_descargado)

    except (DownloadError, ExtractorError) as e:
        print(f"[ERROR] Error al procesar '{uri}': {e}")

def leer_directorio(directorio):
    archivos = []
    for item in directorio.iterdir():
        if item.is_file() and item.suffix.lower() in MEDIA_EXTS:
            archivos.append(item.as_uri())
    return archivos

def leer_archivo_de_texto(archivo):
    uris = []
    with open(archivo, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            path = Path(line)
            if line == "" or line.startswith("#"):
                continue
            elif path.is_absolute() and path.is_file() and path.suffix.lower() in MEDIA_EXTS:
                uris.append(path.as_uri())
            elif urlparse(line).scheme in ("http", "https"):
                uris.append(line)
            else:
                print(f"[WARNING] Línea ignorada: '{line}' no es un archivo válido ni una URL.")
    return uris

def procesar_batch(uris):
    print(f"[INFO] Procesando {len(uris)} recursos...")
    for uri in uris:
        procesar_recurso(uri)

def registrar_descarga(r_id, uri, info):
    archivo_detalle = DESCARGAS_DIR / r_id / "info.json"
    data1 = {
        "uri": uri,
        "title": info.get("title", ""),
        "status": 0,
        "archivo_detalle": str(archivo_detalle),
    }
    data2 = {
        "hash": r_id,
        "title": info.get("title", ""),
        "status": 0
    }
    return ju.guardar_registro(r_id, data1) and ju.guardar_archivo(archivo_detalle, data2)

def registrar_detalles(r_id, uri, info, archivo_descargado):
    archivo_detalle = DESCARGAS_DIR / r_id / "info.json"
    ok = ju.guardar_registro("status", 1, ruta=(r_id,))
    ok = ok and ju.guardar_nodo(archivo_detalle, "descarga", {
        "uri": uri,
        "fuente": info.get("extractor_key", ""),
        "duracion": info.get("duration", 0),
        "tamano": info.get("filesize", 0),
        "archivo_descargado": archivo_descargado,
        "tiempo_descarga": info.get("tiempo_descarga", 0)
    })
    ok = ok and ju.guardar_nodo(archivo_detalle, "status", 1)
    return ok
    
    

if __name__ == "__main__":
    main()