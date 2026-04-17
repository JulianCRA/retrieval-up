import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse
import hashlib

import compartido.json_utils as ju
from compartido.rutas import ARCHIVO_REGISTRO, DESCARGAS_DIR

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
        "postprocessor_args": ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"]
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
    else:
        print("[ERROR] Se requiere una fuente de recursos audiovisuales.")
        sys.exit(1)

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
    hash = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:24]
    if ju.cargar_registro(hash):
        print(f"[INFO] El recurso '{uri}' ya ha sido procesado previamente. Saltando descarga.")
        return

    try:
        with YoutubeDL(OPTS) as ydl:
            info = ydl.extract_info(uri, download=True)
            registrar_descarga(hash, info)
            registrar_detalles(hash, info, ydl.prepare_filename(info))
    except (DownloadError, ExtractorError) as e:
        print(f"[ERROR] Error al descargar '{uri}': {e}")

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
            elif path.is_file() and path.suffix.lower() in MEDIA_EXTS:
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

def registrar_descarga(hash, info):
    data = {
        "uri": info.get("webpage_url", ""),
        "title": info.get("title", ""),
        "status": 1,
        "archivo_detalle": str(DESCARGAS_DIR / f"{hash}.json"),
    }
    ju.anadir_registro(hash, data)
    
def registrar_detalles(hash, info, archivo_descargado):
    ju.anadir_nodos(DESCARGAS_DIR / f"{hash}.json",{
        "hash": hash,
        "title": info.get("title", ""),
        "uri": info.get("webpage_url", ""),
        "archivo": archivo_descargado,
        "status": 'OK'
    })
    
    

if __name__ == "__main__":
    main()