import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse


AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".wma", ".aac"}
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv"}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS

OPTS = {
        "format": "bestaudio/best",
        "outtmpl": str(Path("descargas") / "%(title)s - %(id)s.%(ext)s"),
        "prefer_ffmpeg": True,
        "noplaylist": False,
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

    fuentes = parser.add_mutually_exclusive_group(required=True)

    # provcesar archivo individual
    fuentes.add_argument(
        "-l", "--local",
        help = "Fuente del archivo de audio o video a cargar, lista de archivos o directorio local"
    )

    # procesar varios archivos
    fuentes.add_argument(
        "-r", "--remoto",
        help = "Fuente del archivo de audio o video a cargar remoto (URL)"
    )

    args = parser.parse_args()

    if args.local:
        procesar_fuente_local(args.local)
    elif args.remoto:
        procesar_fuente_remota(args.remoto)

def procesar_fuente_local(ruta):
    path = Path(ruta)

    if not path.exists():
        print(f"Error: La ruta '{ruta}' no existe.")
        sys.exit(1)

    elif path.is_file():
        if path.suffix.lower() in MEDIA_EXTS:
            procesar_archivo(path)
        else:
            #leer archivo de texto
            procesar_batch( leer_archivo_de_texto(path) )


    elif path.is_dir():
        procesar_batch( leer_directorio(path))
                
    else:
        print(f"Error: La ruta '{ruta}' no es un archivo ni un directorio válido.")
        sys.exit(1)
   
       
def leer_archivo_de_texto(ruta):
    archivos = []
    with open(ruta, "r") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            archivo_path = Path(linea)
            if archivo_path.is_file() and archivo_path.suffix.lower() in MEDIA_EXTS:
                archivos.append(archivo_path)
    return archivos

def leer_directorio(ruta):
    archivos = []
    path = Path(ruta)
    for archivo in path.iterdir():
        if archivo.is_file() and archivo.suffix.lower() in MEDIA_EXTS:
            archivos.append(archivo)
    return archivos

def procesar_archivo(ruta):
    # TODO : revisar que el contenido no exista ya en la carpeta de descargas, o que no se haya descargado antes
    with YoutubeDL(OPTS) as ydl:
        info = ydl.extract_info(ruta.as_uri(), download=True)

def procesar_batch(batch):
    if batch:
        print(f"Cargando {len(batch)} archivos del batch:")
        for archivo in batch:
            procesar_archivo(archivo)


def validar_url(u: str) -> bool:
    # puede ser mucho mas estyricta
    u = u.strip()
    p = urlparse(u)
    return p.scheme in ("http", "https") and bool(p.netloc)

from yt_dlp import YoutubeDL

def procesar_fuente_remota(url):
    if not validar_url(url):
        print(f"Error: '{url}' no es una URL válida.")
        sys.exit(1)

    print(f"Descargando desde URL remota: {url}")

    with YoutubeDL(OPTS) as ydl:
        info = ydl.extract_info(url, download=True)

if __name__ == "__main__":
    main()