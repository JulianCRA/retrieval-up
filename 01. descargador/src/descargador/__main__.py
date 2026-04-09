import argparse
import sys
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".wma", ".aac"}
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv"}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS

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
    print(f"Cargando archivo: {ruta}")

def procesar_batch(batch):
    if batch:
        print(f"Cargando {len(batch)} archivos del batch:")
        for archivo in batch:
            procesar_archivo(archivo)

if __name__ == "__main__":
    main()