import argparse
import sys
from pathlib import Path

from descargador.__main__ import procesar_archivo

def main():
    parser = argparse.ArgumentParser(
        prog = "procesador",
        description = "Procesa archivos de audio para remover ruido de fondo, mejorar calidad, etc."
    )

    parser.add_argument(
        "-i", "--input",
        help = "Archivo de audio o directorio a procesar",
        required = True
    )

    args = parser.parse_args()

    if not (Path(args.input).is_file() and Path(args.input).suffix.lower() == ".wav"):
        print(f"Error: La ruta '{args.input}' no es un archivo WAV válido.")
        sys.exit(1)

    procesar_archivo(args.input)

def procesar_archivo(ruta):
    print(f"Procesando archivo: {ruta}")
    
if __name__ == "__main__":
    main()