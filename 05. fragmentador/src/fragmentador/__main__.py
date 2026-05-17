import argparse
import sys

from compartido import json_utils as ju
from compartido.rutas import DESCARGAS_DIR


def main():
	parser = argparse.ArgumentParser(
		prog="fragmentador",
		description="Fragmenta el texto corregido en chunks para su vectorizacion.",
	)
	parser.add_argument(
		"--hash",
		required=True,
		help="Hash del contenido dentro de descargas/",
	)

	args = parser.parse_args()
	procesar_hash(args.hash)


def procesar_hash(hash_id: str):
	folder = DESCARGAS_DIR / hash_id
	correcciones_path = folder / "correcciones.json"

	data = ju.cargar_archivo(correcciones_path)
	if not data:
		print(f"[ERROR] No se pudo cargar '{correcciones_path}'.")
		sys.exit(1)

	transcripciones = data.get("transcripciones")
	if not transcripciones:
		print(f"[ERROR] No se encontraron transcripciones en '{correcciones_path}'.")
		sys.exit(1)

	print(f"[OK] Correcciones cargadas desde '{correcciones_path}'.")
	print(f"[INFO] Modelo corrector: {data.get('modelo_corrector', 'desconocido')}")
	print(f"[INFO] Segmentos disponibles: {len(transcripciones)}")


if __name__ == "__main__":
	main()
