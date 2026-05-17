import argparse
import sys

from compartido import json_utils as ju
from compartido.rutas import DESCARGAS_DIR
from fragmentador.tamano_fijo import fragmentar


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
	parser.add_argument(
		"--max-tokens",
		type=int,
		default=512,
		dest="max_tokens",
		help="Tamano maximo del chunk en tokens estimados (default: 512). En espanol, y a grandes rasgos, 1 token equicale a 3.5 caracteres, o a 0.75 palabras.",
	)

	args = parser.parse_args()
	procesar_hash(args.hash, max_tokens=args.max_tokens)


def procesar_hash(hash_id: str, max_tokens: int = 512):
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
	print(f"[INFO] Tamano maximo de chunk: {max_tokens} tokens estimados")

	fragmentos = fragmentar(transcripciones, max_tokens=max_tokens)
	tiempo_fragmentacion = round(fragmentar.elapsed, 2)

	resultado = {
		"estrategia": "tamano_fijo",
		"max_tokens": max_tokens,
		"tiempo_fragmentacion": tiempo_fragmentacion,
		"num_fragmentos": len(fragmentos),
		"fragmentos": fragmentos,
	}

	fragmentos_path = folder / "fragmentos.json"
	if ju.guardar_archivo(fragmentos_path, resultado):
		print(f"[OK] Fragmentos guardados en '{fragmentos_path}'.")
		return

	print(f"[ERROR] No se pudo guardar '{fragmentos_path}'.")
	sys.exit(1)


if __name__ == "__main__":
	main()
