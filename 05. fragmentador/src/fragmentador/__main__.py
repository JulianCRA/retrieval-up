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
	parser.add_argument(
		"--overlap",
		type=int,
		default=20,
		dest="overlap_pct",
		metavar="PCT",
		help="Porcentaje de max_tokens que se solapa entre chunks consecutivos, 0-50 (default: 20).",
	)

	args = parser.parse_args()
	procesar_hash(args.hash, max_tokens=args.max_tokens, overlap_pct=args.overlap_pct)


def procesar_hash(hash_id: str, max_tokens: int = 512, overlap_pct: int = 10):
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
	print(f"[INFO] Overlap: {overlap_pct}% (~{int(max_tokens * overlap_pct / 100)} tokens)")

	fragmentos, overlap_tokens_aprox = fragmentar(
		transcripciones, max_tokens=max_tokens, overlap_pct=overlap_pct
	)
	tiempo_fragmentacion = round(fragmentar.elapsed, 2)

	total_caracteres = sum(f["num_caracteres"] for f in fragmentos)
	total_palabras = sum(f["num_palabras"] for f in fragmentos)

	resultado = {
		"estrategia": "tamano_fijo",
		"max_tokens": max_tokens,
		"overlap_tokens_aprox": overlap_tokens_aprox,
		"tiempo_fragmentacion": tiempo_fragmentacion,
		"num_fragmentos": len(fragmentos),
		"total_caracteres": total_caracteres,
		"total_palabras": total_palabras,
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
