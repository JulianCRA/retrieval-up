import argparse
import sys

from compartido import json_utils as ju
from compartido.rutas import DESCARGAS_DIR
from fragmentador.tamano_fijo import fragmentar as fragmentar_fijo
from fragmentador.semantico import fragmentar as fragmentar_semantico, MODELO_ID


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
		"--estrategia",
		choices=["tamano_fijo", "semantico"],
		default="tamano_fijo",
		help="Estrategia de fragmentacion (default: tamano_fijo).",
	)
	# --- tamano_fijo ---
	parser.add_argument(
		"--max-tokens",
		type=int,
		default=512,
		dest="max_tokens",
		help="Tamano maximo del chunk en tokens estimados (default: 512). Solo tamano_fijo y semantico (como limite de seguridad).",
	)
	parser.add_argument(
		"--overlap",
		type=int,
		default=20,
		dest="overlap_pct",
		metavar="PCT",
		help="Porcentaje de max_tokens que se solapa entre chunks consecutivos, 0-50 (default: 20). Solo tamano_fijo.",
	)
	# --- semantico ---
	parser.add_argument(
		"--umbral",
		type=float,
		default=0.5,
		help="Umbral de similitud coseno para corte semantico, 0-1 (default: 0.5). Solo semantico.",
	)
	parser.add_argument(
		"--min-tokens",
		type=int,
		default=64,
		dest="min_tokens",
		help="Tamano minimo de chunk; los mas pequenos se fusionan con el anterior (default: 64). Solo semantico.",
	)

	args = parser.parse_args()
	procesar_hash(
		args.hash,
		estrategia=args.estrategia,
		max_tokens=args.max_tokens,
		overlap_pct=args.overlap_pct,
		umbral=args.umbral,
		min_tokens=args.min_tokens,
	)


def procesar_hash(
	hash_id: str,
	estrategia: str = "tamano_fijo",
	max_tokens: int = 512,
	overlap_pct: int = 20,
	umbral: float = 0.5,
	min_tokens: int = 64,
):
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
	print(f"[INFO] Estrategia: {estrategia}")

	if estrategia == "tamano_fijo":
		print(f"[INFO] Tamano maximo de chunk: {max_tokens} tokens estimados")
		print(f"[INFO] Overlap: {overlap_pct}% (~{int(max_tokens * overlap_pct / 100)} tokens)")

		fragmentos, overlap_tokens_aprox = fragmentar_fijo(
			transcripciones, max_tokens=max_tokens, overlap_pct=overlap_pct
		)
		tiempo_fragmentacion = round(fragmentar_fijo.elapsed, 2)

		resultado = {
			"estrategia": "tamano_fijo",
			"max_tokens": max_tokens,
			"overlap_tokens_aprox": overlap_tokens_aprox,
			"tiempo_fragmentacion": tiempo_fragmentacion,
			"num_fragmentos": len(fragmentos),
			"total_caracteres": sum(f["num_caracteres"] for f in fragmentos),
			"total_palabras": sum(f["num_palabras"] for f in fragmentos),
			"fragmentos": fragmentos,
		}

	else:  # semantico
		print(f"[INFO] Modelo de embeddings: {MODELO_ID}")
		print(f"[INFO] Umbral de corte: {umbral}")
		print(f"[INFO] Min tokens por chunk: {min_tokens} | Max tokens: {max_tokens}")

		fragmentos = fragmentar_semantico(
			transcripciones, umbral=umbral, min_tokens=min_tokens, max_tokens=max_tokens
		)
		tiempo_fragmentacion = round(fragmentar_semantico.elapsed, 2)

		resultado = {
			"estrategia": "semantico",
			"modelo": MODELO_ID,
			"umbral": umbral,
			"min_tokens": min_tokens,
			"max_tokens": max_tokens,
			"tiempo_fragmentacion": tiempo_fragmentacion,
			"num_fragmentos": len(fragmentos),
			"total_caracteres": sum(f["num_caracteres"] for f in fragmentos),
			"total_palabras": sum(f["num_palabras"] for f in fragmentos),
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
