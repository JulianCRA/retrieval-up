import argparse
import sys

import torch

from compartido import json_utils as ju
from compartido.rutas import DESCARGAS_DIR
from corrector.alinear import alinear_segmentos
from corrector.kredor import corregir_kredor
from corrector.silero import cargar_silero_te


def main():
	parser = argparse.ArgumentParser(
		prog="corrector",
		description="Restaura puntuacion y mayusculas.",
	)
	parser.add_argument(
		"--hash",
		required=True,
		help="Hash del contenido dentro de descargas/",
	)
	parser.add_argument(
		"--m",
		choices=["silero", "kredor"],
		default="silero",
		help="Motor de puntuacion: silero (default) o kredor/punctuate-all.",
	)

	args = parser.parse_args()
	procesar_hash(args.hash, backend=args.m)


def procesar_hash(hash_id: str, backend: str = "silero"):
	folder = DESCARGAS_DIR / hash_id
	transcripciones_path = folder / "transcripciones.json"

	data = ju.cargar_archivo(transcripciones_path)
	if not data:
		print(f"[ERROR] No se pudo cargar '{transcripciones_path}'.")
		sys.exit(1)

	texto = data.get("texto")
	if not texto:
		print(f"[ERROR] No se encontro 'texto' en '{transcripciones_path}'.")
		sys.exit(1)

	segmentos = data.get("transcripciones")
	if not segmentos:
		print(f"[ERROR] No se encontraron transcripciones en '{transcripciones_path}'.")
		sys.exit(1)

	if backend == "kredor":
		texto_corregido = corregir_kredor(texto)
	else:
		apply_te = cargar_silero_te()
		with torch.inference_mode():
			texto_corregido = apply_te(texto, lan="es")

	print(f"[INFO] Correccion realizada para hash '{hash_id}' (backend={backend}).")

	data["texto_corregido"] = texto_corregido
	data["transcripciones"] = alinear_segmentos(segmentos, texto_corregido)

	if ju.guardar_archivo(transcripciones_path, data):
		print(f"[OK] Correccion guardada en '{transcripciones_path}'.")
		return

	print(f"[ERROR] No se pudo guardar '{transcripciones_path}'.")
	sys.exit(1)


if __name__ == "__main__":
	main()
