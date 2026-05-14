import argparse
import os
import sys

import torch

from compartido import json_utils as ju
from compartido.rutas import DESCARGAS_DIR


CACHE_DIR = DESCARGAS_DIR / "modelos" / "torch_hub"
_APPLY_TE = None


def main():
	parser = argparse.ArgumentParser(
		prog="corrector",
		description="Restaura puntuacion y mayusculas con Silero Text Enhancement.",
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
	transcripciones_path = folder / "transcripciones.json"

	data = ju.cargar_archivo(transcripciones_path)
	if not data:
		print(f"[ERROR] No se pudo cargar '{transcripciones_path}'.")
		sys.exit(1)

	texto = data.get("texto")
	if not texto:
		print(f"[ERROR] No se encontro texto para corregir en '{transcripciones_path}'.")
		sys.exit(1)

	apply_te = cargar_silero_te()

	with torch.inference_mode():
		texto_corregido = apply_te(texto, lan="es")

	print(f"[INFO] Correccion realizada para hash '{hash_id}'.")
	# print(f"[INFO] Texto original: {texto}")
	print(f"[INFO] Texto corregido: {texto_corregido}")

	# if ju.guardar_archivo(transcripciones_path, data):
	# 	print(f"[OK] Correccion guardada en '{transcripciones_path}'.")
	# 	return

	# print(f"[ERROR] No se pudo guardar '{transcripciones_path}'.")
	sys.exit(1)


def cargar_silero_te():
	global _APPLY_TE

	if _APPLY_TE is not None:
		return _APPLY_TE

	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	torch.hub.set_dir(str(CACHE_DIR))
	torch.set_grad_enabled(False)
	torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

	_, _, languages, _, apply_te = torch.hub.load(
		repo_or_dir="snakers4/silero-models",
		model="silero_te",
		trust_repo=True,
	)

	if "es" not in languages:
		raise RuntimeError(f"silero_te no reporta soporte para espanol: {languages}")

	_APPLY_TE = apply_te
	return _APPLY_TE




if __name__ == "__main__":
	main()
