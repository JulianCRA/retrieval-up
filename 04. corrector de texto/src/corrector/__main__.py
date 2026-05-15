import argparse
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from compartido import json_utils as ju
from compartido.rutas import DESCARGAS_DIR
from corrector.alinear import alinear_segmentos


CACHE_DIR = DESCARGAS_DIR / "modelos" / "torch_hub"
_APPLY_TE = None

# ── kredor/punctuate-all ──────────────────────────────────────────────────────
_KREDOR_TOKENIZER = None
_KREDOR_MODEL = None
KREDOR_MODEL_ID = "kredor/punctuate-all"
KREDOR_CACHE_DIR = DESCARGAS_DIR / "modelos" / "kredor"
_KREDOR_CHUNK_WORDS = 180  # stay safely under the 512-token limit


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
	procesar_hash(args.hash, backend=args.backend)


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


def cargar_kredor():
	global _KREDOR_TOKENIZER, _KREDOR_MODEL

	if _KREDOR_MODEL is not None:
		return _KREDOR_TOKENIZER, _KREDOR_MODEL

	print(f"[INFO] Cargando '{KREDOR_MODEL_ID}'...")
	KREDOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
	cache = str(KREDOR_CACHE_DIR)

	_KREDOR_TOKENIZER = AutoTokenizer.from_pretrained(KREDOR_MODEL_ID, cache_dir=cache)
	_KREDOR_MODEL = AutoModelForTokenClassification.from_pretrained(KREDOR_MODEL_ID, cache_dir=cache)
	_KREDOR_MODEL.eval()

	return _KREDOR_TOKENIZER, _KREDOR_MODEL


def _predecir_etiquetas(tokenizer, model, palabras: list) -> list:
	"""Returns one punctuation label per word ('O', 'COMMA', 'PERIOD', etc.)."""
	encoding = tokenizer(
		palabras,
		is_split_into_words=True,
		return_tensors="pt",
		padding=True,
		truncation=True,
		max_length=512,
	)
	with torch.inference_mode():
		logits = model(**encoding).logits

	preds = logits.argmax(dim=-1)[0].tolist()
	id2label = model.config.id2label

	# Map the first subword token of each word to its word index
	word_ids = encoding.word_ids(0)
	etiquetas_por_palabra: dict = {}
	for token_idx, word_id in enumerate(word_ids):
		if word_id is None:
			continue
		if word_id not in etiquetas_por_palabra:
			etiquetas_por_palabra[word_id] = id2label[preds[token_idx]]

	return [etiquetas_por_palabra.get(i, "O") for i in range(len(palabras))]


def corregir_kredor(texto: str) -> str:
	tokenizer, model = cargar_kredor()
	palabras = texto.split()
	if not palabras:
		return texto

	etiquetas: list = []
	for i in range(0, len(palabras), _KREDOR_CHUNK_WORDS):
		chunk = palabras[i : i + _KREDOR_CHUNK_WORDS]
		etiquetas.extend(_predecir_etiquetas(tokenizer, model, chunk))

	partes = []
	for palabra, label in zip(palabras, etiquetas):
		punct = "" if label == "0" else label
		partes.append(palabra + punct)

	return " ".join(partes)


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
