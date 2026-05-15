import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from compartido.rutas import DESCARGAS_DIR


KREDOR_MODEL_ID = "kredor/punctuate-all"
KREDOR_CACHE_DIR = DESCARGAS_DIR / "modelos" / "kredor"
_KREDOR_CHUNK_WORDS = 180  # stay safely under the 512-token limit

_KREDOR_TOKENIZER = None
_KREDOR_MODEL = None


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

	word_ids = encoding.word_ids(0)
	etiquetas_por_palabra: dict = {}
	for token_idx, word_id in enumerate(word_ids):
		if word_id is None:
			continue
		if word_id not in etiquetas_por_palabra:
			etiquetas_por_palabra[word_id] = id2label[preds[token_idx]]

	return [etiquetas_por_palabra.get(i, "0") for i in range(len(palabras))]


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
