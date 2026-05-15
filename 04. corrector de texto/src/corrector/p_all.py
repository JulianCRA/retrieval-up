import spacy
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from compartido.rutas import DESCARGAS_DIR


MODEL_ID = "kredor/punctuate-all"
CACHE_DIR = DESCARGAS_DIR / "modelos" / "kredor"
SPACY_MODEL_ID = "es_core_news_sm"
_CHUNK_WORDS = 180  # stay safely under the 512-token limit

_TOKENIZER = None
_MODEL = None
_DEVICE = None
_SPACY_NLP = None


# ── punctuation ───────────────────────────────────────────────────────────────

def _cargar():
	global _TOKENIZER, _MODEL, _DEVICE

	if _MODEL is not None:
		return _TOKENIZER, _MODEL

	_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[INFO] Cargando '{MODEL_ID}' en {_DEVICE}...")
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	cache = str(CACHE_DIR)

	_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache)
	_MODEL = AutoModelForTokenClassification.from_pretrained(MODEL_ID, cache_dir=cache)
	_MODEL.to(_DEVICE)
	_MODEL.eval()

	return _TOKENIZER, _MODEL


def _predecir_etiquetas(tokenizer, model, palabras: list) -> list:
	encoding = tokenizer(
		palabras,
		is_split_into_words=True,
		return_tensors="pt",
		padding=True,
		truncation=True,
		max_length=512,
	)
	inputs = {k: v.to(_DEVICE) for k, v in encoding.items()}
	with torch.inference_mode():
		logits = model(**inputs).logits

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


def _puntuar(texto: str) -> str:
	tokenizer, model = _cargar()
	palabras = texto.split()
	if not palabras:
		return texto

	etiquetas: list = []
	for i in range(0, len(palabras), _CHUNK_WORDS):
		chunk = palabras[i : i + _CHUNK_WORDS]
		etiquetas.extend(_predecir_etiquetas(tokenizer, model, chunk))

	partes = []
	for palabra, label in zip(palabras, etiquetas):
		punct = "" if label == "0" else label
		partes.append(palabra + punct)

	return " ".join(partes)


# ── capitalization ────────────────────────────────────────────────────────────

def _cargar_spacy():
	global _SPACY_NLP

	if _SPACY_NLP is not None:
		return _SPACY_NLP

	print(f"[INFO] Cargando '{SPACY_MODEL_ID}'...")
	_SPACY_NLP = spacy.load(SPACY_MODEL_ID)
	return _SPACY_NLP


def _capitalizar(texto: str) -> str:
	nlp = _cargar_spacy()
	doc = nlp(texto)
	partes = []
	for token in doc:
		word = token.text
		if token.is_sent_start or token.pos_ == "PROPN" or token.ent_type_:
			word = word[0].upper() + word[1:] if word else word
		partes.append(word + token.whitespace_)
	return "".join(partes)


# ── public entry point ────────────────────────────────────────────────────────

def corregir_p_all(texto: str) -> str:
	return _capitalizar(_puntuar(texto))
