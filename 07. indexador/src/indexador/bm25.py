"""Tokenizacion de texto para BM25 en espanol con spaCy.

Pipeline por token (solo los que pasan el filtro POS):
  1. Cargar es_core_news_lg (carga perezosa, una sola vez)
  2. Conservar solo NOUN, VERB, ADJ, PROPN
  3. Descartar stopwords y puntuacion (spaCy los marca nativamente)
  4. Lemmatizar (forma canonica: 'creamos' -> 'crear')
  5. Minusculas + quitar acentos
  6. Descartar tokens de menos de 2 caracteres tras la normalizacion
  7. Aplicar lista extra de muletillas ASR

El resultado es una lista de lemmas normalizados lista para serializar
en tokens_bm25.json y para construir el campo texto_bm25 del indice
LanceDB (tokens unidos por espacio, que es lo que indexa tantivy).
"""

import unicodedata

# POS tags que aportan valor semantico para BM25.
_POS_UTILES: frozenset[str] = frozenset({"NOUN", "VERB", "ADJ", "PROPN"})

# Muletillas y ruido tipico de ASR que spaCy no filtra como stopword.
EXTRA_STOPWORDS: frozenset[str] = frozenset({
	"pues", "bueno", "entonces", "osea", "verdad", "claro",
	"digamos", "tipo", "sino", "ahi", "asi", "aca", "alla",
})

_nlp = None


def _cargar_spacy():
	global _nlp
	if _nlp is not None:
		return
	import spacy
	_nlp = spacy.load(
		"es_core_news_lg",
		# Desactivar componentes innecesarios para acelerar el pipeline.
		disable=["parser", "ner"],
	)


def _normalizar(texto: str) -> str:
	"""Minusculas y quitar diacriticos (tildes, dieresis)."""
	texto = texto.lower()
	texto = unicodedata.normalize("NFD", texto)
	return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def tokenizar(texto: str) -> list[str]:
	"""Devuelve lista de lemmas normalizados para un fragmento."""
	_cargar_spacy()
	doc = _nlp(texto)
	tokens: list[str] = []
	for token in doc:
		if token.is_stop or token.is_punct or token.is_space:
			continue
		if token.pos_ not in _POS_UTILES:
			continue
		normalizado = _normalizar(token.lemma_)
		if len(normalizado) < 2 or normalizado in EXTRA_STOPWORDS:
			continue
		tokens.append(normalizado)
	return tokens


def tokens_a_texto(tokens: list[str]) -> str:
	"""Une los tokens con espacio para insertar en LanceDB/tantivy."""
	return " ".join(tokens)
