"""Prefijos passage/query por modelo de embedding.

Mantenemos un mini-registro local con los prefijos que cada modelo espera al
indexar pasajes. Sin estos prefijos, modelos como E5 o BGE pierden mucha
calidad. La clave es el `hf_id` tal como lo guarda el fragmentador en
`fragmentos.json`.

Si el modelo no esta listado, asumimos sin prefijo (caso comun en MiniLM/Qwen).
"""

PREFIJOS: dict[str, dict[str, str]] = {
	"intfloat/multilingual-e5-large-instruct": {
		"passage": "passage: ",
		"query": "query: ",
	},
	"intfloat/multilingual-e5-large": {
		"passage": "passage: ",
		"query": "query: ",
	},
	"BAAI/bge-m3": {
		"passage": "",
		"query": "",
	},
	"ibm-granite/granite-embedding-107m-multilingual": {
		"passage": "",
		"query": "",
	},
	"jinaai/jina-embeddings-v3": {
		"passage": "",
		"query": "",
	},
	"Qwen/Qwen3-Embedding-0.6B": {
		"passage": "",
		"query": "",
	},
}


def prefijos_para(hf_id: str) -> tuple[str, str]:
	"""Devuelve (passage, query) para el modelo. Vacios si no hay registro."""
	p = PREFIJOS.get(hf_id, {"passage": "", "query": ""})
	return p.get("passage", ""), p.get("query", "")
