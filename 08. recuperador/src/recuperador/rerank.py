from compartido.utils import cronometrar, medir

RERANKERS = {
    "mmarco": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "bge": "BAAI/bge-reranker-v2-m3",
    "jina": "jinaai/jina-reranker-v2-base-multilingual",
}

_TRUST_REMOTE_CODE = {"jina"}

_modelo_cache: dict = {}


@cronometrar(etiqueta="carga_reranker")
def _cargar_modelo(reranker_id: str, device: str = "cpu"):
    from sentence_transformers import CrossEncoder

    cache_key = (reranker_id, device)
    if cache_key not in _modelo_cache:
        model_name = RERANKERS[reranker_id]
        print(f"[reranker] Cargando '{model_name}' en {device}...")
        trust = reranker_id in _TRUST_REMOTE_CODE
        _modelo_cache[cache_key] = CrossEncoder(model_name, device=device, trust_remote_code=trust)
    return _modelo_cache[cache_key]


@cronometrar(etiqueta="rerank")
def rerank(query: str, filas: list[dict], reranker_id: str, device: str = "cpu") -> list[dict]:
    if not filas:
        return filas

    modelo = _cargar_modelo(reranker_id, device=device)
    textos = [fila.get("texto", "") for fila in filas]
    pares = [(query, t) for t in textos]

    with medir("inferencia_reranker"):
        scores = modelo.predict(pares)

    for fila, score in zip(filas, scores):
        fila["score_rerank"] = float(score)

    return sorted(filas, key=lambda f: f["score_rerank"], reverse=True)
