RERANKERS = {
    "mmarco": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
}

_modelo_cache: dict = {}


def _cargar_modelo(reranker_id: str):
    from sentence_transformers import CrossEncoder

    if reranker_id not in _modelo_cache:
        model_name = RERANKERS[reranker_id]
        print(f"[reranker] Cargando '{model_name}'...")
        _modelo_cache[reranker_id] = CrossEncoder(model_name)
    return _modelo_cache[reranker_id]


def rerank(query: str, filas: list[dict], reranker_id: str) -> list[dict]:
    if not filas:
        return filas

    modelo = _cargar_modelo(reranker_id)
    textos = [fila.get("texto", "") for fila in filas]
    pares = [(query, t) for t in textos]

    scores = modelo.predict(pares)

    for fila, score in zip(filas, scores):
        fila["score_rerank"] = float(score)

    return sorted(filas, key=lambda f: f["score_rerank"], reverse=True)
