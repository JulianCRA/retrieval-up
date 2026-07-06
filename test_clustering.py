"""
Test HDBSCAN clustering quality across all embedders.

Queries are grouped into 4 semantic groups:
  A - IP routing & forwarding tables (5)
  B - Game design & mechanics (4)
  C - UX / Lean UX (3)
  D - Binary trees & data structures (3)

Run:
  python test_clustering.py            # embed all, cache, cluster, print
  python test_clustering.py --no-cache # ignore cache and re-embed
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import hdbscan

sys.path.insert(0, "compartido/src")
from compartido.embedders import cargar_sentence_transformer, get_spec, listar_ids

CACHE_FILE = Path("_test_clustering_cache.json")

QUERIES = [
    # A - routing
    ("A", "tabla de rutas mascara mas especifica primero"),
    ("A", "rutas se ordenan por especificidad mascara 27 26 25"),
    ("A", "si dos reglas coinciden con la misma IP cual se evalua primero"),
    ("A", "entrega directa salto intermedio tabla reenvio"),
    ("A", "ruta nivel 2 ruta nivel 3 gateway alcanzable por nivel 2"),
    # B - game design
    ("B", "mecanicas de juego principales tipos y ejemplos"),
    ("B", "como se define una meta de diseno en game design"),
    ("B", "constraints de diseno en videojuegos que son"),
    ("B", "monetizacion freemium modelos de negocio juegos"),
    # C - UX
    ("C", "que es lean ux diferencia con ux tradicional"),
    ("C", "como se hacen experimentos en lean ux"),
    ("C", "prototipado rapido y feedback en diseño de experiencia"),
    # D - data structures
    ("D", "arboles binarios de busqueda insercion y eliminacion"),
    ("D", "como funciona un arbol AVL rotaciones"),
    ("D", "recorrido en profundidad vs anchura arboles"),
]

EMBEDDERS = listar_ids()


def _normalizar(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def embed_all(force: bool = False) -> dict[str, np.ndarray]:
    """Returns {embedder_id: matrix (15, dim)} loading from cache when possible."""
    cached: dict = {}
    if CACHE_FILE.exists() and not force:
        raw = json.loads(CACHE_FILE.read_text())
        cached = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
        missing = [e for e in EMBEDDERS if e not in cached]
        if not missing:
            print(f"[cache] loaded all {len(EMBEDDERS)} embedders from {CACHE_FILE}")
            return cached
        print(f"[cache] found {list(cached.keys())}, need to embed: {missing}")
        to_embed = missing
    else:
        to_embed = list(EMBEDDERS)

    texts_raw = [q for _, q in QUERIES]

    for emb_id in to_embed:
        print(f"\n[embed] {emb_id} ...", flush=True)
        spec = get_spec(emb_id)
        model = cargar_sentence_transformer(emb_id, device="cpu")

        # Use text-matching task or no task — NOT retrieval.query
        # retrieval.query spreads queries apart (they should match passages, not each other)
        kwargs: dict = {"normalize_embeddings": True}
        if spec.tarea_passage:
            # Use the passage task as a symmetric similarity task
            # (or no task at all — just plain encode)
            pass  # intentionally no task for clustering

        if spec.prefijo_query:
            texts = [spec.prefijo_query + t for t in texts_raw]
        else:
            texts = texts_raw

        vecs = model.encode(texts, **kwargs)
        mat = _normalizar(np.array(vecs, dtype=np.float32))
        cached[emb_id] = mat
        print(f"[embed] {emb_id} done, shape={mat.shape}")

    # Save to cache
    to_save = {k: v.tolist() for k, v in cached.items()}
    CACHE_FILE.write_text(json.dumps(to_save))
    print(f"\n[cache] saved to {CACHE_FILE}")
    return cached


def run_clustering(matrices: dict[str, np.ndarray]) -> None:
    labels_true = [g for g, _ in QUERIES]
    texts = [q for _, q in QUERIES]

    for emb_id, mat in matrices.items():
        print(f"\n{'='*60}")
        print(f"Embedder: {emb_id}  shape={mat.shape}")

        # Pairwise cosine sims
        sims = mat @ mat.T
        np.fill_diagonal(sims, 0)
        print(f"Cosine sim — min: {sims.min():.3f}  max: {sims.max():.3f}  mean: {sims[sims > 0].mean():.3f}")

        for method in ("leaf", "eom"):
            for mcs in (2, 3):
                cl = hdbscan.HDBSCAN(
                    min_cluster_size=mcs,
                    metric="euclidean",
                    cluster_selection_method=method,
                )
                pred = cl.fit_predict(mat.astype(np.float64))
                n_groups = len(set(pred) - {-1})
                n_noise = (pred == -1).sum()
                print(f"  [{method:<4} mcs={mcs}] {n_groups} grupos, {n_noise} ruido  labels={pred.tolist()}")

        # Show best split (leaf mcs=3) with group membership
        cl = hdbscan.HDBSCAN(min_cluster_size=3, metric="euclidean", cluster_selection_method="leaf")
        pred = cl.fit_predict(mat.astype(np.float64))
        groups: dict[int, list] = {}
        for i, lab in enumerate(pred):
            groups.setdefault(int(lab), []).append((labels_true[i], texts[i]))
        print()
        for lab in sorted(groups):
            tag = "RUIDO" if lab == -1 else f"Grupo {lab}"
            print(f"  {tag}:")
            for true_g, txt in groups[lab]:
                print(f"    [{true_g}] {txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    matrices = embed_all(force=args.no_cache)
    run_clustering(matrices)
