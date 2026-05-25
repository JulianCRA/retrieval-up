import argparse
import sys

import numpy as np

from compartido import json_utils as ju
from compartido.rutas import DESCARGAS_DIR
from compartido.utils import crear_perfil_hardware, cronometrar
from compartido.embedders import (
	cargar_sentence_transformer,
	get_spec,
	listar_ids,
)


@cronometrar(etiqueta="Carga modelo")
def cargar_modelo(embedder_id: str, device: str):
	return cargar_sentence_transformer(embedder_id, device=device)


@cronometrar(etiqueta="Inferencia embeddings")
def vectorizar_textos(
	model,
	textos: list[str],
	batch_size: int,
	normalizar: bool,
) -> np.ndarray:
	return model.encode(
		textos,
		batch_size=batch_size,
		show_progress_bar=True,
		convert_to_numpy=True,
		normalize_embeddings=normalizar,
	).astype(np.float32)


def main():
	parser = argparse.ArgumentParser(
		prog="vectorizador",
		description="Calcula embeddings densos por fragmento.",
	)
	parser.add_argument(
		"--hash",
		required=True,
		help="Hash del contenido dentro de descargas/",
	)
	parser.add_argument(
		"--embedder",
		default=None,
		choices=listar_ids(),
		help="Id corto del modelo de embeddings. Por defecto se toma 'embedder_objetivo' de fragmentos.json.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=16,
		dest="batch_size",
		help="Tamano de batch para encode (default: 16).",
	)
	parser.add_argument(
		"--sin-normalizar",
		action="store_false",
		dest="normalizar",
		help="No normalizar embeddings (por defecto se normalizan a norma 1).",
	)

	args = parser.parse_args()

	procesar_hash(
		args.hash,
		embedder=args.embedder,
		batch_size=args.batch_size,
		normalizar=args.normalizar,
	)


def procesar_hash(
	hash_id: str,
	embedder: str | None = None,
	batch_size: int = 16,
	normalizar: bool = True,
):
	folder = DESCARGAS_DIR / hash_id
	fragmentos_path = folder / "fragmentos.json"

	data = ju.cargar_archivo(fragmentos_path)
	if not data:
		print(f"[ERROR] No se pudo cargar '{fragmentos_path}'.")
		sys.exit(1)

	fragmentos = data.get("fragmentos")
	if not fragmentos:
		print(f"[ERROR] No hay fragmentos en '{fragmentos_path}'.")
		sys.exit(1)

	embedder_id = embedder or data.get("embedder_objetivo")
	if not embedder_id:
		print("[ERROR] No se especifico --embedder y fragmentos.json no trae 'embedder_objetivo'.")
		sys.exit(1)

	spec = get_spec(embedder_id)
	device = crear_perfil_hardware()["device"]

	print(f"[OK] Fragmentos cargados desde '{fragmentos_path}'.")
	print(f"[INFO] Embedder: {spec.id_corto} ({spec.hf_id})")
	print(f"[INFO] Dim: {spec.dim} | max_seq_len: {spec.max_seq_len}")
	print(f"[INFO] Device: {device} | batch_size: {batch_size} | normalizar: {normalizar}")
	print(f"[INFO] Fragmentos a vectorizar: {len(fragmentos)}")
	if spec.prefijo_passage:
		print(f"[INFO] Prefijo passage: {spec.prefijo_passage!r}")

	textos = [spec.prefijo_passage + f["texto"] for f in fragmentos]

	model = cargar_modelo(embedder_id, device)
	embeddings = vectorizar_textos(model, textos, batch_size, normalizar)

	dim = int(embeddings.shape[1])
	if dim != spec.dim:
		print(f"[ADVERTENCIA] Dim real ({dim}) != dim del registro ({spec.dim}).")

	vectores_npz = folder / "vectores.npz"
	np.savez_compressed(
		vectores_npz,
		embeddings=embeddings,
		chunk_idx=np.arange(len(fragmentos), dtype=np.int32),
	)
	print(f"[OK] Embeddings guardados en '{vectores_npz}' ({embeddings.nbytes / 1024:.1f} KB en RAM).")

	resultado_meta = {
		"embedder": spec.id_corto,
		"embedder_hf_id": spec.hf_id,
		"dim": dim,
		"num_vectores": len(fragmentos),
		"normalizado": normalizar,
		"prefijo_passage": spec.prefijo_passage,
		"prefijo_query": spec.prefijo_query,
		"device": device,
		"batch_size": batch_size,
		"tiempo_carga_modelo": round(cargar_modelo.elapsed, 2),
		"tiempo_inferencia": round(vectorizar_textos.elapsed, 2),
		"archivo_vectores": vectores_npz.name,
		"fragmentos_origen": fragmentos_path.name,
	}

	vectores_meta_path = folder / "vectores.json"
	if ju.guardar_archivo(vectores_meta_path, resultado_meta):
		print(f"[OK] Metadatos guardados en '{vectores_meta_path}'.")
		return

	print(f"[ERROR] No se pudo guardar '{vectores_meta_path}'.")
	sys.exit(1)


if __name__ == "__main__":
	main()
