import argparse
import sys
import time

import numpy as np

from compartido import json_utils as ju
from compartido.rutas import DESCARGAS_DIR
from compartido.utils import crear_perfil_hardware, cronometrar

from vectorizador.prefijos import prefijos_para


MODELOS_DIR = DESCARGAS_DIR / "modelos" / "embeddings"
MODELOS_DIR.mkdir(parents=True, exist_ok=True)


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
		"--modelo",
		default=None,
		help="HF id del modelo de embeddings. Por defecto se toma 'embedder_hf_id' de fragmentos.json.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=16,
		dest="batch_size",
		help="Tamano de batch para encode (default: 16).",
	)
	parser.add_argument(
		"--device",
		default=None,
		help="Dispositivo torch (cpu, cuda, mps). Por defecto se autodetecta.",
	)
	parser.add_argument(
		"--normalizar",
		action="store_true",
		default=True,
		help="Normalizar embeddings a norma 1 (default: True; permite usar producto punto como coseno).",
	)

	args = parser.parse_args()

	procesar_hash(
		args.hash,
		modelo=args.modelo,
		batch_size=args.batch_size,
		device=args.device,
		normalizar=args.normalizar,
	)


@cronometrar(etiqueta="Vectorizacion total")
def procesar_hash(
	hash_id: str,
	modelo: str | None = None,
	batch_size: int = 16,
	device: str | None = None,
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

	hf_id = modelo or data.get("embedder_hf_id")
	if not hf_id:
		print("[ERROR] No se especifico --modelo y fragmentos.json no trae 'embedder_hf_id'.")
		sys.exit(1)

	if device is None:
		device = crear_perfil_hardware()["device"]

	prefijo_passage, prefijo_query = prefijos_para(hf_id)

	print(f"[OK] Fragmentos cargados desde '{fragmentos_path}'.")
	print(f"[INFO] Modelo: {hf_id}")
	print(f"[INFO] Device: {device} | batch_size: {batch_size} | normalizar: {normalizar}")
	print(f"[INFO] Fragmentos a vectorizar: {len(fragmentos)}")
	if prefijo_passage:
		print(f"[INFO] Prefijo passage: {prefijo_passage!r}")

	textos = [prefijo_passage + f["texto"] for f in fragmentos]

	# Carga del modelo
	from sentence_transformers import SentenceTransformer  # import perezoso

	t0 = time.perf_counter()
	st_kwargs: dict = {"cache_folder": str(MODELOS_DIR), "device": device}
	# jina-v3 requiere trust_remote_code; tolerar otros sin esa necesidad
	if "jina" in hf_id.lower():
		st_kwargs["trust_remote_code"] = True
	model = SentenceTransformer(hf_id, **st_kwargs)
	tiempo_carga = round(time.perf_counter() - t0, 2)
	print(f"[TIEMPO] Carga modelo: {tiempo_carga:.2f}s")

	# Inferencia
	t0 = time.perf_counter()
	embeddings = model.encode(
		textos,
		batch_size=batch_size,
		show_progress_bar=True,
		convert_to_numpy=True,
		normalize_embeddings=normalizar,
	).astype(np.float32)
	tiempo_inferencia = round(time.perf_counter() - t0, 2)
	print(f"[TIEMPO] Inferencia ({len(textos)} fragmentos): {tiempo_inferencia:.2f}s")

	dim = int(embeddings.shape[1])

	# Guardado binario (vectores.npz)
	vectores_npz = folder / "vectores.npz"
	np.savez_compressed(
		vectores_npz,
		embeddings=embeddings,
		chunk_idx=np.arange(len(fragmentos), dtype=np.int32),
	)
	print(f"[OK] Embeddings guardados en '{vectores_npz}' ({embeddings.nbytes / 1024:.1f} KB en RAM).")

	# Metadatos (vectores.json)
	resultado_meta = {
		"modelo": hf_id,
		"dim": dim,
		"num_vectores": len(fragmentos),
		"normalizado": normalizar,
		"prefijo_passage": prefijo_passage,
		"prefijo_query": prefijo_query,
		"device": device,
		"batch_size": batch_size,
		"tiempo_carga_modelo": tiempo_carga,
		"tiempo_inferencia": tiempo_inferencia,
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
