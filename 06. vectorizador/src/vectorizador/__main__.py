import argparse
import sys

import numpy as np

from compartido import json_utils as ju
from compartido.rutas import DESCARGAS_DIR
from compartido.utils import crear_perfil_hardware, cronometrar, cronometro_activo, medir
from compartido.embedders import (
	cargar_sentence_transformer,
	get_spec,
	listar_ids,
)


@cronometrar(etiqueta="carga_modelo")
def cargar_modelo(embedder_id: str, device: str):
	return cargar_sentence_transformer(embedder_id, device=device)


@cronometrar(etiqueta="inferencia")
def vectorizar_textos(
	model,
	textos: list[str],
	batch_size: int,
	normalizar: bool,
	tarea: str = "",
) -> np.ndarray:
	kwargs: dict = {
		"batch_size": batch_size,
		"show_progress_bar": True,
		"convert_to_numpy": True,
		"normalize_embeddings": normalizar,
	}
	if tarea:
		kwargs["task"] = tarea
	return model.encode(textos, **kwargs).astype(np.float32)


def main():
	parser = argparse.ArgumentParser(
		prog="vectorizador",
		description="Calcula embeddings densos por fragmento.",
	)
	parser.add_argument(
		"--hash",
		required=True,
		action="append",
		dest="hashes",
		metavar="HASH",
		help="Hash a vectorizar. Repetir para procesar varios en un solo comando.",
	)
	parser.add_argument(
		"--embedder",
		default=None,
		choices=listar_ids(),
		help=(
			"Id corto del modelo de embeddings. Por defecto se toma 'embedder_objetivo' de fragmentos.json. "
			"Todos los hashes de un mismo comando deben compartir el mismo embedder."
		),
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
	parser.add_argument(
		"--forzar-cpu",
		action="store_true",
		dest="forzar_cpu",
		help="Forzar uso de CPU aunque haya GPU disponible.",
	)

	args = parser.parse_args()

	procesar(
		hashes=args.hashes,
		embedder=args.embedder,
		batch_size=args.batch_size,
		normalizar=args.normalizar,
		forzar_cpu=args.forzar_cpu,
	)


def procesar(
	hashes: list[str],
	embedder: str | None = None,
	batch_size: int = 16,
	normalizar: bool = True,
	forzar_cpu: bool = False,
):
	# Determinar embedder desde el primer hash si no se paso por CLI.
	embedder_id = embedder
	if not embedder_id:
		first_data = ju.cargar_archivo(DESCARGAS_DIR / hashes[0] / "fragmentos.json")
		if first_data:
			embedder_id = first_data.get("embedder_objetivo")
	if not embedder_id:
		print("[ERROR] No se especifico --embedder y fragmentos.json no trae 'embedder_objetivo'.")
		sys.exit(1)

	spec = get_spec(embedder_id)
	forzado = {"device": "cpu"} if forzar_cpu else None
	device = crear_perfil_hardware(forzado=forzado)["device"]

	print(f"[INFO] Embedder: {spec.id_corto} ({spec.hf_id})")
	print(f"[INFO] Dim: {spec.dim} | max_seq_len: {spec.max_seq_len}")
	print(f"[INFO] Device: {device} | batch_size: {batch_size} | normalizar: {normalizar}")

	model = cargar_modelo(embedder_id, device)

	total = len(hashes)
	for i, hash_id in enumerate(hashes, 1):
		print(f"\n[PIPELINE] Vectorizando recurso {i} de {total}")
		_procesar_hash(
			hash_id,
			model=model,
			spec=spec,
			embedder_id=embedder_id,
			batch_size=batch_size,
			normalizar=normalizar,
			device=device,
		)


def _procesar_hash(
	hash_id: str,
	model,
	spec,
	embedder_id: str,
	batch_size: int = 16,
	normalizar: bool = True,
	device: str = "cpu",
):
	folder = DESCARGAS_DIR / hash_id
	fragmentos_path = folder / "fragmentos.json"

	with cronometro_activo() as crono:
		with medir("lectura_fragmentos"):
			data = ju.cargar_archivo(fragmentos_path)
		if not data:
			print(f"[ERROR] No se pudo cargar '{fragmentos_path}'.")
			sys.exit(1)

		fragmentos = data.get("fragmentos")
		if not fragmentos:
			print(f"[AVISO] No hay fragmentos en '{fragmentos_path}' (audio sin voz detectada). Saltando vectorización.")
			return

		print(f"[OK] Fragmentos cargados desde '{fragmentos_path}' (hash={hash_id})")
		print(f"[INFO] Fragmentos a vectorizar: {len(fragmentos)}")
		if spec.prefijo_passage:
			print(f"[INFO] Prefijo passage: {spec.prefijo_passage!r}")
		if spec.tarea_passage:
			print(f"[INFO] Tarea passage: {spec.tarea_passage!r}")

		textos = [spec.prefijo_passage + f["texto"] for f in fragmentos]

		embeddings = vectorizar_textos(model, textos, batch_size, normalizar, tarea=spec.tarea_passage)

		dim = int(embeddings.shape[1])
		if dim != spec.dim:
			print(f"[ADVERTENCIA] Dim real ({dim}) != dim del registro ({spec.dim}).")

		vectores_npz = folder / "vectores.npz"
		with medir("escritura_npz"):
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
			"tarea_passage": spec.tarea_passage,
			"tarea_query": spec.tarea_query,
			"device": device,
			"batch_size": batch_size,
			"tiempos": crono.resumen(),
			"archivo_vectores": vectores_npz.name,
			"fragmentos_origen": fragmentos_path.name,
		}

		vectores_meta_path = folder / "vectores.json"
		if ju.guardar_archivo(vectores_meta_path, resultado_meta):
			print(f"[OK] Metadatos guardados en '{vectores_meta_path}'.")
			ju.guardar_nodo(folder / "info.json", "status", 6)
			ju.guardar_registro("status", 6, ruta=(hash_id,))
			return

		print(f"[ERROR] No se pudo guardar '{vectores_meta_path}'.")
		sys.exit(1)


if __name__ == "__main__":
	main()
