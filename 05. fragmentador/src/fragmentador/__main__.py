import argparse
import sys

from compartido import json_utils as ju
from compartido.embedders import EMBEDDERS, Sizer, get_spec, listar_ids
from compartido.rutas import DESCARGAS_DIR
from compartido.utils import crear_perfil_hardware

from fragmentador.tamano_fijo import fragmentar as fragmentar_fijo
from fragmentador.semantico import fragmentar as fragmentar_semantico


def main():
	parser = argparse.ArgumentParser(
		prog="fragmentador",
		description="Fragmenta el texto corregido en chunks para vectorizacion.",
	)
	parser.add_argument(
		"--hash",
		required=True,
		action="append",
		dest="hashes",
		metavar="HASH",
		help="Hash a fragmentar. Repetir para procesar varios en un solo comando.",
	)
	parser.add_argument(
		"--embedder",
		required=True,
		choices=listar_ids(),
		help="Modelo de embedding objetivo. Determina el tokenizador y el tamano de chunk.",
	)
	parser.add_argument(
		"--chunk-tokens",
		type=int,
		default=None,
		dest="chunk_tokens",
		help="Tamano maximo del chunk en tokens REALES del modelo. Default: el recomendado por el modelo.",
	)
	parser.add_argument(
		"--estrategia",
		choices=["tamano_fijo", "semantico"],
		default="tamano_fijo",
	)
	# tamano_fijo
	parser.add_argument(
		"--overlap",
		type=int,
		default=None,
		dest="overlap_pct",
		metavar="PCT",
		help="Porcentaje de chunk_tokens que se solapa entre chunks (0-50). Default 20. Solo tamano_fijo.",
	)
	# semantico
	parser.add_argument(
		"--umbral",
		type=float,
		default=None,
		help="Umbral de similitud coseno para corte semantico (0-1). Default 0.5. Solo semantico.",
	)
	parser.add_argument(
		"--min-tokens",
		type=int,
		default=None,
		dest="min_tokens",
		help="Tamano minimo de chunk; los menores se fusionan con vecinos. Default 64. Solo semantico.",
	)
	parser.add_argument(
		"--boundary-embedder",
		choices=listar_ids(),
		default=None,
		dest="boundary_embedder",
		help="Modelo para detectar bordes en estrategia semantica. Default: el mismo que --embedder.",
	)

	args = parser.parse_args()

	if args.estrategia == "tamano_fijo":
		if args.umbral is not None:
			parser.error("--umbral solo es valido con --estrategia semantico")
		if args.min_tokens is not None:
			parser.error("--min-tokens solo es valido con --estrategia semantico")
		if args.boundary_embedder is not None:
			parser.error("--boundary-embedder solo es valido con --estrategia semantico")
	else:  # semantico
		if args.overlap_pct is not None:
			parser.error("--overlap solo es valido con --estrategia tamano_fijo")

	# Defaults segun estrategia
	if args.overlap_pct is None:
		args.overlap_pct = 20
	if args.umbral is None:
		args.umbral = 0.5
	if args.min_tokens is None:
		args.min_tokens = 64

	procesar(
		hashes=args.hashes,
		embedder=args.embedder,
		chunk_tokens=args.chunk_tokens,
		estrategia=args.estrategia,
		overlap_pct=args.overlap_pct,
		umbral=args.umbral,
		min_tokens=args.min_tokens,
		boundary_embedder=args.boundary_embedder,
	)


def procesar(
	hashes: list[str],
	embedder: str,
	chunk_tokens: int | None = None,
	estrategia: str = "tamano_fijo",
	overlap_pct: int = 20,
	umbral: float = 0.5,
	min_tokens: int = 64,
	boundary_embedder: str | None = None,
):
	spec = get_spec(embedder)
	sizer = Sizer(embedder, chunk_tokens=chunk_tokens)

	print(f"[INFO] Estrategia: {estrategia}")
	print(f"[INFO] Embedder objetivo: {spec.id_corto} ({spec.hf_id})")
	print(f"[INFO] Chunk maximo: {sizer.chunk_max} tokens reales (max_seq_len={spec.max_seq_len})")

	for hash_id in hashes:
		_procesar_hash(
			hash_id,
			spec=spec,
			sizer=sizer,
			embedder=embedder,
			estrategia=estrategia,
			overlap_pct=overlap_pct,
			umbral=umbral,
			min_tokens=min_tokens,
			boundary_embedder=boundary_embedder,
		)


def _procesar_hash(
	hash_id: str,
	spec,
	sizer: "Sizer",
	embedder: str,
	estrategia: str = "tamano_fijo",
	overlap_pct: int = 20,
	umbral: float = 0.5,
	min_tokens: int = 64,
	boundary_embedder: str | None = None,
):
	folder = DESCARGAS_DIR / hash_id
	correcciones_path = folder / "correcciones.json"

	data = ju.cargar_archivo(correcciones_path)
	if not data:
		print(f"[ERROR] No se pudo cargar '{correcciones_path}'.")
		sys.exit(1)

	transcripciones = data.get("transcripciones")
	if not transcripciones:
		print(f"[ERROR] No se encontraron transcripciones en '{correcciones_path}'.")
		sys.exit(1)

	print(f"[OK] Correcciones cargadas desde '{correcciones_path}' (hash={hash_id})")
	print(f"[INFO] Modelo corrector: {data.get('modelo_corrector', 'desconocido')}")
	print(f"[INFO] Segmentos disponibles: {len(transcripciones)}")

	resultado_base = {
		"estrategia": estrategia,
		"embedder_objetivo": spec.id_corto,
		"embedder_hf_id": spec.hf_id,
		"embedder_max_seq_len": spec.max_seq_len,
		"embedder_dim": spec.dim,
		"chunk_tokens": sizer.chunk_max,
		"tokenizer_real": True,
	}

	if estrategia == "tamano_fijo":
		print(f"[INFO] Overlap: {overlap_pct}% (~{int(sizer.chunk_max * overlap_pct / 100)} tokens)")
		fragmentos, overlap_tokens = fragmentar_fijo(
			transcripciones, sizer=sizer, overlap_pct=overlap_pct
		)
		tiempo_total = round(fragmentar_fijo.elapsed, 2)
		resultado = {
			**resultado_base,
			"overlap_pct": overlap_pct,
			"overlap_tokens": overlap_tokens,
			"tiempo_fragmentacion": tiempo_total,
		}
	else:  # semantico
		perfil = crear_perfil_hardware()
		device = perfil["device"]
		boundary_id = boundary_embedder or embedder
		print(f"[INFO] Modelo de boundary: {boundary_id} (device={device})")
		print(f"[INFO] Umbral de corte: {umbral} | min_tokens={min_tokens}")
		out = fragmentar_semantico(
			transcripciones,
			sizer=sizer,
			umbral=umbral,
			min_tokens=min_tokens,
			boundary_embedder=boundary_id,
			device=device,
		)
		fragmentos = out["fragmentos"]
		tiempo_total = round(fragmentar_semantico.elapsed, 2)
		resultado = {
			**resultado_base,
			"boundary_embedder": boundary_id,
			"boundary_hf_id": out["boundary_hf_id"],
			"umbral": umbral,
			"min_tokens": min_tokens,
			"tiempo_fragmentacion": tiempo_total,
			"tiempo_carga_modelo": out["tiempos"]["carga_modelo"],
			"tiempo_inferencia": out["tiempos"]["inferencia"],
		}

	resultado.update({
		"num_fragmentos": len(fragmentos),
		"total_tokens": sum(f["num_tokens"] for f in fragmentos),
		"total_caracteres": sum(f["num_caracteres"] for f in fragmentos),
		"total_palabras": sum(f["num_palabras"] for f in fragmentos),
		"fragmentos": fragmentos,
	})

	fragmentos_path = folder / "fragmentos.json"
	if ju.guardar_archivo(fragmentos_path, resultado):
		print(f"[OK] Fragmentos guardados en '{fragmentos_path}'.")
		return

	print(f"[ERROR] No se pudo guardar '{fragmentos_path}'.")
	sys.exit(1)


if __name__ == "__main__":
	main()
