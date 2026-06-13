import argparse
import json
import sys

import numpy as np

from compartido import json_utils as ju
import compartido.indice_utils as iu
from compartido.rutas import DESCARGAS_DIR, INDICE_DIR
from compartido.embedders import listar_ids
from compartido.utils import cronometrar, cronometro_activo, medir

import importlib

from compartido.bm25 import tokenizar, tokens_a_texto

def _parse_tag(valor: str) -> tuple[str, str]:
	"""Parsea 'CLAVE=VALOR' y aborta si el formato es incorrecto."""
	if "=" not in valor:
		raise argparse.ArgumentTypeError(
			f"El tag '{valor}' no tiene el formato CLAVE=VALOR."
		)
	clave, _, val = valor.partition("=")
	clave = clave.strip()
	if not clave:
		raise argparse.ArgumentTypeError(
			f"El tag '{valor}' tiene la clave vacia."
		)
	return clave, val


def main():
	parser = argparse.ArgumentParser(
		prog="indexador",
		description=(
			"Indexa fragmentos + embeddings en un indice hibrido. "
			"Soporta busqueda densa, BM25 y hibrida."
		),
	)

	# --- entrada ---
	parser.add_argument(
		"--hash",
		required=True,
		action="append",
		dest="hashes",
		metavar="HASH",
		help="Hash a indexar. Repetir para indexar varios en un solo comando.",
	)
	parser.add_argument(
		"--embedder",
		default=None,
		choices=listar_ids(),
		help=(
			"Id corto del embedder. Por defecto se lee 'embedder' de vectores.json. "
			"Todos los hashes de un mismo comando deben compartir el mismo embedder."
		),
	)

	# --- indice ---
	parser.add_argument(
		"--backend",
		default="lance",
		choices=["lance", "qdrant", "milvus"],
		help="Backend de indexacion (default: lance).",
	)
	parser.add_argument(
		"--db",
		default=str(INDICE_DIR),
		metavar="URI",
		help=(
			f"URI o ruta del backend. "
			f"Para 'lance': ruta al directorio local (default: {INDICE_DIR})."
		),
	)
	parser.add_argument(
		"--tabla",
		default=None,
		metavar="NOMBRE",
		help=(
			"Nombre de la tabla dentro de la base. "
			"Por defecto es el id corto del embedder (ej. 'granite-107m')."
		),
	)
	parser.add_argument(
		"--recrear",
		action="store_true",
		help="Eliminar y recrear la tabla si ya existe (util para reindexar desde cero).",
	)

	# --- metadata de usuario ---
	parser.add_argument(
		"--tag",
		action="append",
		type=_parse_tag,
		dest="tags",
		metavar="CLAVE=VALOR",
		help=(
			"Metadata adicional que se adjunta a todos los chunks del lote. "
			"Repetir para multiples tags: --tag tema=diseno --tag nivel=avanzado."
		),
	)

	args = parser.parse_args()

	# tags -> dict plano
	tags: dict[str, str] = dict(args.tags) if args.tags else {}

	procesar(
		hashes=args.hashes,
		embedder=args.embedder,
		db_ruta=args.db,
		tabla=args.tabla,
		reclear=args.recrear,
		tags=tags,
		backend=args.backend,
	)


def procesar(
	hashes: list[str],
	embedder: str | None,
	db_ruta: str,
	tabla: str | None,
	reclear: bool,
	tags: dict[str, str],
	backend: str = "lance",
):
	mod = importlib.import_module(f"indexador.{backend}")
	db = mod.abrir(db_ruta)
	for hash_id in hashes:
		_procesar_hash(
			hash_id,
			db=db,
			backend_mod=mod,
			embedder=embedder,
			tabla=tabla,
			reclear=reclear,
			tags=tags,
		)
		# Tras el primer hash, no recrear de nuevo en los siguientes.
		reclear = False


@cronometrar(etiqueta="tokenizacion_bm25")
def _tokenizar_fragmentos(fragmentos: list[dict]) -> list[list[str]]:
	return [tokenizar(f["texto"]) for f in fragmentos]


def _procesar_hash(
	hash_id: str,
	db,
	backend_mod,
	embedder: str | None,
	tabla: str | None,
	reclear: bool,
	tags: dict[str, str],
):
	folder = DESCARGAS_DIR / hash_id
	fragmentos_path = folder / "fragmentos.json"
	vectores_meta_path = folder / "vectores.json"
	vectores_npz_path = folder / "vectores.npz"
	info_path = folder / "info.json"

	tiempos_previos: dict = {}
	with cronometro_activo() as crono:
		with medir("lectura_fragmentos"):
			data = ju.cargar_archivo(fragmentos_path)
		if not data:
			print(f"[ERROR] No se pudo cargar '{fragmentos_path}'.")
			sys.exit(1)

		fragmentos = data.get("fragmentos")
		if not fragmentos:
			print(f"[ERROR] No hay fragmentos en '{fragmentos_path}'.")
			sys.exit(1)

		print(f"[OK] Fragmentos cargados: {len(fragmentos)} (hash={hash_id})")

		# --- Info de la fuente ---
		info = ju.cargar_archivo(info_path) or {}
		descarga = info.get("descarga") or {}
		titulo = info.get("title") or ""
		uri = descarga.get("uri") or ""
		fuente = descarga.get("fuente") or ""

		# --- Vectores ---
		meta_vec = ju.cargar_archivo(vectores_meta_path)
		if not meta_vec:
			print(f"[ERROR] No se pudo cargar '{vectores_meta_path}'.")
			sys.exit(1)

		# --- Tiempos de etapas anteriores ---
		_transcripciones_meta = ju.cargar_archivo(folder / "transcripciones.json")
		_correcciones_meta = ju.cargar_archivo(folder / "correcciones.json")
		tiempos_previos = {
			"descarga": (info.get("descarga") or {}).get("tiempos") or {},
			"procesamiento": (info.get("procesamiento") or {}).get("tiempos") or {},
			"transcripcion": (_transcripciones_meta or {}).get("tiempos") or {},
			"correccion": (_correcciones_meta or {}).get("tiempos") or {},
			"fragmentacion": data.get("tiempos") or {},
			"vectorizacion": meta_vec.get("tiempos") or {},
		}

		embedder_id = embedder or meta_vec.get("embedder")
		if not embedder_id:
			print(f"[ERROR] No se pudo determinar el embedder para '{hash_id}'.")
			sys.exit(1)
		if embedder and meta_vec.get("embedder") and embedder != meta_vec["embedder"]:
			print(
				f"[ERROR] El embedder solicitado '{embedder}' no coincide con "
				f"el de vectores.json ('{meta_vec['embedder']}') para hash={hash_id}."
			)
			sys.exit(1)

		dim = int(meta_vec.get("dim") or 0)
		if dim <= 0:
			print(f"[ERROR] 'dim' invalido en '{vectores_meta_path}'.")
			sys.exit(1)

		# --- Deduplicacion ---
		nombre_tabla = tabla or embedder_id
		with medir("dedup_check"):
			ya_indexado = (not reclear) and backend_mod.hash_indexado(db, nombre_tabla, hash_id)
		if ya_indexado:
			print(f"[SKIP] hash={hash_id} ya indexado en tabla '{nombre_tabla}'. Usar --recrear para reindexar.")
			return

		try:
			with medir("carga_npz"):
				npz = np.load(vectores_npz_path)
		except FileNotFoundError:
			print(f"[ERROR] No se encontro '{vectores_npz_path}'.")
			sys.exit(1)
		embeddings = npz["embeddings"]
		if embeddings.shape != (len(fragmentos), dim):
			print(
				f"[ERROR] Shape de embeddings {embeddings.shape} no coincide con "
				f"(num_fragmentos={len(fragmentos)}, dim={dim})."
			)
			sys.exit(1)
		embeddings = embeddings.astype(np.float32, copy=False)

		# --- BM25: tokenizar ---
		tokens_por_fragmento = _tokenizar_fragmentos(fragmentos)

		# --- Construir filas ---
		tags_json = json.dumps(tags, ensure_ascii=False)
		tiempos_json = json.dumps({**tiempos_previos, "indexacion": crono.resumen()}, ensure_ascii=False)

		lance_filas = []
		chunk_filas = []
		for i, frag in enumerate(fragmentos):
			chunk_id = f"{hash_id}:{i}"
			segmentos = [
				{
					"inicio": float(s.get("inicio", 0.0)),
					"fin": float(s.get("fin", 0.0)),
					"texto": s.get("texto", ""),
				}
				for s in (frag.get("segmentos") or [])
			]
			lance_filas.append({
				"id": chunk_id,
				"hash": hash_id,
				"texto_bm25": tokens_a_texto(tokens_por_fragmento[i]),
				"tags": tags_json,
				"vector": embeddings[i].tolist(),
			})
			chunk_filas.append({
				"id": chunk_id,
				"hash": hash_id,
				"chunk_idx": i,
				"texto": frag.get("texto", ""),
				"inicio": float(frag.get("inicio", 0.0)),
				"fin": float(frag.get("fin", 0.0)),
				"segmentos_json": json.dumps(segmentos, ensure_ascii=False),
			})

		# --- Thumbnail ---
		thumbnail = _cargar_thumbnail(folder)

		# --- SQLite ---
		iu.crear_tablas()
		iu.escribir_recurso(hash_id, titulo, uri, fuente, tiempos_json=tiempos_json, thumbnail=thumbnail)
		with medir("escritura_sqlite"):
			iu.escribir_chunks(chunk_filas)

		with medir("escritura_db"):
			backend_mod.escribir_tabla(db, nombre_tabla, lance_filas, dim=dim, reclear=reclear)

	print(f"[TIEMPOS] hash={hash_id}: {crono.resumen()}")


def _cargar_thumbnail(folder) -> bytes | None:
	for ext in (".jpg", ".jpeg", ".webp", ".png"):
		matches = list(folder.glob(f"*{ext}"))
		if matches:
			return matches[0].read_bytes()
	return None


if __name__ == "__main__":
	main()
