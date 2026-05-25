import argparse
import json
import sys

import lancedb
import numpy as np
import pyarrow as pa

from compartido import json_utils as ju
from compartido.rutas import DESCARGAS_DIR
from compartido.embedders import listar_ids
from compartido.utils import cronometrar

from indexador.bm25 import tokenizar, tokens_a_texto


# Directorio por defecto del indice LanceDB.
INDICE_DIR = DESCARGAS_DIR / "indice"


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
			"Indexa fragmentos + embeddings en una base LanceDB local. "
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
		"--db",
		default=str(INDICE_DIR),
		metavar="RUTA",
		help=f"Ruta al directorio LanceDB (default: {INDICE_DIR}).",
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
	)


def procesar(
	hashes: list[str],
	embedder: str | None,
	db_ruta: str,
	tabla: str | None,
	reclear: bool,
	tags: dict[str, str],
):
	db = _abrir_lancedb(db_ruta)
	for hash_id in hashes:
		_procesar_hash(
			hash_id,
			db=db,
			embedder=embedder,
			tabla=tabla,
			reclear=reclear,
			tags=tags,
		)
		# Tras el primer hash, no recrear de nuevo en los siguientes.
		reclear = False


@cronometrar(etiqueta="Tokenizacion BM25")
def _tokenizar_fragmentos(fragmentos: list[dict]) -> list[list[str]]:
	return [tokenizar(f["texto"]) for f in fragmentos]


def _procesar_hash(
	hash_id: str,
	db: lancedb.DBConnection,
	embedder: str | None,
	tabla: str | None,
	reclear: bool,
	tags: dict[str, str],
):
	folder = DESCARGAS_DIR / hash_id
	fragmentos_path = folder / "fragmentos.json"
	vectores_meta_path = folder / "vectores.json"
	vectores_npz_path = folder / "vectores.npz"

	data = ju.cargar_archivo(fragmentos_path)
	if not data:
		print(f"[ERROR] No se pudo cargar '{fragmentos_path}'.")
		sys.exit(1)

	fragmentos = data.get("fragmentos")
	if not fragmentos:
		print(f"[ERROR] No hay fragmentos en '{fragmentos_path}'.")
		sys.exit(1)

	print(f"[OK] Fragmentos cargados: {len(fragmentos)} (hash={hash_id})")

	# --- Vectores ---
	meta_vec = ju.cargar_archivo(vectores_meta_path)
	if not meta_vec:
		print(f"[ERROR] No se pudo cargar '{vectores_meta_path}'.")
		sys.exit(1)

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

	try:
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

	# --- BM25: tokenizar y persistir ---
	tokens_por_fragmento = _tokenizar_fragmentos(fragmentos)

	tokens_bm25_path = folder / "tokens_bm25.json"
	resultado_bm25 = {
		"hash": hash_id,
		"tokenizador": "spacy_es_core_news_lg",
		"con_stemming": True,
		"num_fragmentos": len(fragmentos),
		"fragmentos": [
			{
				"chunk_idx": i,
				"tokens": tokens,
				"texto_bm25": tokens_a_texto(tokens),
			}
			for i, tokens in enumerate(tokens_por_fragmento)
		],
	}
	if ju.guardar_archivo(tokens_bm25_path, resultado_bm25):
		print(f"[OK] Tokens BM25 guardados en '{tokens_bm25_path}'.")
	else:
		print(f"[ERROR] No se pudo guardar '{tokens_bm25_path}'.")
		sys.exit(1)

	# --- LanceDB: construir filas y escribir ---
	nombre_tabla = tabla or embedder_id
	tags_json = json.dumps(tags, ensure_ascii=False)

	filas = []
	for i, frag in enumerate(fragmentos):
		filas.append({
			"id": f"{hash_id}:{i}",
			"hash": hash_id,
			"chunk_idx": i,
			"texto": frag.get("texto", ""),
			"texto_bm25": tokens_a_texto(tokens_por_fragmento[i]),
			"inicio": float(frag.get("inicio", 0.0)),
			"fin": float(frag.get("fin", 0.0)),
			"tags": tags_json,
			"vector": embeddings[i].tolist(),
		})

	_escribir_tabla(db, nombre_tabla, filas, dim=dim, reclear=reclear)


def _abrir_lancedb(db_ruta: str) -> lancedb.DBConnection:
	db = lancedb.connect(db_ruta)
	print(f"[OK] LanceDB conectado en '{db_ruta}'. Tablas existentes: {db.list_tables()}")
	return db


def _esquema(dim: int) -> pa.Schema:
	return pa.schema([
		pa.field("id", pa.string()),
		pa.field("hash", pa.string()),
		pa.field("chunk_idx", pa.int32()),
		pa.field("texto", pa.string()),
		pa.field("texto_bm25", pa.string()),
		pa.field("inicio", pa.float32()),
		pa.field("fin", pa.float32()),
		pa.field("tags", pa.string()),
		pa.field("vector", pa.list_(pa.float32(), dim)),
	])


def _escribir_tabla(
	db: lancedb.DBConnection,
	nombre: str,
	filas: list[dict],
	dim: int,
	reclear: bool,
):
	existe = nombre in db.list_tables()
	if existe and reclear:
		db.drop_table(nombre)
		print(f"[OK] Tabla '{nombre}' eliminada (--recrear).")
		existe = False

	esquema = _esquema(dim)
	if not existe:
		tabla = db.create_table(nombre, data=filas, schema=esquema)
		print(f"[OK] Tabla '{nombre}' creada con {len(filas)} filas (dim={dim}).")
	else:
		tabla = db.open_table(nombre)
		if tabla.schema.field("vector").type.list_size != dim:
			print(
				f"[ERROR] Dim de la tabla '{nombre}' "
				f"({tabla.schema.field('vector').type.list_size}) "
				f"no coincide con dim del lote ({dim})."
			)
			sys.exit(1)
		tabla.add(filas)
		print(f"[OK] Tabla '{nombre}': agregadas {len(filas)} filas (total={tabla.count_rows()}).")


if __name__ == "__main__":
	main()
