import argparse
import sys

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
	for hash_id in hashes:
		_procesar_hash(hash_id, embedder=embedder, db_ruta=db_ruta, tabla=tabla, reclear=reclear, tags=tags)


@cronometrar(etiqueta="Tokenizacion BM25")
def _tokenizar_fragmentos(fragmentos: list[dict]) -> list[list[str]]:
	return [tokenizar(f["texto"]) for f in fragmentos]


def _procesar_hash(
	hash_id: str,
	embedder: str | None,
	db_ruta: str,
	tabla: str | None,
	reclear: bool,
	tags: dict[str, str],
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

	print(f"[OK] Fragmentos cargados: {len(fragmentos)} (hash={hash_id})")

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

	# --- LanceDB: pendiente de implementar ---


if __name__ == "__main__":
	main()
