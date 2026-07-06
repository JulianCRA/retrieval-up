"""CLI del modulo de analitica docente.

Calcula y exporta, de forma offline o bajo demanda, las salidas operativas:
  - corpus         : vista agregada del corpus indexado
  - encontrados    : videos mas encontrados por el sistema
  - seleccionados  : videos mas seleccionados por los usuarios
  - comparar       : encontrados vs. seleccionados
  - grupos         : agrupacion de consultas (HDBSCAN) + etiquetas (KeyBERT)
  - todo           : ejecuta todas las salidas anteriores

Uso:
    insights corpus
    insights encontrados --top-n 20 --rank-max 3
    insights grupos --embedder bge-m3 --min-cluster-size 3
    insights todo
"""
from __future__ import annotations

import argparse

from compartido.utils import crear_perfil_hardware

from insights import corpus as mod_corpus
from insights import encontrados as mod_encontrados
from insights import seleccionados as mod_seleccionados
from insights import exportar
from insights.registro import cargar_actividad


def _device(forzar_cpu: bool) -> str:
	forzado = {"device": "cpu"} if forzar_cpu else None
	return crear_perfil_hardware(forzado=forzado)["device"]


def _imprimir_filas(titulo: str, filas: list[dict], columnas: list[str]) -> None:
	print(f"\n=== {titulo} ===")
	if not filas:
		print("(sin datos)")
		return
	for fila in filas:
		partes = [f"{col}={fila.get(col)}" for col in columnas]
		print("  " + "  ".join(partes))


# ─── Sub-comandos ────────────────────────────────────────────────────────────

def cmd_corpus(args) -> None:
	resumen = mod_corpus.resumen_corpus()
	print("\n=== Vista agregada del corpus indexado ===")
	print(f"  recursos:        {resumen['num_recursos']}")
	print(f"  duracion total:  {resumen['duracion_total_horas']} h ({resumen['duracion_total_seg']} s)")
	print(f"  fuentes:         {resumen['num_fuentes']}")
	print(f"  chunks/modelo:   {resumen['chunks_por_modelo']}")
	print(f"  tags:            {resumen['tags']}")
	print(f"  sin tags:        {resumen['recursos_sin_tags']}")
	if not args.no_export:
		exportar.exportar_json("corpus", resumen)


def cmd_encontrados(args) -> None:
	actividad = cargar_actividad().filtrar_embedder(args.embedder)
	filas = mod_encontrados.videos_mas_encontrados(
		actividad, top_n=args.top_n, rank_max=args.rank_max
	)
	_imprimir_filas("Videos mas encontrados", filas, ["rank", "apariciones", "busquedas_distintas", "titulo"])
	if not args.no_export:
		exportar.exportar_tabla("encontrados", filas,
			campos=["rank", "video_id", "titulo", "apariciones", "busquedas_distintas"])


def cmd_seleccionados(args) -> None:
	actividad = cargar_actividad().filtrar_embedder(args.embedder)
	filas = mod_seleccionados.videos_mas_seleccionados(actividad, top_n=args.top_n)
	_imprimir_filas("Videos mas seleccionados", filas, ["rank", "selecciones", "titulo"])
	if not args.no_export:
		exportar.exportar_tabla("seleccionados", filas,
			campos=["rank", "video_id", "titulo", "selecciones"])


def cmd_comparar(args) -> None:
	actividad = cargar_actividad().filtrar_embedder(args.embedder)
	filas = mod_seleccionados.comparar_encontrados_seleccionados(actividad, top_n=args.top_n)
	_imprimir_filas("Encontrados vs. seleccionados", filas,
		["apariciones", "selecciones", "tasa_seleccion", "titulo"])
	if not args.no_export:
		exportar.exportar_tabla("comparacion", filas,
			campos=["video_id", "titulo", "apariciones", "selecciones", "tasa_seleccion"])


def cmd_grupos(args) -> None:
	from insights.agrupador import agrupar_consultas

	actividad = cargar_actividad().filtrar_embedder(args.embedder)
	resultado = agrupar_consultas(
		actividad,
		device=_device(args.forzar_cpu),
		min_cluster_size=args.min_cluster_size,
		min_samples=args.min_samples,
	)

	print("\n=== Agrupacion de consultas ===")
	print(f"  embedder:   {resultado['embedder']}")
	print(f"  consultas:  {resultado['num_consultas']}")
	print(f"  grupos:     {resultado['num_grupos']}")
	for g in resultado["grupos"]:
		flag = " [sin selecciones]" if g["sin_seleccion"] else ""
		print(f"\n  · grupo {g['grupo']} (n={g['tamano']}, freq={g['frecuencia_total']}){flag}")
		print(f"    etiqueta: {g['etiqueta']}")
		for q in g["consultas"]:
			print(f"      - {q}")
	if resultado["ruido"]:
		print(f"\n  ruido ({len(resultado['ruido'])} consultas aisladas):")
		for r in resultado["ruido"]:
			print(f"      - {r['query']}")

	if not args.no_export:
		# Tabla plana de grupos para inspeccion.
		filas = [
			{
				"grupo": g["grupo"],
				"etiqueta": g["etiqueta"],
				"tamano": g["tamano"],
				"frecuencia_total": g["frecuencia_total"],
				"selecciones_total": g["selecciones_total"],
				"sin_seleccion": g["sin_seleccion"],
				"consultas": g["consultas"],
			}
			for g in resultado["grupos"]
		]
		exportar.exportar_tabla("grupos", filas,
			campos=["grupo", "etiqueta", "tamano", "frecuencia_total",
					"selecciones_total", "sin_seleccion", "consultas"])
		exportar.exportar_json("grupos_detalle", resultado)


def cmd_todo(args) -> None:
	cmd_corpus(args)
	cmd_encontrados(args)
	cmd_seleccionados(args)
	cmd_comparar(args)
	cmd_grupos(args)


# ─── Parser ──────────────────────────────────────────────────────────────────

def _add_comunes(sub: argparse.ArgumentParser, *, top_default: int = 20) -> None:
	sub.add_argument(
		"--embedder",
		default=None,
		help="Restringe el analisis a las busquedas de un embedder concreto.",
	)
	sub.add_argument(
		"--top-n",
		type=int,
		default=top_default,
		dest="top_n",
		help=f"Numero de filas a devolver (default: {top_default}, 0 = todas).",
	)
	sub.add_argument(
		"--no-export",
		action="store_true",
		dest="no_export",
		help="Solo imprimir por consola, sin escribir archivos en resultados/insights/.",
	)
	sub.add_argument(
		"--forzar-cpu",
		action="store_true",
		dest="forzar_cpu",
		help="Forzar CPU aunque haya GPU (afecta al encoder de consultas y KeyBERT).",
	)


def main() -> None:
	parser = argparse.ArgumentParser(
		prog="insights",
		description="Modulo de analitica docente: corpus indexado y comportamiento de busqueda.",
	)
	sub = parser.add_subparsers(dest="comando", required=True)

	p_corpus = sub.add_parser("corpus", help="Vista agregada del corpus indexado.")
	_add_comunes(p_corpus)
	p_corpus.set_defaults(func=cmd_corpus, rank_max=None,
					   min_cluster_size=5, min_samples=2)
	p_enc = sub.add_parser("encontrados", help="Videos mas encontrados por el sistema.")
	_add_comunes(p_enc)
	p_enc.add_argument("--rank-max", type=int, default=None, dest="rank_max",
					   help="Solo contar apariciones con rank <= RANK_MAX (ej. primeros 3).")
	p_enc.set_defaults(func=cmd_encontrados, min_cluster_size=5, min_samples=2)

	p_sel = sub.add_parser("seleccionados", help="Videos mas seleccionados por los usuarios.")
	_add_comunes(p_sel)
	p_sel.set_defaults(func=cmd_seleccionados, rank_max=None,
				   min_cluster_size=5, min_samples=2)
	p_cmp = sub.add_parser("comparar", help="Encontrados vs. seleccionados.")
	_add_comunes(p_cmp)
	p_cmp.set_defaults(func=cmd_comparar, rank_max=None,
				   min_cluster_size=5, min_samples=2)
	p_grp = sub.add_parser("grupos", help="Agrupar consultas similares (HDBSCAN + KeyBERT).")
	_add_comunes(p_grp)
p_grp.add_argument("--min-cluster-size", type=int, default=5, dest="min_cluster_size",
				   help="Tamano minimo de grupo para HDBSCAN (default: 5).")
	p_grp.add_argument("--min-samples", type=int, default=2, dest="min_samples",
				   help="Parametro min_samples de HDBSCAN (default: 2).")
	p_grp.set_defaults(func=cmd_grupos, rank_max=None)

	p_todo = sub.add_parser("todo", help="Ejecuta todas las salidas.")
	_add_comunes(p_todo)
	p_todo.add_argument("--rank-max", type=int, default=None, dest="rank_max")
	p_todo.add_argument("--min-cluster-size", type=int, default=5, dest="min_cluster_size")
	p_todo.add_argument("--min-samples", type=int, default=2, dest="min_samples")
	p_todo.set_defaults(func=cmd_todo)

	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
