"""Modulo de analitica docente (insights).

Expone una vista agregada del corpus indexado y del comportamiento de busqueda
de los usuarios. No transforma contenido audiovisual ni interviene en el ranking:
solo consulta el indice (`indice.db`) y el registro de actividad
(`resultados.db`) para ofrecer informacion operativa al docente o administrador.

Salidas principales:
  1. Videos mas encontrados   -> `insights.encontrados`
  2. Videos mas seleccionados -> `insights.seleccionados`
  3. Agrupacion de consultas  -> `insights.agrupador` (HDBSCAN + KeyBERT)

Todas las salidas se calculan de forma offline / bajo demanda y se exportan
como tablas de inspeccion (CSV + JSON) en `resultados/insights/`.
"""

__all__ = [
	"registro",
	"corpus",
	"encontrados",
	"seleccionados",
	"agrupador",
	"exportar",
]
