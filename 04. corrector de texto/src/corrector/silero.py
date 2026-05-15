import os

import torch

from compartido.rutas import DESCARGAS_DIR
from compartido.utils import cronometrar


CACHE_DIR = DESCARGAS_DIR / "modelos" / "torch_hub"

_APPLY_TE = None


@cronometrar(etiqueta="Carga silero_te")
def _hacer_carga_silero_te():
	global _APPLY_TE

	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	torch.hub.set_dir(str(CACHE_DIR))
	torch.set_grad_enabled(False)
	torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

	_, _, languages, _, apply_te = torch.hub.load(
		repo_or_dir="snakers4/silero-models",
		model="silero_te",
		trust_repo=True,
	)

	if "es" not in languages:
		raise RuntimeError(f"silero_te no reporta soporte para espanol: {languages}")

	_APPLY_TE = apply_te


def cargar_silero_te():
	global _APPLY_TE

	if _APPLY_TE is not None:
		return _APPLY_TE

	_hacer_carga_silero_te()
	return _APPLY_TE
