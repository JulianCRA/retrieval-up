import argparse
import sys

from compartido import json_utils as ju
from compartido.rutas import DESCARGAS_DIR
from compartido.utils import crear_perfil_hardware, cronometrar, cronometro_activo, medir
from corrector.alinear import alinear_segmentos


def main():
	parser = argparse.ArgumentParser(
		prog="corrector",
		description="Restaura puntuacion y mayusculas.",
	)
	parser.add_argument(
		"--hash",
		required=True,
		action="append",
		dest="hashes",
		metavar="HASH",
		help="Hash a corregir. Repetir para procesar varios en un solo comando.",
	)
	parser.add_argument(
		"--m",
		choices=["silero", "p-all"],
		default="silero",
		help="Motor de puntuacion: silero (default) o p-all (punctuate-all + spacy).",
	)
	parser.add_argument(
		"--forzar-cpu",
		action="store_true",
		dest="forzar_cpu",
		help="Forzar uso de CPU aunque haya GPU disponible (solo aplica al backend p-all; silero ya corre en CPU).",
	)

	args = parser.parse_args()
	procesar(args.hashes, backend=args.m, forzar_cpu=args.forzar_cpu)


def procesar(hashes: list[str], backend: str = "silero", forzar_cpu: bool = False):
	forzado = {"device": "cpu"} if forzar_cpu else None
	perfil = crear_perfil_hardware(forzado=forzado)
	# Pre-load model once before the loop.
	apply_te = None
	if backend == "silero":
		import torch
		from corrector.silero import cargar_silero_te
		apply_te = cargar_silero_te()
	elif backend == "p-all":
		from corrector.p_all import _cargar
		_cargar(device=perfil["device"])  # warm up global cache

	for hash_id in hashes:
		procesar_hash(hash_id, backend=backend, apply_te=apply_te, perfil=perfil)


def _texto_desde_segmentos(segmentos: list[dict]) -> str:
	return " ".join(seg.get("texto", "") for seg in segmentos).strip()


def procesar_hash(hash_id: str, backend: str = "silero", apply_te=None, perfil=None):
	folder = DESCARGAS_DIR / hash_id
	transcripciones_path = folder / "transcripciones.json"
	correcciones_path = folder / "correcciones.json"

	with cronometro_activo() as crono:
		with medir("lectura_transcripciones"):
			data = ju.cargar_archivo(transcripciones_path)
		if not data:
			print(f"[ERROR] No se pudo cargar '{transcripciones_path}'.")
			sys.exit(1)

		segmentos = data.get("transcripciones")
		if not segmentos:
			print(f"[ERROR] No se encontraron transcripciones en '{transcripciones_path}'.")
			sys.exit(1)

		modelo_asr = data.get("modelo", "")
		texto = data.get("texto") or _texto_desde_segmentos(segmentos)

		if "vosk" not in modelo_asr.lower():
			print(
				f"[INFO] El modelo ASR '{modelo_asr}' ya agrega puntuacion. "
				"No hay necesidad de postprocesamiento."
			)
			resultado = {
				"modelo_corrector": "ninguno",
				"modelo_asr": modelo_asr,
				"tiempos": crono.resumen(),
				"texto": texto,
				"transcripciones": [
					{
						"inicio": s["inicio"],
						"fin": s["fin"],
						"duracion": s["duracion"],
						"texto": s["texto"],
					}
					for s in segmentos
				],
			}
			if ju.guardar_archivo(correcciones_path, resultado):
				print(f"[OK] Correcciones guardadas en '{correcciones_path}'.")
				return
			print(f"[ERROR] No se pudo guardar '{correcciones_path}'.")
			sys.exit(1)

		if backend == "p-all":
			from corrector.p_all import corregir_p_all
			device = perfil["device"] if perfil else None
			@cronometrar(etiqueta="correccion")
			def _procesar_p_all():
				return corregir_p_all(texto, device=device)
			texto_corregido = _procesar_p_all()
		else:
			import torch
			if apply_te is None:
				from corrector.silero import cargar_silero_te
				apply_te = cargar_silero_te()
			@cronometrar(etiqueta="correccion")
			def _procesar_silero():
				return apply_te(texto, lan="es")
			with torch.inference_mode():
				texto_corregido = _procesar_silero()

		print(f"[INFO] Correccion realizada para hash '{hash_id}' (backend={backend}).")

		with medir("alineacion"):
			segs_corregidos = alinear_segmentos(segmentos, texto_corregido)
		resultado = {
			"modelo_corrector": backend,
			"tiempos": crono.resumen(),
			"texto": texto_corregido,
			"transcripciones": [
				{
					"inicio": s["inicio"],
					"fin": s["fin"],
					"duracion": s["duracion"],
					"texto": s.get("texto_corregido", s.get("texto", "")),
				}
				for s in segs_corregidos
			],
		}

		if ju.guardar_archivo(correcciones_path, resultado):
			print(f"[OK] Correcciones guardadas en '{correcciones_path}'.")
			return

		print(f"[ERROR] No se pudo guardar '{correcciones_path}'.")
		sys.exit(1)


if __name__ == "__main__":
	main()
