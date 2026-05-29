from faster_whisper import BatchedInferencePipeline, WhisperModel
import soundfile as sf
import time

from compartido.rutas import DESCARGAS_DIR
from compartido.utils import cronometrar, crear_perfil_hardware
from compartido.json_utils import cargar_archivo, guardar_archivo

from .chunks import obtener_fragmentos_asr


MODELOS_DIR = DESCARGAS_DIR / "modelos" / "whisper"

PERFIL_WHISPER = {
	"padding": 0.18,
	"join_gap": 0.50,
	"duracion_minima": 20.00,
	"duracion_target": 28.00,
	"duracion_maxima": 30.00,
	"overlap": 0.15,
}

# Empirically calibrated on RTX 16 GB, batch=92, chunks ≤30 s
# Cost is dominated by encoder activations, which scale with model depth
_VRAM_POR_CHUNK_MB = {
	"tiny":     40,
	"base":     90,
	"small":   160,
	"medium":  200,
	"large":   245,
	"large-v2": 245,
	"large-v3": 245,
	"turbo":   245,  # large-v3 encoder
}

def _vram_por_chunk(model_size_or_path: str) -> int:
	key = next((k for k in _VRAM_POR_CHUNK_MB if model_size_or_path.startswith(k)), "large")
	return _VRAM_POR_CHUNK_MB[key]

def _batch_size_cpu(n_fragmentos: int, model_size_or_path: str, hardware: dict) -> int:
	physical = hardware.get("cpu_physical_cores") or hardware.get("cpu_logical_cores") or 1
	ram_gb = hardware.get("ram_gb", 4)
	chunk_mb = _vram_por_chunk(model_size_or_path)
	batch_size = 1

	if ram_gb >= 16 and physical >= 8:
		batch_size = 2

	if ram_gb >= 32 and physical >= 12 and chunk_mb <= _VRAM_POR_CHUNK_MB["small"]:
		batch_size = 4

	return max(1, min(batch_size, n_fragmentos))

def computar_parametros(n_fragmentos: int, model_size_or_path: str, margen: float = 0.85):
	# hardware = crear_perfil_hardware(forzado={"vram_gb": 8})
	hardware = crear_perfil_hardware()
	dev = hardware.get("device", "cpu")

	params = {"device": dev, "ram_gb": hardware.get("ram_gb", 0)}
	
	if dev in ["cuda", "mps", "xpu"]:
		total_vram = hardware.get("vram_gb", 0) * 1024
		free_vram  = total_vram * margen
		chunk_mb   = _vram_por_chunk(model_size_or_path)
		optimal    = int(free_vram // chunk_mb)
		params["batch_size"]   = max(1, min(optimal, n_fragmentos))
		params["compute_type"] = "float16" if hardware["device"] == "cuda" else "int8_float16"
		params["vram_gb"] = hardware.get("vram_gb", 0)
	else:
		threads = hardware.get("cpu_physical_cores") or hardware.get("cpu_logical_cores") or 1
		params["num_workers"]  = 1
		params["cpu_threads"]  = max(1, threads)
		params["batch_size"]   = _batch_size_cpu(n_fragmentos, model_size_or_path, hardware)
		params["compute_type"] = "int8"

	return params

def serializar_transcripciones(segmentos):
    transcripciones = []

    for segmento in segmentos:
        texto = segmento.text.strip()
        if not texto:
            continue

        inicio = round(float(segmento.start), 3)
        fin = round(float(segmento.end), 3)

        if fin <= inicio:
            continue

        transcripciones.append({
            "texto": texto,
            "inicio": inicio,
            "fin": fin,
            "duracion": round(fin - inicio, 3),
        })

    transcripciones.sort(key=lambda item: (item["inicio"], item["fin"]))
    return transcripciones

@cronometrar(etiqueta="Cargando modelo Whisper")
def cargar_modelo_whisper(model_size_or_path, params):
	print(f"[INFO] 'whisper-{model_size_or_path}' en {params['device'].upper()} ({params['compute_type']})...")
	model = WhisperModel(
		model_size_or_path,
		device=params["device"],
		compute_type=params["compute_type"],
		cpu_threads=params.get("cpu_threads", 1),
		num_workers=params.get("num_workers", 1),
	)
	return model

@cronometrar(etiqueta="Whisper")
def transcribir_whisper(paths, modelo="small"):
	segmentos_raw = cargar_archivo(paths["segmentos"])["segmentos"]
	spans = obtener_fragmentos_asr(segmentos_raw, PERFIL_WHISPER)
	audio_path = paths["audio"]

	params = computar_parametros(len(spans), modelo)

	model = cargar_modelo_whisper(modelo, params)
	carga_tiempo = round(cargar_modelo_whisper.elapsed, 3)

	transcripciones = []

	audio_data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
	clip_timestamps = [{"start": s, "end": e} for s, e in spans]
	pipeline = BatchedInferencePipeline(model=model)

	tiempo_inicial = time.perf_counter()
	if params["device"] in ["cuda", "mps", "xpu"]:
		print(f"Procesando {len(spans)} fragmentos con batch size {params['batch_size']} en {params['device'].upper()} ({params['compute_type']})...")
		segments, info = pipeline.transcribe(
			audio_data,
			batch_size=params["batch_size"],
			language="es",
			without_timestamps=True,
			log_progress=True,
			clip_timestamps=clip_timestamps,
			vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=False,
		)
	else:
		print(
			f"Procesando {len(spans)} fragmentos en CPU ({params['compute_type']}) "
			f"batch_size={params['batch_size']} cpu_threads={params['cpu_threads']}..."
		)
		segments, info = pipeline.transcribe(
			audio_data,
			batch_size=params["batch_size"],
			language="es",
			without_timestamps=True,
			log_progress=True,
			clip_timestamps=clip_timestamps,
			vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=False,
		)

	transcripciones = serializar_transcripciones(list(segments))
	tiempo_transcripcion = round(time.perf_counter() - tiempo_inicial, 3)
	
	data = {
		"modelo": f"whisper-{modelo}",
		"tiempo_carga": carga_tiempo,
		"hardware": params,
		"tiempo_transcripcion": tiempo_transcripcion,
		"tiempo_total": None,
		"speed_up": None,
		"rtf": None,
		"duracion_audio": info.duration,
		"fragmentos": len(spans),
		"perfil_fragmentacion": PERFIL_WHISPER,
		"transcripciones": transcripciones,
		"texto_completo": " ".join([t["texto"] for t in transcripciones])
	}
	
	if guardar_archivo(paths["transcripciones"], data):
		print(f"[INFO] Transcripciones guardadas correctamente.")


# HASH = "3bb50e9b2b8c3960"
# audio_path = DESCARGAS_DIR / HASH / "audio_procesado.wav"
# transcribir_whisper({"segmentos": DESCARGAS_DIR / HASH / "segmentos.json", "audio": audio_path, "transcripciones": DESCARGAS_DIR / HASH / "transcripciones.json"}, modelo="turbo")