import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

from compartido.rutas import MODELOS_COHERE_DIR
from compartido.utils import cronometrar, crear_perfil_hardware, medir
from compartido.json_utils import cargar_archivo, guardar_archivo
from .chunks import obtener_fragmentos_asr

MODELOS_DIR = MODELOS_COHERE_DIR
MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"

PERFIL_COHERE = {
    "padding": 0.18,
    "join_gap": 0.50,
    "duracion_minima": 5.00,
    "duracion_target": 20.00,
    "duracion_maxima": 24.00,
    "overlap": 0.15,
}

# Conservative VRAM estimate per ~20 s segment at float16 (encoder activations + decoder KV)
_VRAM_POR_SEGMENTO_MB = 150

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _computar_parametros(n_segmentos: int, hardware: dict) -> dict:
    device = hardware["device"]
    params = {"device": device}

    if device in ("cuda", "mps", "xpu"):
        vram_mb = hardware.get("vram_gb", 0) * 1024 * 0.82
        batch_size = max(1, min(int(vram_mb // _VRAM_POR_SEGMENTO_MB), n_segmentos, 16))
        # MPS does not support float16 for all ops; use float32 there
        params["torch_dtype"] = "float16" if device == "cuda" else "float32"
        params["batch_size"] = batch_size
        params["vram_gb"] = hardware.get("vram_gb", 0)
    else:
        threads = hardware.get("cpu_physical_cores") or hardware.get("cpu_logical_cores") or 1
        params["torch_dtype"] = "float32"
        params["batch_size"] = 1
        params["cpu_threads"] = min(threads, 16)

    return params


def _device_map(device: str) -> str:
    """Normaliza el dispositivo al formato que acepta device_map."""
    if device == "cuda":
        return "cuda:0"
    return device  # "mps", "cpu", "xpu"


@cronometrar(etiqueta="carga_modelo")
def _cargar_modelo(params: dict):
    torch_dtype = _DTYPE_MAP[params["torch_dtype"]]
    device = params["device"]

    if device == "cpu":
        torch.set_num_threads(params.get("cpu_threads", 1))

    print(f"[INFO] Cargando '{MODEL_ID}' en {device.upper()} ({params['torch_dtype']})...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=str(MODELOS_DIR))
    model = CohereAsrForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        device_map=_device_map(device),
        cache_dir=str(MODELOS_DIR),
    )
    model.eval()
    return processor, model


_COHERE_CACHE: dict = {}


def cargar_modelo(params: dict):
    key = (params["device"], params["torch_dtype"])
    if key in _COHERE_CACHE:
        return _COHERE_CACHE[key]
    result = _cargar_modelo(params)
    _COHERE_CACHE[key] = result
    return result


@cronometrar(etiqueta="transcripcion")
def transcribir_cohere(paths, idioma: str = "es", perfil=None, batch_size: int | None = None):
    segmentos_raw = cargar_archivo(paths["segmentos"])["segmentos"]
    spans = obtener_fragmentos_asr(segmentos_raw, PERFIL_COHERE)

    hardware = perfil if perfil is not None else crear_perfil_hardware()
    params = _computar_parametros(len(spans), hardware)

    processor, model = cargar_modelo(params)

    audio_data, sr = sf.read(str(paths["audio"]), dtype="float32", always_2d=False)

    torch_dtype = _DTYPE_MAP[params["torch_dtype"]]
    device = params["device"]
    batch_size = batch_size if batch_size is not None else params["batch_size"]
    model_device = next(model.parameters()).device

    print(
        f"[INFO] Procesando {len(spans)} segmentos, "
        f"batch_size={batch_size}, {device.upper()} ({params['torch_dtype']})..."
    )

    transcripciones = []
    batches = [spans[i:i + batch_size] for i in range(0, len(spans), batch_size)]

    with medir("inferencia"):
        for batch_spans in tqdm(batches, desc="Transcribiendo", unit="lote"):
            audios = [audio_data[int(s * sr):int(e * sr)] for s, e in batch_spans]

            inputs = processor(
                audios,
                sampling_rate=sr,
                return_tensors="pt",
                language=idioma,
                padding=True,
            )
            inputs = {
                k: (v.to(device=model_device, dtype=torch_dtype) if v.is_floating_point() else v.to(device=model_device))
                if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=256)

            for j, (inicio, fin) in enumerate(batch_spans):
                texto = processor.decode(output_ids[j], skip_special_tokens=True).strip()
                if texto:
                    transcripciones.append({
                        "inicio": round(inicio, 3),
                        "fin": round(fin, 3),
                        "duracion": round(fin - inicio, 3),
                        "texto": texto,
                    })

    transcripciones.sort(key=lambda x: x["inicio"])
    full_text = " ".join(t["texto"] for t in transcripciones)

    data = {
        "modelo": MODEL_ID,
        "hardware": params,
        "fragmentos": len(spans),
        "perfil_fragmentacion": PERFIL_COHERE,
        "transcripciones": transcripciones,
        "texto_completo": full_text,
    }

    if guardar_archivo(paths["transcripciones"], data):
        print("[INFO] Transcripciones guardadas correctamente.")
