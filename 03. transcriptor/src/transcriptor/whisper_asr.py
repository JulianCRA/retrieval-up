import soundfile as sf
from tqdm import tqdm
from faster_whisper import WhisperModel, BatchedInferencePipeline

from compartido.rutas import DESCARGAS_DIR
from compartido.json_utils import cargar_archivo, guardar_archivo
from .chunks import obtener_fragmentos_asr

MODELOS_DIR = DESCARGAS_DIR / "modelos" / "whisper"

MODELO_DEFAULT = "large-v3-turbo"  # opciones: tiny, base, small, large-v3-turbo
# MODELO_DEFAULT = "base"

PERFIL_WHISPER = {
    "padding": 0.18,
    "join_gap": 0.50,
    "duracion_minima": 10.00,
    "duracion_target": 28.00,
    "duracion_maxima": 30.00,
    "overlap": 0.15,
}

WHISPER_SAMPLE_RATE = 16_000

BATCH_SIZE = 16  # segmentos procesados en paralelo por el GPU


def _cargar_modelo(nombre=MODELO_DEFAULT, device="cuda", compute_type="default"):
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Cargando modelo Whisper '{nombre}' (device={device}, compute_type={compute_type})...")
    base = WhisperModel(nombre, device=str(device), compute_type=compute_type, download_root=str(MODELOS_DIR))
    return BatchedInferencePipeline(model=base)


def transcribir_whisper(paths, nombre_modelo=MODELO_DEFAULT, device="cuda", compute_type="default", batch_size=BATCH_SIZE):
    pipeline = _cargar_modelo(nombre_modelo, device, compute_type)

    audio_data, sample_rate = sf.read(str(paths["audio"]), dtype="float32", always_2d=False)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if sample_rate != WHISPER_SAMPLE_RATE:
        raise ValueError(
            f"[ERROR] Tasa de muestreo {sample_rate} Hz incompatible con Whisper ({WHISPER_SAMPLE_RATE} Hz). "
            "Remuestrea el audio antes de transcribir."
        )

    segmentos = cargar_archivo(paths["segmentos"])
    if segmentos is None:
        print(f"[ERROR] No se pudo cargar los segmentos desde '{paths['segmentos']}'.")
        return
    segmentos = obtener_fragmentos_asr(segmentos["segmentos"], PERFIL_WHISPER)

    transcripciones = []
    for inicio, fin in tqdm(segmentos, desc="Transcribiendo", unit="segmento"):
        frame_inicio = int(inicio * WHISPER_SAMPLE_RATE)
        frame_fin = int(fin * WHISPER_SAMPLE_RATE)
        audio_seg = audio_data[frame_inicio:frame_fin]

        segments, _ = pipeline.transcribe(
            audio_seg,
            language="es",
            beam_size=5,
            batch_size=batch_size,
            vad_filter=True,
        )
        texto = " ".join(seg.text.strip() for seg in segments)
        transcripciones.append({
            "inicio": inicio,
            "fin": fin,
            "duracion": round(fin - inicio, 2),
            "texto": texto,
        })

    full_text = " ".join(t["texto"] for t in transcripciones)

    data = {
        "texto": full_text,
        "num_segmentos": len(transcripciones),
        "duracion_promedio_segmento": round(sum(t["duracion"] for t in transcripciones) / len(transcripciones), 2) if transcripciones else 0,
        "modelo": f"Whisper {nombre_modelo}",
        "perfil": PERFIL_WHISPER,
        "transcripciones": transcripciones,
    }
    guardar_archivo(paths["transcripciones"], data)