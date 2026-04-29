import soundfile as sf
from faster_whisper import BatchedInferencePipeline, WhisperModel
from tqdm import tqdm

from compartido.json_utils import cargar_archivo, guardar_archivo
from compartido.rutas import DESCARGAS_DIR
from compartido.utils import crear_perfil_hardware, cronometrar
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
WHISPER_BEAM_SIZE = 5
REDONDEO_TIEMPOS = 3


def _obtener_cpu_threads(perfil_hardware):
    cpu_threads = perfil_hardware["cpu_physical_cores"] or perfil_hardware["cpu_logical_cores"] or 1
    return max(1, cpu_threads)


def _resolver_configuracion_whisper(perfil_hardware, batch_size=BATCH_SIZE):
    device_detectado = perfil_hardware["device"]
    ram_gb = perfil_hardware["ram_gb"]
    vram_gb = perfil_hardware["vram_gb"]
    cpu_threads = _obtener_cpu_threads(perfil_hardware)

    if device_detectado == "cuda":
        if vram_gb >= 14:
            compute_type = "float16"
            batch_size = min(batch_size, 16)
        elif vram_gb >= 8:
            compute_type = "int8_float16"
            batch_size = min(batch_size, 8)
        else:
            compute_type = "int8_float16"
            batch_size = min(batch_size, 4)
        device = "cuda"
    else:
        if device_detectado in {"mps", "xpu"}:
            print(f"[ADVERTENCIA] faster-whisper no soporta '{device_detectado}' en esta configuracion. Se usara CPU.")
        device = "cpu"
        compute_type = "int8" if ram_gb >= 8 else "float32"
        batch_size = 2 if ram_gb >= 16 else 1

    return {
        "device": device,
        "compute_type": compute_type,
        "cpu_threads": cpu_threads,
        "num_workers": 1,
        "batch_size": max(1, batch_size),
    }


def _cargar_modelo(nombre=MODELO_DEFAULT, configuracion=None):
    configuracion = configuracion or _resolver_configuracion_whisper(crear_perfil_hardware())
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] Cargando modelo Whisper '{nombre}' "
        f"(device={configuracion['device']}, compute_type={configuracion['compute_type']}, "
        f"cpu_threads={configuracion['cpu_threads']}, batch_size={configuracion['batch_size']})..."
    )
    base = WhisperModel(
        nombre,
        device=configuracion["device"],
        compute_type=configuracion["compute_type"],
        cpu_threads=configuracion["cpu_threads"],
        num_workers=configuracion["num_workers"],
        download_root=str(MODELOS_DIR),
    )
    return BatchedInferencePipeline(model=base), configuracion


def _redondear_tiempo(valor):
    return round(max(0.0, float(valor)), REDONDEO_TIEMPOS)


def _serializar_segmentos_whisper(segmentos_whisper, inicio_fragmento, fin_fragmento):
    segmentos_serializados = []

    for segmento in segmentos_whisper:
        texto = segmento.text.strip()
        if not texto:
            continue

        inicio = _redondear_tiempo(max(inicio_fragmento, inicio_fragmento + float(segmento.start)))
        fin = _redondear_tiempo(min(fin_fragmento, inicio_fragmento + float(segmento.end)))
        if fin <= inicio:
            continue

        segmentos_serializados.append({
            "inicio": inicio,
            "fin": fin,
            "duracion": round(fin - inicio, 2),
            "texto": texto,
        })

    return segmentos_serializados


def _obtener_metadata_ejecucion(configuracion):
    metadata = {
        "device": configuracion["device"],
        "compute_type": configuracion["compute_type"],
        "batch_size": configuracion["batch_size"],
    }

    if configuracion["device"] == "cpu":
        metadata["cpu_threads"] = configuracion["cpu_threads"]
        metadata["num_workers"] = configuracion["num_workers"]

    return metadata


@cronometrar(etiqueta="Transcripcion total")
def transcribir_whisper(paths, nombre_modelo=MODELO_DEFAULT, batch_size=BATCH_SIZE):
    configuracion = _resolver_configuracion_whisper(crear_perfil_hardware(), batch_size=batch_size)
    pipeline, configuracion = _cargar_modelo(nombre_modelo, configuracion)

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
    print(
        f"[INFO] Transcribiendo {len(segmentos)} fragmentos con Whisper "
        f"(device={configuracion['device']}, compute_type={configuracion['compute_type']}, "
        f"batch_size={configuracion['batch_size']})..."
    )

    for inicio, fin in tqdm(segmentos, desc="Transcribiendo", unit="segmento"):
        frame_inicio = int(inicio * WHISPER_SAMPLE_RATE)
        frame_fin = int(fin * WHISPER_SAMPLE_RATE)
        audio_seg = audio_data[frame_inicio:frame_fin]
        if audio_seg.size == 0:
            continue

        segments, _ = pipeline.transcribe(
            audio_seg,
            language="es",
            beam_size=WHISPER_BEAM_SIZE,
            batch_size=configuracion["batch_size"],
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=False,
        )

        segmentos_whisper = _serializar_segmentos_whisper(list(segments), inicio, fin)
        if not segmentos_whisper:
            continue

        transcripciones.extend(segmentos_whisper)

    transcripciones.sort(key=lambda x: x["inicio"])
    full_text = " ".join(t["texto"] for t in transcripciones).strip()

    data = {
        "modelo": f"Whisper {nombre_modelo}",
        "perfil": PERFIL_WHISPER,
        "tiempo_transcripcion": None,
        "speed_up": None,
        "rt_factor": None,
        "num_segmentos": len(transcripciones),
        "duracion_promedio_segmento": round(sum(t["duracion"] for t in transcripciones) / len(transcripciones), 2) if transcripciones else 0,
        "transcripciones": transcripciones,
        "texto": full_text
    }
    data.update(_obtener_metadata_ejecucion(configuracion))
    guardar_archivo(paths["transcripciones"], data)