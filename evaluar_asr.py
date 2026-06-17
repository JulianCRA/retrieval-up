from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import subprocess
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parent
MODELOS_DIR = ROOT / "modelos"
MODELOS_VOSK_DIR = MODELOS_DIR / "vosk"
MODELOS_COHERE_DIR = MODELOS_DIR / "cohere"
RESULTADOS_DIR = ROOT / "resultados" / "asr_eval"

VOSK_MODEL_ID = "vosk-model-es-0.42"
COHERE_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"

_VRAM_POR_SEGMENTO_COHERE_MB = 150
_VRAM_POR_CHUNK_WHISPER_MB = {
    "tiny": 40,
    "base": 90,
    "small": 160,
    "medium": 200,
    "large": 245,
    "large-v2": 245,
    "large-v3": 245,
    "turbo": 245,
}


@dataclass(frozen=True)
class DatasetPreset:
    alias: str
    dataset_id: str
    configs: tuple[str, ...]
    split: str
    audio_field: str
    text_field: str
    id_field: str
    language: str = "es"


PRESETS: dict[str, DatasetPreset] = {
    "mls-es-test": DatasetPreset(
        alias="mls-es-test",
        dataset_id="facebook/multilingual_librispeech",
        configs=("spanish",),
        split="test",
        audio_field="audio",
        text_field="transcript",
        id_field="id",
    ),
    "fleurs-es-test": DatasetPreset(
        alias="fleurs-es-test",
        dataset_id="google/fleurs",
        configs=("es_419", "es_es"),
        split="test",
        audio_field="audio",
        text_field="transcription",
        id_field="id",
    ),
    "common-voice-es-test": DatasetPreset(
        alias="common-voice-es-test",
        dataset_id="fixie-ai/common_voice_17_0",
        configs=("es",),
        split="test",
        audio_field="audio",
        text_field="sentence",
        id_field="path",
    ),
}


class JsonlCheckpoint:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.run_dir / "predicciones.jsonl"
        self.state_path = self.run_dir / "checkpoint.json"
        self.summary_path = self.run_dir / "resumen.json"
        self.csv_path = self.run_dir / "predicciones.csv"

    def reset(self) -> None:
        for path in (self.records_path, self.state_path, self.summary_path, self.csv_path):
            if path.exists():
                path.unlink()

    def load_records(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        if not self.records_path.exists():
            return records
        with self.records_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records[str(record["example_id"])] = record
        return records

    def append_records(self, records: Iterable[dict[str, Any]]) -> None:
        with self.records_path.open("a", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_state(self, state: dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_summary(self, summary: dict[str, Any]) -> None:
        self.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_csv(self, records: list[dict[str, Any]]) -> None:
        fieldnames = [
            "example_id",
            "index",
            "status",
            "audio_seconds",
            "infer_seconds",
            "rtf",
            "reference_raw",
            "prediction_raw",
            "reference_norm",
            "prediction_norm",
            "error",
        ]
        with self.csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow({key: record.get(key) for key in fieldnames})


def _safe_slug(texto: str) -> str:
    texto = texto.strip().lower().replace(":", "-")
    texto = re.sub(r"[^a-z0-9._-]+", "-", texto)
    return texto.strip("-") or "run"


def _normalize_whitespace(texto: str) -> str:
    return " ".join((texto or "").split())


def _pick_existing(example: dict[str, Any], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in example and example[candidate] is not None:
            return candidate
    return None


def normalize_text(texto: str, *, strip_punctuation: bool, strip_accents: bool) -> str:
    texto = (texto or "").lower()
    if strip_accents:
        texto = unicodedata.normalize("NFD", texto)
        texto = "".join(ch for ch in texto if unicodedata.category(ch) != "Mn")
    if strip_punctuation:
        chars: list[str] = []
        for ch in texto:
            cat = unicodedata.category(ch)
            if cat and cat[0] in {"P", "S"}:
                chars.append(" ")
            else:
                chars.append(ch)
        texto = "".join(chars)
    return _normalize_whitespace(texto)


def _edit_distance(seq1: Sequence[str], seq2: Sequence[str]) -> int:
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1
    if not seq2:
        return len(seq1)
    previous = list(range(len(seq2) + 1))
    for i, left in enumerate(seq1, 1):
        current = [i]
        for j, right in enumerate(seq2, 1):
            insertions = previous[j] + 1
            deletions = current[j - 1] + 1
            substitutions = previous[j - 1] + (left != right)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]


def _wer(preds: Sequence[str], refs: Sequence[str]) -> float:
    errors = 0
    ref_words = 0
    for pred, ref in zip(preds, refs):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        errors += _edit_distance(pred_tokens, ref_tokens)
        ref_words += len(ref_tokens)
    return 0.0 if ref_words == 0 else errors / ref_words


def _cer(preds: Sequence[str], refs: Sequence[str]) -> float:
    errors = 0
    ref_chars = 0
    for pred, ref in zip(preds, refs):
        errors += _edit_distance(list(pred), list(ref))
        ref_chars += len(ref)
    return 0.0 if ref_chars == 0 else errors / ref_chars


def detectar_gpu_nvidia() -> tuple[bool, int]:
    if shutil.which("nvidia-smi") is None:
        return False, 0
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        vram_mb = int(result.stdout.strip().splitlines()[0])
        return True, round(vram_mb / 1024)
    except Exception:
        return False, 0


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("[ERROR] Falta torch en el entorno actual.") from exc
    return torch


def crear_perfil_hardware(forzar_cpu: bool = False) -> dict[str, Any]:
    if forzar_cpu:
        return {
            "device": "cpu",
            "vram_gb": 0,
            "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3)),
            "cpu_physical_cores": psutil.cpu_count(logical=False) or 1,
            "cpu_logical_cores": psutil.cpu_count(logical=True) or 1,
        }

    has_nvidia, vram_gb = detectar_gpu_nvidia()
    if has_nvidia:
        device = "cuda"
    else:
        try:
            torch = _load_torch()
            if hasattr(torch, "is_xpu_available") and torch.is_xpu_available():
                device = "xpu"
                vram_gb = 4
            elif torch.cuda.is_available():
                device = "cuda"
                vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
            else:
                device = "cpu"
                vram_gb = 0
        except SystemExit:
            device = "cpu"
            vram_gb = 0

    return {
        "device": device,
        "vram_gb": vram_gb,
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3)),
        "cpu_physical_cores": psutil.cpu_count(logical=False) or 1,
        "cpu_logical_cores": psutil.cpu_count(logical=True) or 1,
    }


def _resample_audio(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr <= 0 or target_sr <= 0 or src_sr == target_sr:
        return np.ascontiguousarray(audio.astype(np.float32, copy=False))
    if audio.size == 0:
        return np.ascontiguousarray(audio.astype(np.float32, copy=False))
    duration = audio.shape[0] / float(src_sr)
    target_len = max(1, int(round(duration * target_sr)))
    if target_len == audio.shape[0]:
        return np.ascontiguousarray(audio.astype(np.float32, copy=False))
    src_positions = np.linspace(0.0, audio.shape[0] - 1, num=audio.shape[0], dtype=np.float64)
    target_positions = np.linspace(0.0, audio.shape[0] - 1, num=target_len, dtype=np.float64)
    resampled = np.interp(target_positions, src_positions, audio.astype(np.float64, copy=False))
    return np.ascontiguousarray(resampled.astype(np.float32, copy=False))


def _to_float32_mono(audio_obj: Any, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    if isinstance(audio_obj, dict) and "array" in audio_obj:
        array = np.asarray(audio_obj["array"], dtype=np.float32)
        sr = int(audio_obj["sampling_rate"])
    elif isinstance(audio_obj, dict) and audio_obj.get("bytes") is not None:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise SystemExit("[ERROR] Falta soundfile en el entorno actual.") from exc
        array, sr = sf.read(io.BytesIO(audio_obj["bytes"]), dtype="float32", always_2d=False)
    elif isinstance(audio_obj, dict) and audio_obj.get("path"):
        try:
            import soundfile as sf
        except ImportError as exc:
            raise SystemExit("[ERROR] Falta soundfile en el entorno actual.") from exc
        array, sr = sf.read(str(audio_obj["path"]), dtype="float32", always_2d=False)
    else:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise SystemExit("[ERROR] Falta soundfile en el entorno actual.") from exc
        array, sr = sf.read(str(audio_obj), dtype="float32", always_2d=False)
    if array.ndim == 2:
        array = array.mean(axis=1)
    array = np.ascontiguousarray(array.astype(np.float32, copy=False))
    if target_sr is not None and sr != target_sr:
        array = _resample_audio(array, sr, target_sr)
        sr = target_sr
    return array, sr


def _pcm16_bytes(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16).tobytes()


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_records = [record for record in records if record.get("status") == "ok"]
    refs_raw = [record["reference_raw"] for record in ok_records]
    preds_raw = [record["prediction_raw"] for record in ok_records]
    refs_norm = [record["reference_norm"] for record in ok_records]
    preds_norm = [record["prediction_norm"] for record in ok_records]
    audio_seconds = sum(float(record.get("audio_seconds") or 0.0) for record in ok_records)
    infer_seconds = sum(float(record.get("infer_seconds") or 0.0) for record in ok_records)
    return {
        "examples_completed": len(records),
        "examples_failed": sum(1 for record in records if record.get("status") != "ok"),
        "wer_raw": round(_wer(preds_raw, refs_raw) * 100, 3) if refs_raw else None,
        "cer_raw": round(_cer(preds_raw, refs_raw) * 100, 3) if refs_raw else None,
        "wer_norm": round(_wer(preds_norm, refs_norm) * 100, 3) if refs_norm else None,
        "cer_norm": round(_cer(preds_norm, refs_norm) * 100, 3) if refs_norm else None,
        "audio_seconds": round(audio_seconds, 3),
        "infer_seconds": round(infer_seconds, 3),
        "rtf": None if audio_seconds <= 0 else round(infer_seconds / audio_seconds, 4),
        "speedup": None if infer_seconds <= 0 else round(audio_seconds / infer_seconds, 3),
    }


def _total_batches(total_items: int, batch_size: int) -> int:
    if batch_size <= 0:
        return 0
    return (total_items + batch_size - 1) // batch_size


def _backend_runtime_summary(info: dict[str, Any]) -> str:
    parts = [f"backend={info.get('backend')}"]
    if info.get("model_id"):
        parts.append(f"model={info['model_id']}")
    if info.get("device"):
        parts.append(f"device={info['device']}")
    if info.get("compute_type"):
        parts.append(f"compute={info['compute_type']}")
    if info.get("torch_dtype"):
        parts.append(f"dtype={info['torch_dtype']}")
    if info.get("batch_size") is not None:
        parts.append(f"batch_size={info['batch_size']}")
    if info.get("workers") is not None:
        parts.append(f"workers={info['workers']}")
    if info.get("cpu_threads") is not None:
        parts.append(f"cpu_threads={info['cpu_threads']}")
    if info.get("vram_gb") is not None:
        parts.append(f"vram_gb={info['vram_gb']}")
    return " | ".join(parts)


def _backend_batch_summary(stats: dict[str, Any] | None) -> str | None:
    if not stats:
        return None
    parts: list[str] = []
    if stats.get("stage"):
        parts.append(f"stage={stats['stage']}")
    if stats.get("batch_examples") is not None:
        parts.append(f"batch_examples={stats['batch_examples']}")
    if stats.get("audio_seconds") is not None:
        parts.append(f"audio={stats['audio_seconds']:.1f}s")
    if stats.get("processor_seconds") is not None:
        parts.append(f"processor={stats['processor_seconds']:.2f}s")
    if stats.get("transfer_seconds") is not None:
        parts.append(f"transfer={stats['transfer_seconds']:.2f}s")
    if stats.get("generate_seconds") is not None:
        parts.append(f"generate={stats['generate_seconds']:.2f}s")
    if stats.get("decode_seconds") is not None:
        parts.append(f"decode={stats['decode_seconds']:.2f}s")
    if stats.get("backend_seconds") is not None:
        parts.append(f"backend_total={stats['backend_seconds']:.2f}s")
    if stats.get("input_shape"):
        parts.append(f"input_shape={stats['input_shape']}")
    return " | ".join(parts) if parts else None


class VoskBackend:
    def __init__(self, workers: int):
        try:
            import vosk
        except ImportError as exc:
            raise SystemExit("[ERROR] Falta vosk en el entorno actual.") from exc
        self.vosk = vosk
        self.workers = workers
        self.model_path = MODELOS_VOSK_DIR / VOSK_MODEL_ID
        if not self.model_path.exists():
            raise SystemExit(f"[ERROR] No se encontró el modelo Vosk en '{self.model_path}'.")
        vosk.SetLogLevel(-1)
        self.model = vosk.Model(str(self.model_path))

    def _transcribe_one(self, audio: np.ndarray, sr: int) -> str:
        rec = self.vosk.KaldiRecognizer(self.model, sr)
        rec.AcceptWaveform(_pcm16_bytes(audio))
        result = json.loads(rec.FinalResult())
        return (result.get("text") or "").strip()

    def transcribe_batch(self, audios: list[np.ndarray], srs: list[int], language: str) -> list[str]:
        del language
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            return list(executor.map(self._transcribe_one, audios, srs))

    def describe(self) -> dict[str, Any]:
        return {"backend": "vosk", "model_id": VOSK_MODEL_ID, "workers": self.workers}


class WhisperBackend:
    def __init__(self, variant: str, perfil: dict[str, Any], batch_size: int | None):
        try:
            from faster_whisper import BatchedInferencePipeline, WhisperModel
        except ImportError as exc:
            raise SystemExit("[ERROR] Falta faster-whisper en el entorno actual.") from exc
        self.variant = variant
        self.perfil = perfil
        self.params = self._compute_params(batch_size)
        self.model = WhisperModel(
            variant,
            device=self.params["device"],
            compute_type=self.params["compute_type"],
            cpu_threads=self.params.get("cpu_threads", 1),
            num_workers=self.params.get("num_workers", 1),
        )
        self.pipeline = BatchedInferencePipeline(model=self.model)

    def _vram_por_chunk(self) -> int:
        key = next((item for item in _VRAM_POR_CHUNK_WHISPER_MB if self.variant.startswith(item)), "large")
        return _VRAM_POR_CHUNK_WHISPER_MB[key]

    def _compute_params(self, batch_size: int | None) -> dict[str, Any]:
        dev = self.perfil.get("device", "cpu")
        params: dict[str, Any] = {"device": dev}
        if dev in ("cuda", "mps", "xpu"):
            total_vram = self.perfil.get("vram_gb", 0) * 1024
            free_vram = total_vram * 0.85
            optimal = int(free_vram // self._vram_por_chunk()) if self._vram_por_chunk() else 1
            params["batch_size"] = max(1, min(optimal, 64))
            params["compute_type"] = "float16" if dev == "cuda" else "int8_float16"
        else:
            threads = self.perfil.get("cpu_physical_cores") or self.perfil.get("cpu_logical_cores") or 1
            params["cpu_threads"] = threads
            params["num_workers"] = 1
            params["batch_size"] = 2 if self.perfil.get("ram_gb", 0) >= 16 else 1
            params["compute_type"] = "int8"
        if batch_size is not None:
            params["batch_size"] = batch_size
        return params

    def _concat_batch(self, audios: list[np.ndarray], sr: int) -> tuple[np.ndarray, list[dict[str, float]], list[tuple[float, float]]]:
        gap = np.zeros(int(sr * 0.25), dtype=np.float32)
        pieces: list[np.ndarray] = []
        clip_timestamps: list[dict[str, float]] = []
        ranges: list[tuple[float, float]] = []
        cursor = 0
        for audio in audios:
            if pieces:
                pieces.append(gap)
                cursor += len(gap)
            start = cursor / sr
            pieces.append(audio)
            cursor += len(audio)
            end = cursor / sr
            clip_timestamps.append({"start": start, "end": end})
            ranges.append((start, end))
        return np.concatenate(pieces), clip_timestamps, ranges

    def transcribe_batch(self, audios: list[np.ndarray], srs: list[int], language: str) -> list[str]:
        if not audios:
            return []
        sr = srs[0]
        if any(item != sr for item in srs):
            raise ValueError("Whisper requiere una misma frecuencia de muestreo por lote.")
        merged, clip_timestamps, ranges = self._concat_batch(audios, sr)
        segments, _info = self.pipeline.transcribe(
            merged,
            batch_size=min(self.params["batch_size"], len(audios)),
            language=language,
            without_timestamps=True,
            log_progress=False,
            clip_timestamps=clip_timestamps,
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        texts: list[list[str]] = [[] for _ in audios]
        for segment in list(segments):
            mid = (float(segment.start) + float(segment.end)) / 2.0
            for idx, (start, end) in enumerate(ranges):
                if start <= mid <= end:
                    texts[idx].append(segment.text.strip())
                    break
        return [" ".join(part for part in group if part).strip() for group in texts]

    def describe(self) -> dict[str, Any]:
        return {"backend": "whisper", "model_id": f"whisper-{self.variant}", **self.params}


class CohereBackend:
    def __init__(self, perfil: dict[str, Any], batch_size: int):
        try:
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        except ImportError as exc:
            raise SystemExit("[ERROR] Falta transformers en el entorno actual.") from exc
        self.torch = _load_torch()
        self.AutoProcessor = AutoProcessor
        self.CohereAsrForConditionalGeneration = CohereAsrForConditionalGeneration
        self.perfil = perfil
        self.batch_size_override = batch_size
        self.params = self._compute_params()
        self.last_batch_stats: dict[str, Any] | None = None
        self.processor = AutoProcessor.from_pretrained(COHERE_MODEL_ID, cache_dir=str(MODELOS_COHERE_DIR))
        torch_dtype = getattr(self.torch, self.params["torch_dtype"])
        if self.params["device"] == "cpu":
            self.torch.set_num_threads(self.params.get("cpu_threads", 1))
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            COHERE_MODEL_ID,
            torch_dtype=torch_dtype,
            device_map="cuda:0" if self.params["device"] == "cuda" else self.params["device"],
            cache_dir=str(MODELOS_COHERE_DIR),
        )
        self.model.eval()
        self.model_device = next(self.model.parameters()).device

    def _compute_params(self) -> dict[str, Any]:
        device = self.perfil["device"]
        params = {"device": device}
        if device in ("cuda", "mps", "xpu"):
            params["torch_dtype"] = "float16" if device == "cuda" else "float32"
            params["batch_size"] = self.batch_size_override
        else:
            threads = self.perfil.get("cpu_physical_cores") or self.perfil.get("cpu_logical_cores") or 1
            params["torch_dtype"] = "float32"
            params["batch_size"] = 1
            params["cpu_threads"] = min(threads, 24)
        return params

    def transcribe_batch(self, audios: list[np.ndarray], srs: list[int], language: str) -> list[str]:
        if not audios:
            return []
        sr = srs[0]
        if any(item != sr for item in srs):
            raise ValueError("Cohere requiere una misma frecuencia de muestreo por lote.")
        audio_seconds = sum(len(audio) / float(sr) for audio in audios)
        self.last_batch_stats = {
            "stage": "start",
            "batch_examples": len(audios),
            "audio_seconds": audio_seconds,
        }
        torch_dtype = getattr(self.torch, self.params["torch_dtype"])
        processor_t0 = time.perf_counter()
        inputs = self.processor(
            audios,
            sampling_rate=sr,
            return_tensors="pt",
            language=language,
            padding=True,
        )
        processor_seconds = time.perf_counter() - processor_t0
        transfer_t0 = time.perf_counter()
        inputs = {
            key: (
                value.to(device=self.model_device, dtype=torch_dtype) if value.is_floating_point() else value.to(device=self.model_device)
            )
            if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        transfer_seconds = time.perf_counter() - transfer_t0
        input_shape = None
        for value in inputs.values():
            shape = getattr(value, "shape", None)
            if shape is not None:
                input_shape = tuple(int(dim) for dim in shape)
                break
        generate_t0 = time.perf_counter()
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=256)
        generate_seconds = time.perf_counter() - generate_t0
        decode_t0 = time.perf_counter()
        decoded = [self.processor.decode(ids, skip_special_tokens=True).strip() for ids in output_ids]
        decode_seconds = time.perf_counter() - decode_t0
        self.last_batch_stats = {
            "stage": "done",
            "batch_examples": len(audios),
            "audio_seconds": audio_seconds,
            "processor_seconds": processor_seconds,
            "transfer_seconds": transfer_seconds,
            "generate_seconds": generate_seconds,
            "decode_seconds": decode_seconds,
            "backend_seconds": processor_seconds + transfer_seconds + generate_seconds + decode_seconds,
            "input_shape": input_shape,
        }
        return decoded

    def describe(self) -> dict[str, Any]:
        return {"backend": "cohere", "model_id": COHERE_MODEL_ID, **self.params}


def _build_backend(model_name: str, args: argparse.Namespace, perfil: dict[str, Any]):
    if model_name == "vosk":
        workers = args.vosk_workers or perfil.get("cpu_physical_cores") or 1
        return VoskBackend(workers=workers)
    if model_name == "cohere":
        return CohereBackend(perfil=perfil, batch_size=args.cohere_batch_size)
    if model_name.startswith("whisper:"):
        return WhisperBackend(variant=model_name.split(":", 1)[1], perfil=perfil, batch_size=args.whisper_batch_size)
    raise SystemExit(f"[ERROR] Modelo no soportado: {model_name}")


def _resolve_dataset(args: argparse.Namespace) -> tuple[str, list[str], str, str, str, str, str, str]:
    if args.preset:
        preset = PRESETS[args.preset]
        return (
            preset.dataset_id,
            list(preset.configs),
            preset.split,
            preset.audio_field,
            preset.text_field,
            preset.id_field,
            preset.language,
            preset.alias,
        )
    if not args.dataset_id:
        raise SystemExit("[ERROR] Debes indicar --preset o --dataset-id.")
    alias = args.run_name or _safe_slug(f"{args.dataset_id}-{args.config or 'default'}-{args.split}")
    return (
        args.dataset_id,
        [args.config],
        args.split,
        args.audio_field,
        args.text_field,
        args.id_field,
        args.language,
        alias,
    )


def _load_dataset_rows(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise SystemExit("[ERROR] Falta datasets en el entorno actual.") from exc

    dataset_id, configs, split, audio_field, text_field, id_field, language, alias = _resolve_dataset(args)
    dataset = None
    resolved_config = None
    last_error: Exception | None = None
    load_mode = "stream"
    for config in configs:
        try:
            if args.prepare_dataset:
                dataset = load_dataset(dataset_id, config, split=split)
                dataset = dataset.cast_column(audio_field, Audio(sampling_rate=args.sampling_rate, decode=False))
                load_mode = "prepare"
            else:
                dataset_stream = load_dataset(dataset_id, config, split=split, streaming=True)
                if hasattr(dataset_stream, "cast_column"):
                    dataset_stream = dataset_stream.cast_column(audio_field, Audio(sampling_rate=args.sampling_rate, decode=False))
                dataset = []
                for sample in dataset_stream:
                    dataset.append(sample)
                    if args.limit is not None and len(dataset) >= args.limit:
                        break
                if args.limit is not None:
                    print(f"[INFO] Cargado subconjunto streaming de {len(dataset)} ejemplo(s) para {dataset_id}:{config}:{split}.")
                else:
                    print(f"[INFO] Cargado split completo por streaming de {len(dataset)} ejemplo(s) para {dataset_id}:{config}:{split}.")
            resolved_config = config
            break
        except Exception as exc:
            last_error = exc
    if dataset is None:
        raise SystemExit(f"[ERROR] No se pudo cargar {dataset_id} configs={configs}: {last_error}")
    return {
        "dataset": dataset,
        "dataset_id": dataset_id,
        "config": resolved_config,
        "split": split,
        "audio_field": audio_field,
        "text_field": text_field,
        "id_field": id_field,
        "language": language,
        "alias": alias,
        "load_mode": load_mode,
    }


def _iter_pending_examples(dataset, audio_field: str, text_field: str, id_field: str, existing: set[str]):
    text_candidates = (text_field, "text", "transcript", "sentence", "transcription", "raw_transcription")
    id_candidates = (id_field, "id", "file", "path", "original_path")
    path_candidates = ("path", "file", "original_path")
    resolved_rows: list[dict[str, Any]] = []
    id_counts: dict[str, int] = {}
    for index in range(len(dataset)):
        example = dataset[index]
        resolved_text_field = _pick_existing(example, text_candidates)
        resolved_id_field = _pick_existing(example, id_candidates)
        if resolved_text_field is None or resolved_id_field is None:
            continue
        reference = str(example[resolved_text_field] or "").strip()
        if not reference:
            continue
        raw_id = str(example[resolved_id_field])
        resolved_path_field = _pick_existing(example, path_candidates)
        path_value = str(example[resolved_path_field]) if resolved_path_field is not None else ""
        resolved_rows.append({
            "index": index,
            "raw_id": raw_id,
            "path_value": path_value,
            "audio_obj": example[audio_field],
            "reference": reference,
        })
        id_counts[raw_id] = id_counts.get(raw_id, 0) + 1

    for row in resolved_rows:
        example_id = row["raw_id"]
        if id_counts.get(example_id, 0) > 1:
            if row["path_value"]:
                example_id = f"{example_id}|{row['path_value']}"
            else:
                example_id = f"{example_id}#{row['index']}"
        if example_id in existing:
            continue
        yield {
            "index": row["index"],
            "example_id": example_id,
            "audio_obj": row["audio_obj"],
            "reference": row["reference"],
        }


def _chunked(items: list[dict[str, Any]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _record_ok(item: dict[str, Any], prediction: str, infer_seconds: float, strip_punctuation: bool, strip_accents: bool) -> dict[str, Any]:
    ref_raw = _normalize_whitespace(item["reference"])
    pred_raw = _normalize_whitespace(prediction)
    ref_norm = normalize_text(ref_raw, strip_punctuation=strip_punctuation, strip_accents=strip_accents)
    pred_norm = normalize_text(pred_raw, strip_punctuation=strip_punctuation, strip_accents=strip_accents)
    audio_seconds = item["audio_seconds"]
    return {
        "example_id": item["example_id"],
        "index": item["index"],
        "status": "ok",
        "audio_seconds": round(audio_seconds, 3),
        "infer_seconds": round(infer_seconds, 3),
        "rtf": None if audio_seconds <= 0 else round(infer_seconds / audio_seconds, 4),
        "reference_raw": ref_raw,
        "prediction_raw": pred_raw,
        "reference_norm": ref_norm,
        "prediction_norm": pred_norm,
        "error": None,
    }


def _record_error(item: dict[str, Any], exc: Exception, strip_punctuation: bool, strip_accents: bool) -> dict[str, Any]:
    ref_raw = _normalize_whitespace(item["reference"])
    ref_norm = normalize_text(ref_raw, strip_punctuation=strip_punctuation, strip_accents=strip_accents)
    return {
        "example_id": item["example_id"],
        "index": item["index"],
        "status": "error",
        "audio_seconds": round(item["audio_seconds"], 3),
        "infer_seconds": 0.0,
        "rtf": None,
        "reference_raw": ref_raw,
        "prediction_raw": "",
        "reference_norm": ref_norm,
        "prediction_norm": "",
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_model(args: argparse.Namespace, dataset_info: dict[str, Any], model_name: str, perfil: dict[str, Any]) -> Path:
    run_dir = Path(args.output_root) / dataset_info["alias"] / _safe_slug(model_name)
    checkpoint = JsonlCheckpoint(run_dir)
    if args.reset:
        checkpoint.reset()
    existing_records = checkpoint.load_records()
    pending = list(
        _iter_pending_examples(
            dataset_info["dataset"],
            dataset_info["audio_field"],
            dataset_info["text_field"],
            dataset_info["id_field"],
            set(existing_records),
        )
    )
    backend = _build_backend(model_name, args, perfil)
    meta = {
        "dataset": dataset_info["dataset_id"],
        "config": dataset_info["config"],
        "split": dataset_info["split"],
        "alias": dataset_info["alias"],
        "language": dataset_info["language"],
        "sampling_rate": args.sampling_rate,
        "hardware": perfil,
        "backend": backend.describe(),
        "completed_before_run": len(existing_records),
        "pending_examples": len(pending),
        "normalization": {
            "strip_punctuation": not args.keep_punctuation,
            "strip_accents": args.strip_accents,
        },
    }
    checkpoint.write_state(meta)

    if not pending:
        records = sorted(existing_records.values(), key=lambda item: item["index"])
        summary = {**meta, **_summarize(records)}
        checkpoint.write_summary(summary)
        checkpoint.write_csv(records)
        return checkpoint.summary_path

    backend_info = backend.describe()
    if backend_info["backend"] == "cohere":
        logical_batch_size = max(1, int(backend_info.get("batch_size", args.cohere_batch_size)))
    elif backend_info["backend"] == "whisper":
        logical_batch_size = max(1, int(backend_info.get("batch_size", 8)))
    else:
        logical_batch_size = max(1, int(args.vosk_batch_size))

    total_pending = len(pending)
    total_batch_count = _total_batches(total_pending, logical_batch_size)
    total_target = len(existing_records) + total_pending

    print(f"[INFO] {model_name}: {len(existing_records)} resueltos, {total_pending} pendientes.")
    print(f"[INFO] {model_name}: {_backend_runtime_summary(backend_info)}")
    print(
        f"[INFO] {model_name}: dataset={dataset_info['dataset_id']} "
        f"config={dataset_info['config']} split={dataset_info['split']} "
        f"batches={total_batch_count} logical_batch_size={logical_batch_size}"
    )

    for batch_index, batch in enumerate(_chunked(pending, logical_batch_size), 1):
        print(
            f"[INFO] {model_name}: lote {batch_index}/{total_batch_count} "
            f"preparando {len(batch)} ejemplo(s)..."
        )
        prepared: list[dict[str, Any]] = []
        audios: list[np.ndarray] = []
        srs: list[int] = []
        for item in batch:
            audio_array, sr = _to_float32_mono(item["audio_obj"], target_sr=args.sampling_rate)
            prepared.append({
                **item,
                "audio_seconds": len(audio_array) / float(sr),
            })
            audios.append(audio_array)
            srs.append(sr)

        batch_audio_seconds = sum(item["audio_seconds"] for item in prepared)
        print(
            f"[INFO] {model_name}: lote {batch_index}/{total_batch_count} "
            f"inferencia iniciada | audio={batch_audio_seconds:.1f}s"
        )

        batch_t0 = time.perf_counter()
        elapsed = 0.0
        try:
            predictions = backend.transcribe_batch(audios, srs, dataset_info["language"])
            elapsed = time.perf_counter() - batch_t0
            seconds_per_item = elapsed / max(len(prepared), 1)
            records_to_append = [
                _record_ok(item, pred, seconds_per_item, not args.keep_punctuation, args.strip_accents)
                for item, pred in zip(prepared, predictions)
            ]
        except Exception as exc:
            elapsed = time.perf_counter() - batch_t0
            print(f"[ERROR] Lote fallido en {model_name}: {exc}")
            records_to_append = [
                _record_error(item, exc, not args.keep_punctuation, args.strip_accents)
                for item in prepared
            ]
        checkpoint.append_records(records_to_append)
        for record in records_to_append:
            existing_records[record["example_id"]] = record
        records = sorted(existing_records.values(), key=lambda item: item["index"])
        summary = {**meta, **_summarize(records)}
        checkpoint.write_state(summary)
        checkpoint.write_summary(summary)
        done_now = len(records)
        batch_rtf = elapsed / batch_audio_seconds if batch_audio_seconds > 0 else 0.0
        backend_batch_summary = _backend_batch_summary(getattr(backend, "last_batch_stats", None))
        message = (
            f"[INFO] {model_name}: lote {batch_index}/{total_batch_count} listo | "
            f"elapsed={elapsed:.2f}s | audio={batch_audio_seconds:.1f}s | "
            f"rtf={batch_rtf:.4f} | completados={done_now}/{total_target}"
        )
        if backend_batch_summary:
            message += f" | {backend_batch_summary}"
        print(message)

    records = sorted(existing_records.values(), key=lambda item: item["index"])
    summary = {**meta, **_summarize(records)}
    checkpoint.write_summary(summary)
    checkpoint.write_csv(records)
    return checkpoint.summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluar_asr.py",
        description="Script standalone para evaluar Vosk, Whisper y Cohere con checkpoints reanudables.",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Preset de benchmark listo para usar.")
    parser.add_argument("--dataset-id", help="Dataset HF o script local compatible con datasets.load_dataset().")
    parser.add_argument("--config", default=None, help="Config/subset del dataset.")
    parser.add_argument("--split", default="test", help="Split a evaluar.")
    parser.add_argument("--audio-field", default="audio", help="Campo de audio.")
    parser.add_argument("--text-field", default="text", help="Campo de referencia.")
    parser.add_argument("--id-field", default="id", help="Campo identificador único.")
    parser.add_argument("--language", default="es", help="Idioma pasado al decoder cuando aplica.")
    parser.add_argument("--modelo", action="append", dest="modelos", required=True, help="Modelo: vosk | cohere | whisper:turbo | whisper:small | whisper:base")
    parser.add_argument("--limit", type=int, default=None, help="Limita la cantidad de ejemplos cargados.")
    parser.add_argument("--prepare-dataset", action="store_true", help="Usa load_dataset normal en vez de streaming. Puede preparar/cachear todos los splits del dataset.")
    parser.add_argument("--sampling-rate", type=int, default=16000, help="Frecuencia de muestreo objetivo.")
    parser.add_argument("--output-root", default=str(RESULTADOS_DIR), help="Directorio raíz para checkpoints y resúmenes.")
    parser.add_argument("--run-name", default=None, help="Alias manual para el benchmark genérico.")
    parser.add_argument("--reset", action="store_true", help="Borra el checkpoint del run antes de arrancar.")
    parser.add_argument("--forzar-cpu", action="store_true", help="Fuerza CPU aunque haya GPU.")
    parser.add_argument("--strip-accents", action="store_true", help="Quita tildes al normalizar.")
    parser.add_argument("--keep-punctuation", action="store_true", help="Mantiene puntuación en la normalización en lugar de quitarla.")
    parser.add_argument("--whisper-batch-size", type=int, default=None, help="Override del batch size de Whisper.")
    parser.add_argument("--cohere-batch-size", type=int, default=32, help="Batch size de Cohere. Default: 32.")
    parser.add_argument("--vosk-workers", type=int, default=None, help="Workers CPU para Vosk. Default: cores físicos.")
    parser.add_argument("--vosk-batch-size", type=int, default=64, help="Tamaño lógico de lote para checkpointing con Vosk.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    perfil = crear_perfil_hardware(forzar_cpu=args.forzar_cpu)
    dataset_info = _load_dataset_rows(args)
    print(
        "[INFO] Hardware detectado: "
        f"device={perfil['device']} | vram_gb={perfil['vram_gb']} | "
        f"ram_gb={perfil['ram_gb']} | cpu_physical_cores={perfil['cpu_physical_cores']} | "
        f"cpu_logical_cores={perfil['cpu_logical_cores']}"
    )
    print(
        f"[INFO] Dataset: {dataset_info['dataset_id']} | config={dataset_info['config']} | "
        f"split={dataset_info['split']} | alias={dataset_info['alias']} | "
        f"modo_carga={dataset_info['load_mode']} | ejemplos_cargados={len(dataset_info['dataset'])}"
    )
    for model_name in args.modelos:
        summary_path = run_model(args, dataset_info, model_name, perfil)
        print(f"[OK] Resumen guardado en '{summary_path}'.")


if __name__ == "__main__":
    main()
