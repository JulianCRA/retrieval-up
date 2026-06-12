"""Registro de modelos de embedding soportados y utilidades de tokenizacion.

El fragmentador y el vectorizador deben coincidir en el modelo objetivo. Este
modulo centraliza:
  - metadatos por modelo (hf_id, ventana, dim, prefijos, etc.)
  - un `Sizer` que cuenta tokens REALES con el tokenizador del modelo
  - una factoria `cargar_sentence_transformer` para evaluar embeddings
"""
from dataclasses import dataclass, field
from typing import Optional

from compartido.rutas import MODELOS_EMBEDDINGS_DIR

MODELOS_DIR = MODELOS_EMBEDDINGS_DIR
MODELOS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EmbedderSpec:
	id_corto: str
	hf_id: str
	max_seq_len: int          # ventana de contexto del modelo (tokens)
	chunk_default: int        # tamano recomendado de chunk (tokens reales)
	dim: int                  # dimension del vector de salida
	prefijo_passage: str = "" # prefijo obligatorio en indexacion (passages)
	prefijo_query: str = ""   # prefijo obligatorio en consulta (queries)
	tarea_passage: str = ""   # task= kwarg para encode() en indexacion (jina-v3 y similares)
	tarea_query: str = ""     # task= kwarg para encode() en consulta (jina-v3 y similares)
	trust_remote_code: bool = False
	notas: str = ""


# Modelos soportados (orden ~calidad / coste).
EMBEDDERS: dict[str, EmbedderSpec] = {
	"qwen3-0.6b": EmbedderSpec(
		id_corto="qwen3-0.6b",
		hf_id="Qwen/Qwen3-Embedding-0.6B",
		max_seq_len=32768,
		chunk_default=512,
		dim=1024,
		prefijo_query = (
			"Instruct: Match the search query with relevant passages\n"
			"Query: "
		),
		notas=(
			"Fuerte: top de MTEB multilingue, espanol muy solido, ventana enorme (32k). "
			"Debil: ~1.2GB en disco, ~2-3GB de RAM en inferencia, lento en CPU "
			"(~0.5-2s por chunk segun longitud). "
			"HW: ideal con GPU (>=4GB VRAM); en CPU requiere >=8GB RAM y paciencia."
		),
	),
	"bge-m3": EmbedderSpec(
		id_corto="bge-m3",
		hf_id="BAAI/bge-m3",
		max_seq_len=8192,
		chunk_default=512,
		dim=1024,
		notas=(
			"Fuerte: calidad excelente multilingue (>100 idiomas, espanol incluido), "
			"ventana 8k, soporta dense + sparse + multi-vector (ColBERT) en el mismo "
			"forward; muy robusto para retrieval. "
			"Debil: ~2.2GB FP32 / ~1.1GB FP16 en disco; encode mas lento que e5-base. "
			"HW: GPU con 2-3GB VRAM ideal; en CPU funciona con >=6GB RAM en batches chicos."
		),
	),
	"e5-large-instruct": EmbedderSpec(
		id_corto="e5-large-instruct",
		hf_id="intfloat/multilingual-e5-large-instruct",
		max_seq_len=512,
		chunk_default=480,
		dim=1024,
		prefijo_query=(
			"Instruct: Retrieve semantically relevant passages for the search query in Spanish\n"
			"Query: "
		),
		notas=(
			"Fuerte: muy buena calidad en espanol/EN, instruct (acepta instrucciones "
			"por tarea en el query). 1024-dim. "
			"Debil: ventana de solo 512 tokens (chunks chicos obligados); ~2.2GB FP32. "
			"Sin la instruccion adecuada del lado query la calidad cae notoriamente. "
			"HW: ~4GB RAM CPU, GPU con 2GB VRAM lo acelera bastante."
		),
	),
	"granite-107m": EmbedderSpec(
		id_corto="granite-107m",
		hf_id="ibm-granite/granite-embedding-107m-multilingual",
		max_seq_len=512,
		chunk_default=480,
		dim=384,
		notas=(
			"Fuerte: muy liviano (~210MB), rapidisimo en CPU (~10-30ms por chunk), "
			"384-dim baja el coste de indexado y memoria del vector store. "
			"Cubre 12 idiomas incluyendo espanol. "
			"Debil: calidad notoriamente inferior a bge-m3/qwen3 en consultas ambiguas "
			"o de larga cola; ventana 512. "
			"HW: corre comodo con 2GB RAM, sin GPU."
		),
	),
	"jina-v3": EmbedderSpec(
		id_corto="jina-v3",
		hf_id="jinaai/jina-embeddings-v3",
		max_seq_len=8192,
		chunk_default=512,
		dim=1024,
		tarea_passage="retrieval.passage",
		tarea_query="retrieval.query",
		trust_remote_code=True,
		notas=(
			"Fuerte: muy fuerte en lenguas romances (espanol top), ventana 8k, "
			"LoRA adapters por tarea (retrieval/classification/separation), "
			"Matryoshka -> permite truncar el vector a 256/512/768/1024 sin reentrenar. "
			"Debil: ~570M params (~2.3GB FP32), encode mas lento que bge-m3 en CPU. "
			"Requiere trust_remote_code=True: ejecuta codigo Python que viene en el "
			"repo del modelo (sigue siendo 100% local, no llama a ninguna API). "
			"HW: GPU con 3GB VRAM ideal; en CPU >=8GB RAM."
		),
	),
}


def listar_ids() -> list[str]:
	return list(EMBEDDERS.keys())


def get_spec(id_corto: str) -> EmbedderSpec:
	if id_corto not in EMBEDDERS:
		raise ValueError(
			f"Embedder desconocido '{id_corto}'. Disponibles: {', '.join(EMBEDDERS)}"
		)
	return EMBEDDERS[id_corto]


class Sizer:
	"""Cuenta tokens reales con el tokenizador del modelo objetivo.

	No carga pesos del modelo, solo el tokenizador (es liviano).
	El conteo incluye el prefijo de passage (si el modelo lo requiere) pero
	NO los tokens especiales (BOS/EOS), de modo que `chunk_max` deja margen
	suficiente para que el embedder los anada al codificar.
	"""

	def __init__(self, embedder_id: str, chunk_tokens: Optional[int] = None):
		from transformers import AutoTokenizer  # import perezoso

		self.spec = get_spec(embedder_id)
		self.tokenizer = AutoTokenizer.from_pretrained(
			self.spec.hf_id,
			cache_dir=str(MODELOS_DIR),
			trust_remote_code=self.spec.trust_remote_code,
		)
		self.chunk_max = chunk_tokens or self.spec.chunk_default
		if self.chunk_max > self.spec.max_seq_len:
			raise ValueError(
				f"chunk_tokens={self.chunk_max} excede max_seq_len={self.spec.max_seq_len} "
				f"de {self.spec.hf_id}"
			)

	def count(self, texto: str) -> int:
		"""Tokens reales que ocupara `texto` como passage (sin especiales)."""
		if not texto:
			return 0
		ids = self.tokenizer.encode(
			self.spec.prefijo_passage + texto, add_special_tokens=False
		)
		return len(ids)

	def fits(self, texto: str) -> bool:
		return self.count(texto) <= self.chunk_max

	def trocear_texto(self, texto: str) -> list[str]:
		"""Trocea un texto que excede chunk_max en sub-textos a nivel de tokens.
		Decodifica los ids de vuelta a texto para mantener compatibilidad.
		Solo se usa como fallback cuando ni siquiera una oracion entra.
		"""
		texto = texto.strip()
		if not texto:
			return []
		ids = self.tokenizer.encode(texto, add_special_tokens=False)
		if len(ids) <= self.chunk_max:
			return [texto]
		piezas: list[str] = []
		for i in range(0, len(ids), self.chunk_max):
			trozo = self.tokenizer.decode(
				ids[i : i + self.chunk_max], skip_special_tokens=True
			).strip()
			if trozo:
				piezas.append(trozo)
		return piezas


def _parchear_jina_lora() -> None:
	"""Parcha el archivo modeling_lora.py cacheado de jina-embeddings-v3 si le
	falta la llamada a self.post_init() al final de XLMRobertaLoRA.__init__,
	requerida desde transformers>=5.x para inicializar all_tied_weights_keys."""
	import pathlib
	cache_base = pathlib.Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "jinaai"
	if not cache_base.exists():
		return
	for lora_file in cache_base.rglob("modeling_lora.py"):
		text = lora_file.read_text(encoding="utf-8")
		# Ya parchado
		if "post_init" in text:
			continue
		# Insertar self.post_init() justo antes del primer @property tras el __init__
		patched = text.replace(
			"        self.main_params_trainable = config.lora_main_params_trainable\n\n    @property",
			"        self.main_params_trainable = config.lora_main_params_trainable\n        self.post_init()\n\n    @property",
		)
		if patched != text:
			lora_file.write_text(patched, encoding="utf-8")
			print(f"[OK] Parche post_init aplicado en: {lora_file}")


def cargar_sentence_transformer(embedder_id: str, device: str = "cpu"):
	"""Carga el modelo como SentenceTransformer (para boundary detection / vectorizacion).
	Import perezoso para que el solo conteo de tokens no requiera torch.
	"""
	import logging
	import sys
	import warnings
	from sentence_transformers import SentenceTransformer

	spec = get_spec(embedder_id)
	kwargs: dict = {"cache_folder": str(MODELOS_DIR), "device": device}
	if spec.trust_remote_code:
		kwargs["trust_remote_code"] = True
		_parchear_jina_lora()

	_PRINT_SUPPRESS = {"flash_attn is not installed"}

	class _FilteredStream:
		def __init__(self, stream):
			self._stream = stream
		def write(self, s):
			if not any(p in s for p in _PRINT_SUPPRESS):
				self._stream.write(s)
		def flush(self):
			self._stream.flush()

	_loggers = [
		logging.getLogger(name)
		for name in ("transformers", "sentence_transformers", "torch")
	]
	_prev_levels = [lg.level for lg in _loggers]
	for lg in _loggers:
		lg.setLevel(logging.ERROR)

	_prev_stdout, _prev_stderr = sys.stdout, sys.stderr
	sys.stdout = _FilteredStream(sys.__stdout__)
	sys.stderr = _FilteredStream(sys.__stderr__)
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", message=".*flash_attn.*")
		try:
			return SentenceTransformer(spec.hf_id, **kwargs)
		except RuntimeError:
			if kwargs.get("device") != "cpu":
				sys.__stdout__.write(f"[ADVERTENCIA] No se pudo cargar en '{kwargs['device']}', reintentando en CPU...\n")
				kwargs["device"] = "cpu"
				return SentenceTransformer(spec.hf_id, **kwargs)
			raise
		finally:
			sys.stdout = _prev_stdout
			sys.stderr = _prev_stderr
			for lg, lvl in zip(_loggers, _prev_levels):
				lg.setLevel(lvl)
