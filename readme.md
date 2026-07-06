# Recuperación de Documentos de Audio y Video

Pipeline modular para transformar contenido de audio y video en fragmentos recuperables mediante búsqueda semántica.

**Repositorio:** https://github.com/JulianCRA/retrieval-up

---

## Descripción

El sistema procesa fuentes de audio y video (locales o remotas) a través de una cadena de módulos independientes, produciendo chunks de texto indexables para tareas de recuperación de información.

```
descargar → procesar → transcribir → corregir → fragmentar → vectorizar → indexar → recuperar
```

Cada módulo es un paquete Python instalable de forma independiente. Todos comparten el directorio `descargas/` como almacenamiento de estado, identificando cada contenido mediante un hash único generado en la descarga.

---

## Módulos

### 1. `descargador`
Descarga audio y video desde URLs remotas (YouTube y cualquier fuente compatible con yt-dlp) o carga archivos locales. Convierte el resultado a WAV mono 16 kHz.

```bash
descargar -s <URL o ruta>
```

- Acepta URLs, archivos de media, archivos de texto con listas de URLs, o directorios.
- Genera un hash único por contenido y guarda los metadatos en `descargas/<hash>/info.json`.

**Dependencias:** `yt-dlp`, `ffmpeg` (externo)

---

### 2. `procesador`
Reduce ruido de fondo y aplica VAD (Voice Activity Detection) para segmentar el audio en intervalos de voz.

```bash
procesar --hash <hash> [-m energia|silero|webrtc]
```

| Método | Velocidad | Precisión | Requisitos |
|--------|-----------|-----------|------------|
| `energia` | ⚡⚡⚡ Muy rápido | Básica | Solo CPU, ~50 MB RAM |
| `webrtc` | ⚡⚡ Rápido | Media | Solo CPU, ~100 MB RAM |
| `silero` | ⚡ No tan rápido, pero igual rápido | Alta | CPU recomendada, ~500 MB RAM |

Guarda los segmentos detectados en `segmentos.json`.

**Dependencias:** `soundfile`, `noisereduce`, `silero-vad`, `webrtcvad-wheels`, `torch`

---

### 3. `transcriptor`
Transcribe los segmentos de audio a texto mediante modelos ASR.

```bash
transcribir --hash <hash> [-m vosk|whisper:small|whisper:base|whisper:turbo|cohere]
transcribir --hash <hash> -i vosk|whisper|cohere   # info detallada del modelo
```

| Modelo | Motor |  Calidad | Anotacion |
|--------|-------|----------|-----------|
| `vosk` | CPU | Media | requiere corrector|
| `whisper:base` | GPU y CPU | peor |  |
| `whisper:small` | GPU y CPU | moderada |  |
| `whisper:turbo` | GPU y CPU | Alta | CPU es posible pero no es viable |
| `cohere` | GPU y CPU | Excelente | CPU es posible pero no es viable |

Guarda el resultado en `transcripciones.json`.

**Dependencias:** `vosk`, `faster-whisper` (opcional), `transformers` + `torch` (opcional para Cohere)

---

### 4. `corrector`
Restaura puntuación y mayúsculas en transcripciones producidas por Vosk (que no incluye puntuación).

```bash
corr --hash <hash> [--m silero|p-all]
```

- **`silero`** (default): modelo neuronal liviano, rápido, recomendado.
- **`p-all`**: pipeline `punctuate-all` + spaCy (`es_core_news_lg`).
- Si el modelo ASR ya incluye puntuación (Whisper, Cohere), el módulo lo detecta y copia la transcripción sin modificaciones.

Guarda el resultado en `correcciones.json`.

**Dependencias:** `torch`, `transformers`, `spacy`, `es_core_news_lg`

---

### 5. `fragmentador`
Divide el texto corregido en chunks para su posterior vectorización e indexación.

```bash
frag --hash <hash> --embedder <id> [--estrategia tamano_fijo|semantico] [opciones]
```

`--embedder` es obligatorio y debe coincidir con el que se usará en la vectorización. Los IDs disponibles son los mismos que en el vectorizador (ver tabla más abajo).

**Estrategia `tamano_fijo`** (default):

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--chunk-tokens N` | según modelo | Tamaño máximo del chunk en tokens reales del modelo |
| `--overlap PCT` | 20 | Solapamiento entre chunks consecutivos (%) |

**Estrategia `semantico`**:

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--umbral` | 0.5 | Similitud coseno mínima para corte temático (0–1) |
| `--min-tokens` | 64 | Tamaño mínimo de chunk antes de fusionar |
| `--chunk-tokens N` | según modelo | Límite de seguridad por chunk |
| `--boundary-embedder` | mismo que `--embedder` | Modelo alternativo solo para detectar bordes |

Guarda los fragmentos en `fragmentos.json`.

**Dependencias:** `sentence-transformers`, `numpy`

---

### 6. `vectorizador`
Calcula embeddings densos por fragmento y los persiste para su indexación.

```bash
vect --hash <hash> [--embedder <id>] [--batch-size N] [--forzar-cpu]
```

Si no se indica `--embedder`, se lee `embedder_objetivo` desde `fragmentos.json`.

**Embedders disponibles:**

| ID | Modelo HF | Dim | Ventana | Notas |
|----|-----------|-----|---------|-------|
| `granite-107m` | `ibm-granite/granite-embedding-107m-multilingual` | 384 | 512 | Muy liviano, ~210 MB, ideal CPU |
| `e5-large-instruct` | `intfloat/multilingual-e5-large-instruct` | 1024 | 512 | Buena calidad ES/EN, requiere instrucción en query |
| `bge-m3` | `BAAI/bge-m3` | 1024 | 8192 | Excelente multilingüe, ventana 8k |
| `jina-v3` | `jinaai/jina-embeddings-v3` | 1024 | 8192 | Muy fuerte en español, Matryoshka |
| `qwen3-0.6b` | `Qwen/Qwen3-Embedding-0.6B` | 1024 | 32768 | Top MTEB multilingüe, ventana 32k, requiere GPU |

Guarda los embeddings en `vectores.json`.

**Dependencias:** `sentence-transformers`, `transformers`, `numpy`

---

### 7. `indexador`
Construye un índice híbrido (denso + BM25) a partir de los embeddings y fragmentos.

```bash
indexar --hash <hash> [--embedder <id>] [--backend lance|qdrant|milvus] [--tag CLAVE=VALOR]
```

- El backend por defecto es `lance` (LanceDB local, sin servidor).
- `--tag` permite adjuntar metadata a todos los chunks del lote (repetible).
- Persiste el corpus en `indice.db` (SQLite) y los vectores en `indice/` (LanceDB).

**Dependencias:** `lancedb`, `pyarrow`, `tantivy`, `numpy`, `spacy`, `es_core_news_lg`

---

### 8. `recuperador`
Búsqueda híbrida (semántica + BM25) sobre el índice.

```bash
recuperar --query <texto> --embedder <id> [--modo rrf|wrrf|denso|bm25] [--reranker mmarco|bge|jina] [--top-k N]
```

| Modo | Descripción |
|------|-------------|
| `denso` (default) | Solo similitud coseno |
| `bm25` | Solo keywords |
| `rrf` | Fusión de ranks (Reciprocal Rank Fusion) |
| `wrrf` | RRF ponderado (`--peso-semantica`, default 0.7) |

Los rerankers (`mmarco`, `bge`, `jina`) reordenan los resultados con un cross-encoder.

Guarda cada consulta en `resultados.db` para trazabilidad.

**Dependencias:** `lancedb`, `tantivy`, `sentence-transformers`, `numpy`, `spacy`

---

### 9. `insights`
Módulo de analítica docente: vista agregada del corpus indexado y del comportamiento de búsqueda. Opera exclusivamente en lectura sobre `indice.db` y `resultados.db`.

```bash
insights corpus                          # resumen del corpus indexado
insights encontrados [--top-n N]         # videos más devueltos por el sistema
insights seleccionados [--top-n N]       # videos más seleccionados por usuarios
insights comparar                        # encontrados vs. seleccionados
insights grupos --embedder <id>          # agrupación HDBSCAN + etiquetas KeyBERT
insights todo                            # todas las salidas anteriores
```

Exporta CSV y JSON a `resultados/insights/`.

**Dependencias:** `numpy`, `scikit-learn`, `hdbscan`, `umap-learn`, `keybert`, `sentence-transformers`

---

### `gui`
Interfaz web local para buscar, explorar resultados y consultar analíticas.

```bash
gui
```

**Dependencias:** `flask`, `recuperador`, `insights`

---

## Instalación

Requiere Python 3.11–3.12 y `ffmpeg` instalado en el sistema.

```bash
# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS

# Instalar módulos (desde la raíz del repositorio)
pip install -e compartido/
pip install -e "01. descargador/"
pip install -e "02. procesador de audio/"        # CPU
pip install -e "02. procesador de audio/[gpu]" --index-url https://download.pytorch.org/whl/cu128  # GPU
pip install -e "03. transcriptor/"               # solo Vosk
pip install -e "03. transcriptor/[whisper]"      # + Whisper
pip install -e "03. transcriptor/[cohere-transcribe]"  # + Cohere
pip install -e "04. corrector de texto/"
pip install -e "05. fragmentador/"
pip install -e "06. vectorizador/"
pip install -e "07. indexador/"
pip install -e "08. recuperador/"
pip install -e "09. insights/"
pip install -e gui/
```

---

## Uso típico

```bash
# 1. Descargar contenido
descargar -s "https://www.youtube.com/watch?v=..."

# 2. Procesar audio (segmentar silencios)
procesar --hash <hash> -m silero

# 3. Transcribir
transcribir --hash <hash> -m whisper:turbo

# 4. Corregir puntuación
corr --hash <hash>

# 5. Fragmentar (el --embedder determina el tokenizador y el tamaño de chunk)
frag --hash <hash> --embedder bge-m3 --estrategia semantico --umbral 0.4

# 6. Vectorizar
vect --hash <hash>

# 7. Indexar
indexar --hash <hash>

# 8. Buscar
recuperar --query "¿Cómo se aplica el teorema de Bayes?" --embedder bge-m3 --modo rrf
```

El `<hash>` se obtiene de la salida del descargador o consultando `descargas/registros.json`.

---

## Estructura de datos

```
descargas/
├── registros.json              # índice global de todos los contenidos descargados
└── <hash>/
    ├── info.json               # metadatos y rutas del contenido
    ├── segmentos.json          # intervalos de voz detectados por el procesador
    ├── transcripciones.json    # texto por segmento del transcriptor
    ├── correcciones.json       # texto con puntuación restaurada
    ├── fragmentos.json         # chunks listos para vectorizar
    └── vectores.json           # embeddings densos por chunk

indice/
├── indice.db                   # SQLite: corpus, chunks, tags, duracion
└── <embedder>/                 # tablas LanceDB (una por embedder)

resultados/
├── resultados.db               # SQLite: historial de búsquedas y selecciones
└── insights/                   # exportaciones CSV/JSON del módulo insights
```

---

## Paquete compartido

`compartido` provee utilidades comunes a todos los módulos:

- `rutas.py` — rutas absolutas del proyecto (`DESCARGAS_DIR`, `ARCHIVO_REGISTRO`)
- `json_utils.py` — carga y guardado seguro de JSON
- `utils.py` — decorador `@cronometrar`, `obtener_identificador`, `crear_perfil_hardware`
