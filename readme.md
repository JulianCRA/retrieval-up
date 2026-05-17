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
frag --hash <hash> [--estrategia tamano_fijo|semantico] [opciones]
```

**Estrategia `tamano_fijo`** (default):

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--max-tokens` | 512 | Tamaño máximo del chunk en tokens estimados |
| `--overlap PCT` | 20 | Solapamiento entre chunks consecutivos (%) |

**Estrategia `semantico`**:

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--umbral` | 0.5 | Similitud coseno mínima para corte temático (0–1) |
| `--min-tokens` | 64 | Tamaño mínimo de chunk antes de fusionar |
| `--max-tokens` | 512 | Límite de seguridad por chunk |

Usa el modelo `paraphrase-multilingual-MiniLM-L12-v2` para embeddings. Guarda los fragmentos en `fragmentos.json`.

**Dependencias:** `sentence-transformers`, `numpy`

---

### 6–8. `vectorizador`, `indexador`, `recuperador`
En desarrollo.

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

# 5. Fragmentar
frag --hash <hash> --estrategia semantico --umbral 0.4
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
    └── fragmentos.json         # chunks listos para vectorizar
```

---

## Paquete compartido

`compartido` provee utilidades comunes a todos los módulos:

- `rutas.py` — rutas absolutas del proyecto (`DESCARGAS_DIR`, `ARCHIVO_REGISTRO`)
- `json_utils.py` — carga y guardado seguro de JSON
- `utils.py` — decorador `@cronometrar`, `obtener_identificador`, `crear_perfil_hardware`
