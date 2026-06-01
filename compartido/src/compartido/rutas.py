from pathlib import Path

# Directorio raíz del proyecto (dos niveles arriba de este archivo)
RAIZ = Path(__file__).resolve().parents[3]

DESCARGAS_DIR = RAIZ / "descargas"
DESCARGAS_DIR.mkdir(parents=True, exist_ok=True)

INDICE_DIR = RAIZ / "indice"

MODELOS_DIR = RAIZ / "modelos"
MODELOS_EMBEDDINGS_DIR = MODELOS_DIR / "embeddings"
MODELOS_VOSK_DIR = MODELOS_DIR / "vosk"
MODELOS_WHISPER_DIR = MODELOS_DIR / "whisper"
MODELOS_TORCH_HUB_DIR = MODELOS_DIR / "torch_hub"
MODELOS_PUNCTUATE_ALL_DIR = MODELOS_DIR / "punctuate-all"
MODELOS_COHERE_DIR = MODELOS_DIR / "cohere"

RESULTADOS_DIR = RAIZ / "resultados"

ARCHIVO_REGISTRO = DESCARGAS_DIR / "registros.json"