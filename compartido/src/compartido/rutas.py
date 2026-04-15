from pathlib import Path

# Directorio raíz del proyecto (dos niveles arriba de este archivo)
RAIZ = Path(__file__).resolve().parents[3]

DESCARGAS_DIR = RAIZ / "descargas"
ARCHIVO_REGISTRO = DESCARGAS_DIR / "registros.json"