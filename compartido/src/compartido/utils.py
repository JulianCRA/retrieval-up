import hashlib

import time
import functools
import contextvars
from contextlib import contextmanager

import psutil
import shutil
import subprocess

def obtener_identificador(semilla, length=16):
    semilla_str = str(semilla)
    return hashlib.sha256(semilla_str.encode()).hexdigest()[:length]


def detectar_gpu_nvidia():
    if shutil.which("nvidia-smi") is None:
        return False, 0
        
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        # Tomar el primer valor si hay múltiples GPUs
        vram_mb = int(result.stdout.strip().split('\n')[0])
        return True, round(vram_mb / 1024)
    except Exception:
        return False, 0

import platform

def detectar_apple_silicon():
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # Apple unifica la memoria RAM y VRAM. se toma un estimado de la RAM total limitando al 70%.
        ram_total_gb = round(psutil.virtual_memory().total / (1024**3))
        vram_estimada_gb = round(ram_total_gb * 0.7)
        return True, vram_estimada_gb
    return False, 0

def crear_perfil_hardware(forzado=None):
    
    # NVIDIA (Windows/Linux)
    has_nvidia, vram_gb = detectar_gpu_nvidia()
    if has_nvidia:
        device = "cuda"
    else:
        # Apple Silicon (Mac)
        has_mac, vram_gb = detectar_apple_silicon()
        if has_mac:
            device = "mps"
        else:
            # 3. Importar torch por si hay AMD ROCm u otro backend instalado
            try:
                import torch
                if hasattr(torch, "is_xpu_available") and torch.is_xpu_available():
                    device = "xpu" # Intel
                    vram_gb = 4 # Default fallback
                elif torch.cuda.is_available(): 
                    # AMD ROCm en Linux usa el mismo flag 'is_available' que CUDA
                    device = "cuda"
                    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3))
                else:
                    device = "cpu"
                    vram_gb = 0
            except ImportError:
                device = "cpu"
                vram_gb = 0

    perfil = {
        "device": device,
        "vram_gb": vram_gb,
        "ram_gb": round(psutil.virtual_memory().total / (1024**3)),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
    }

    if forzado is not None:
        perfil = forzar_perfil(perfil, forzado)

    return perfil

def forzar_perfil(perfil, forzado):
        
    if isinstance(forzado, dict):
        for key, value in forzado.items():
            if key in perfil:
                # Si es numérico (vram_gb, ram_gb, cores), no dejar que exceda el límite real
                if isinstance(perfil[key], (int, float)) and isinstance(value, (int, float)) and value > perfil[key]:
                    perfil[key] = min(perfil[key], value)
                    print(f"[ADVERTENCIA] Forzado '{key}' a {value}, pero se detectó un máximo de {perfil[key]}. Usando {perfil[key]}.")

                # Si forzan un dispositivo acelerado pero el perfil real es CPU, ignorar
                elif key == "device":
                    if value in ["cuda", "mps", "xpu"] and perfil["device"] == "cpu":
                        print(f"[ADVERTENCIA] Imposible forzar '{value}'. No se detectó acelerador. Mantenido en 'cpu'.")
                    else:
                        perfil[key] = value

                # Si hay alguna otra clave en el futuro
                else:
                    perfil[key] = value
            else:
                raise SystemExit(f"[ERROR] Clave de perfil desconocida para forzar: '{key}'")        
        if perfil["device"] == "cpu":
            perfil["vram_gb"] = 0
    else:
        raise SystemExit(f"[ERROR] El perfil forzado debe ser un diccionario con claves como 'device', 'vram_gb', 'ram_gb', etc.")
    
    return perfil

def cronometrar(func=None, *, etiqueta=None, imprimir=True):
    """Decorator que mide el tiempo de ejecucion de una funcion.

    - Imprime `[TIEMPO] <etiqueta>: X.XXs` por defecto.
    - Si hay un `Cronometro` activo en el contexto (ver `cronometro_activo`),
      acumula el tiempo bajo `etiqueta` (suma sobre invocaciones).
    - Conserva `wrapper.elapsed` con el tiempo de la ultima llamada
      (compatibilidad hacia atras).

    Uso:
        @cronometrar
        @cronometrar(etiqueta="Carga modelo")
        @cronometrar(etiqueta="...", imprimir=False)
    """
    if func is None:
        return lambda f: cronometrar(f, etiqueta=etiqueta, imprimir=imprimir)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nombre = etiqueta or func.__name__
        inicio = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - inicio
            wrapper.elapsed = elapsed
            crono = _cronometro_actual()
            if crono is not None:
                crono.registrar(nombre, elapsed)
            if imprimir:
                print(f"[TIEMPO] {nombre}: {elapsed:.2f}s")

    wrapper.elapsed = None
    return wrapper


# ── Esquema uniforme de tiempos ──────────────────────────────────────────────
#
# Cada modulo envuelve su trabajo por hash con `with cronometro_activo() as
# crono:` y al final escribe (o imprime) `crono.resumen()`. El resultado es
# un dict plano `{"<etiqueta>": segundos, ..., "_total": segundos}` que se
# embebe bajo la clave `"tiempos"` en el JSON de salida correspondiente.

_PILA_CRONOMETROS: contextvars.ContextVar = contextvars.ContextVar(
    "_pila_cronometros", default=()
)


class Cronometro:
    """Acumulador de tiempos por etiqueta.

    El `_total` se mide desde la creacion del cronometro hasta `resumen()`.
    Si una etiqueta se registra varias veces, los segundos se suman.
    """

    def __init__(self):
        self._t0 = time.perf_counter()
        self._registros: dict[str, float] = {}

    def registrar(self, etiqueta: str, segundos: float) -> None:
        self._registros[etiqueta] = self._registros.get(etiqueta, 0.0) + segundos

    @contextmanager
    def medir(self, etiqueta: str, imprimir: bool = True):
        inicio = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - inicio
            self.registrar(etiqueta, elapsed)
            if imprimir:
                print(f"[TIEMPO] {etiqueta}: {elapsed:.2f}s")

    def resumen(self, decimales: int = 3) -> dict:
        salida = {k: round(v, decimales) for k, v in self._registros.items()}
        salida["_total"] = round(time.perf_counter() - self._t0, decimales)
        return salida


def _cronometro_actual() -> Cronometro | None:
    pila = _PILA_CRONOMETROS.get()
    return pila[-1] if pila else None


@contextmanager
def cronometro_activo(cronometro: Cronometro | None = None):
    """Activa un `Cronometro` durante el bloque. `@cronometrar` y `medir`
    registran sus tiempos en el cronometro mas reciente de la pila."""
    crono = cronometro or Cronometro()
    pila = _PILA_CRONOMETROS.get()
    token = _PILA_CRONOMETROS.set(pila + (crono,))
    try:
        yield crono
    finally:
        _PILA_CRONOMETROS.reset(token)


@contextmanager
def medir(etiqueta: str, imprimir: bool = True):
    """Context manager para medir un bloque de codigo y registrarlo en el
    cronometro activo (si lo hay). Equivalente a un `@cronometrar` puntual."""
    inicio = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - inicio
        crono = _cronometro_actual()
        if crono is not None:
            crono.registrar(etiqueta, elapsed)
        if imprimir:
            print(f"[TIEMPO] {etiqueta}: {elapsed:.2f}s")

