import hashlib

import time
import functools

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

def obtener_perfil_hardware():
    # NVIDIA (Windows/Linux)
    inicio = time.perf_counter()
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
    duracion = time.perf_counter() - inicio
    print(f"[INFO] Perfil de hardware obtenido en {duracion:.2f}s: {perfil}")
    return perfil

def cronometrar(func=None, *, etiqueta=None):
    """Decorator que imprime el tiempo de ejecución de una función.
    
    Uso directo:       @cronometrar
    Con etiqueta:      @cronometrar(etiqueta="Descarga")
    """
    if func is None:
        return lambda f: cronometrar(f, etiqueta=etiqueta)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nombre = etiqueta or func.__name__
        inicio = time.perf_counter()
        resultado = func(*args, **kwargs)
        wrapper.elapsed = time.perf_counter() - inicio
        print(f"[TIEMPO] {nombre}: {wrapper.elapsed:.2f}s")
        return resultado

    wrapper.elapsed = None
    return wrapper
