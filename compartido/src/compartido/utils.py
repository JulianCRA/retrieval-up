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
