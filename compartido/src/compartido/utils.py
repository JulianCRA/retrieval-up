import hashlib

def obtener_identificador(path):
    path_str = str(path)
    return hashlib.sha256(path_str.encode()).hexdigest()[:16]

def obtener_dispositivo():
    """Devuelve el dispositivo torch disponible (cuda o cpu) y emite una advertencia
    si hay una GPU NVIDIA presente pero PyTorch fue instalado sin soporte CUDA."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.version.cuda is None:
        import shutil
        if shutil.which("nvidia-smi") is not None:
            import warnings
            warnings.warn(
                "\n"
                "[ADVERTENCIA] Se detectó una GPU NVIDIA pero PyTorch está instalado sin soporte CUDA.\n"
                "Para aprovechar la GPU, reinstala PyTorch con soporte CUDA:\n"
                "\n"
                "  CUDA 12.8 (recomendado para RTX 40xx/50xx):\n"
                "    pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall\n"
                "\n"
                "  CUDA 12.1 (para GPUs más antiguas):\n"
                "    pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall\n"
                "\n"
                "Consulta https://pytorch.org/get-started/locally/ para más opciones.",
                stacklevel=2
            )
    return torch.device("cpu")