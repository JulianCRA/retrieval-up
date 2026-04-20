import hashlib

def obtener_identificador(path):
    path_str = str(path)
    return hashlib.sha256(path_str.encode()).hexdigest()[:16]