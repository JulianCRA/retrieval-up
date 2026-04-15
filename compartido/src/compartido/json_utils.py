import json
from pathlib import Path
from .rutas import ARCHIVO_REGISTRO

def cargar_registros():
    if ARCHIVO_REGISTRO.exists():
        with open(ARCHIVO_REGISTRO, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def obtener_status(hash):
    registros = cargar_registros()
    if hash in registros:
        return registros[hash]["status"]
    return None

def actualizar_status(hash, status):
    registros = cargar_registros()
    registros[hash] = {"status": status}
    with open(ARCHIVO_REGISTRO, "w", encoding="utf-8") as f:
        json.dump(registros, f, indent=4)

def obtener_registro(hash):
    registros = cargar_registros()
    return registros.get(hash, None)

def eliminar_registro(hash):
    registros = cargar_registros()
    if hash in registros:
        del registros[hash]
        with open(ARCHIVO_REGISTRO, "w", encoding="utf-8") as f:
            json.dump(registros, f, indent=4)

def actualizar_registro(hash, data):
    registros = cargar_registros()
    if hash in registros:
        registros[hash] = data
        with open(ARCHIVO_REGISTRO, "w", encoding="utf-8") as f:
            json.dump(registros, f, indent=4)

def anadir_info(hash, key, value):
    registros = cargar_registros()
    if hash in registros:
        registros[hash][key] = value
        with open(ARCHIVO_REGISTRO, "w", encoding="utf-8") as f:
            json.dump(registros, f, indent=4)