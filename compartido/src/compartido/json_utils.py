import json
from .rutas import ARCHIVO_REGISTRO

def cargar_archivo(ruta):
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    
def guardar_archivo(ruta, datos):
    try:
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"Error al guardar el archivo {ruta}: {e}")



def anadir_nodo(archivo, key, valor):
    registros = cargar_archivo(archivo)
    registros[key] = valor
    guardar_archivo(archivo, registros)

def cargar_nodo(archivo, key):
    registros = cargar_archivo(archivo)
    return registros.get(key, None)

def actualizar_nodo(archivo, key, data):
    registros = cargar_archivo(archivo)
    if key in registros:
        registros[key].update(data)
        guardar_archivo(archivo, registros)

def eliminar_nodo(archivo, key):
    registros = cargar_archivo(archivo)
    if key in registros:
        del registros[key]
        guardar_archivo(archivo, registros)


def anadir_registro(key, valor):
    anadir_nodo(ARCHIVO_REGISTRO, key, valor)
def cargar_registro(key):
    return cargar_nodo(ARCHIVO_REGISTRO, key)
def actualizar_registro(key, data):
    actualizar_nodo(ARCHIVO_REGISTRO, key, data)
def eliminar_registro(key):
    eliminar_nodo(ARCHIVO_REGISTRO, key)