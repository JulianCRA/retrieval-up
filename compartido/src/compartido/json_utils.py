import json
from .rutas import ARCHIVO_REGISTRO


def cargar_archivo(ruta):
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"[WARNING] Error en lectura de archivo. Se creará uno nuevo.")
        return {}

def guardar_archivo(ruta, datos):
    try:
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"[ERROR] Error al guardar el archivo {ruta}: {e}")


def cargar_nodo(archivo, key):
    registros = cargar_archivo(archivo)
    return registros.get(key, None)

def anadir_nodo(archivo, key, valor):
    registros = cargar_archivo(archivo)
    if key in registros:
        raise KeyError(f"La clave '{key}' ya existe. Usa reemplazar_nodo para sobrescribir.")
    registros[key] = valor
    guardar_archivo(archivo, registros)

def anadir_nodos(archivo, datos):
    registros = cargar_archivo(archivo)
    conflictos = set(datos) & set(registros)
    if conflictos:
        raise KeyError(f"Las claves {conflictos} ya existen. Usa reemplazar_nodo para sobrescribir.")
    registros.update(datos)
    guardar_archivo(archivo, registros)

def reemplazar_nodo(archivo, key, valor):
    registros = cargar_archivo(archivo)
    if key not in registros:
        raise KeyError(f"La clave '{key}' no existe. Usa anadir_nodo para crear.")
    registros[key] = valor
    guardar_archivo(archivo, registros)

def eliminar_nodo(archivo, key):
    registros = cargar_archivo(archivo)
    if key not in registros:
        raise KeyError(f"La clave '{key}' no existe.")
    del registros[key]
    guardar_archivo(archivo, registros)


def cargar_registro(key):
    return cargar_nodo(ARCHIVO_REGISTRO, key)
def anadir_registro(key, valor):
    anadir_nodo(ARCHIVO_REGISTRO, key, valor)
def anadir_registros(datos):
    anadir_nodos(ARCHIVO_REGISTRO, datos)
def reemplazar_registro(key, valor):
    reemplazar_nodo(ARCHIVO_REGISTRO, key, valor)
def eliminar_registro(key):
    eliminar_nodo(ARCHIVO_REGISTRO, key)