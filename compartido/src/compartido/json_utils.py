import json
from .rutas import ARCHIVO_REGISTRO


def cargar_archivo(ruta):
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[INFO] Creando archivo.")
        return {}
    except json.JSONDecodeError:
        print(f"[WARNING] Error en lectura de archivo. Se creará uno nuevo.")
        return {}

def guardar_archivo(ruta, datos):
    try:
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"[ERROR] Error al guardar el archivo {ruta}: {e}")


def _navegar(datos, ruta):
    nodo = datos
    for p in ruta:
        if not isinstance(nodo.get(p), dict):
            raise KeyError(f"La clave '{p}' no existe o no es un dict.")
        nodo = nodo[p]
    return nodo


def cargar_nodo(archivo, key, ruta=()):
    registros = cargar_archivo(archivo)
    contenedor = _navegar(registros, ruta)
    return contenedor.get(key, None)

def anadir_nodo(archivo, key, valor, ruta=()):
    registros = cargar_archivo(archivo)
    contenedor = _navegar(registros, ruta)
    if key in contenedor:
        raise KeyError(f"La clave '{key}' ya existe. Usa reemplazar_nodo para sobrescribir.")
    contenedor[key] = valor
    guardar_archivo(archivo, registros)

def anadir_nodos(archivo, datos, ruta=()):
    registros = cargar_archivo(archivo)
    contenedor = _navegar(registros, ruta)
    conflictos = set(datos) & set(contenedor)
    if conflictos:
        raise KeyError(f"Las claves {conflictos} ya existen. Usa reemplazar_nodo para sobrescribir.")
    contenedor.update(datos)
    guardar_archivo(archivo, registros)

def reemplazar_nodo(archivo, key, valor, ruta=()):
    registros = cargar_archivo(archivo)
    contenedor = _navegar(registros, ruta)
    if key not in contenedor:
        raise KeyError(f"La clave '{key}' no existe. Usa anadir_nodo para crear.")
    contenedor[key] = valor
    guardar_archivo(archivo, registros)

def eliminar_nodo(archivo, key, ruta=()):
    registros = cargar_archivo(archivo)
    contenedor = _navegar(registros, ruta)
    if key not in contenedor:
        raise KeyError(f"La clave '{key}' no existe.")
    del contenedor[key]
    guardar_archivo(archivo, registros)


def cargar_registro(key, ruta=()):
    return cargar_nodo(ARCHIVO_REGISTRO, key, ruta)
def anadir_registro(key, valor, ruta=()):
    anadir_nodo(ARCHIVO_REGISTRO, key, valor, ruta)
def anadir_registros(datos, ruta=()):
    anadir_nodos(ARCHIVO_REGISTRO, datos, ruta)
def reemplazar_registro(key, valor, ruta=()):
    reemplazar_nodo(ARCHIVO_REGISTRO, key, valor, ruta)
def eliminar_registro(key, ruta=()):
    eliminar_nodo(ARCHIVO_REGISTRO, key, ruta)