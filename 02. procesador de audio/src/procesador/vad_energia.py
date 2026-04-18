import numpy as np


# deteccion basada en energ´ıa mediante el calculo de la amplitud cuadratica media (RMS) por ventana temporal
def vad_energia(audio, samplerate, umbral_db=-40.0, duracion_minima=0.5):
    print(f"[INFO] Detectando silencios en el audio...")
    ventana_duracion = 0.02
    ventana_muestras = int(ventana_duracion * samplerate)
    umbral_lineal = 10 ** (umbral_db / 20.0)

    silencio = []
    en_silencio = False
    inicio_silencio = 0

    for i in range(0, len(audio), ventana_muestras):
        ventana = audio[i:i + ventana_muestras]
        rms = np.sqrt(np.mean(ventana**2))

        if rms < umbral_lineal:
            if not en_silencio:
                en_silencio = True
                inicio_silencio = i / samplerate
        else:
            if en_silencio:
                en_silencio = False
                duracion_silencio = (i / samplerate) - inicio_silencio
                if duracion_silencio >= duracion_minima:
                    silencio.append((inicio_silencio, duracion_silencio))

    if en_silencio:
        duracion_silencio = (len(audio) / samplerate) - inicio_silencio
        if duracion_silencio >= duracion_minima:
            silencio.append((inicio_silencio, duracion_silencio))

    for inicio, duracion in silencio:
        print(f"Silencio detectado: Inicio = {inicio:.2f} s, Fin = {inicio + duracion:.2f} s")

    return silencio
