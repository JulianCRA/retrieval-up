import numpy as np


# deteccion basada en energ´ıa mediante el calculo de la amplitud cuadratica media (RMS) por ventana temporal
def vad_energia(audio, samplerate, umbral_db=-40.0, duracion_minima=0.5):
    print(f"[INFO] VAD por energía (CPU)")
    ventana_duracion = 0.02
    ventana_muestras = int(ventana_duracion * samplerate)
    umbral_lineal = 10 ** (umbral_db / 20.0)

    segmentos = []
    en_voz = False
    inicio_voz = 0

    for i in range(0, len(audio), ventana_muestras):
        ventana = audio[i:i + ventana_muestras]
        rms = np.sqrt(np.mean(ventana**2))

        if rms >= umbral_lineal:
            if not en_voz:
                en_voz = True
                inicio_voz = i / samplerate
        else:
            if en_voz:
                en_voz = False
                fin_voz = i / samplerate
                if (fin_voz - inicio_voz) >= duracion_minima:
                    segmentos.append((round(inicio_voz, 3), round(fin_voz, 3)))

    if en_voz:
        fin_voz = len(audio) / samplerate
        if (fin_voz - inicio_voz) >= duracion_minima:
            segmentos.append((round(inicio_voz, 3), round(fin_voz, 3)))

    print(f"[INFO] {len(segmentos)} segmentos detectados")

    return segmentos
