def redondear_segmentos(segmentos, ndigits=3):
    return [(round(inicio, ndigits), round(fin, ndigits)) for inicio, fin in segmentos]

def aplicar_padding(segmentos, padding):
    segmentos_con_padding = []
    for inicio, fin in segmentos:
        inicio_padded = max(0, inicio - padding)
        fin_padded = fin + padding
        segmentos_con_padding.append((round(inicio_padded, 3), round(fin_padded, 3)))
    return segmentos_con_padding

def fusion_por_gap(segmentos, join_gap, duracion_target, duracion_maxima):
    if not segmentos:
        return []

    fusionados = []
    current_inicio, current_fin = segmentos[0]

    for inicio, fin in segmentos[1:]:
        gap = inicio - current_fin
        duracion_fusionada = fin - current_inicio

        if gap <= join_gap and duracion_fusionada <= duracion_target:
            current_fin = fin
            continue

        if gap <= 0.15 and duracion_fusionada <= duracion_maxima:
            current_fin = fin
            continue

        fusionados.append((round(current_inicio, 3), round(current_fin, 3)))
        current_inicio, current_fin = inicio, fin

    fusionados.append((round(current_inicio, 3), round(current_fin, 3)))
    return fusionados
        
def absorber_chunks_pequenos(segmentos, duracion_minima, duracion_maxima):
    if not segmentos:
        return []
    
    ajustados = []
    for inicio, fin in segmentos:
        ultimo_inicio, ultimo_fin = ajustados[-1] if ajustados else (None, None)
        ultima_duracion = (ultimo_fin - ultimo_inicio) if ultimo_inicio is not None else None

        if ultima_duracion < duracion_minima and (fin - ultimo_inicio) <= duracion_maxima:
            ajustados[-1] = (ultimo_inicio, fin)
        else:
            ajustados.append((inicio, fin))

    return ajustados

def limitar_duracion(segmentos, duracion_target, duracion_maxima, overlap):
    finales = []

    for inicio, fin in segmentos:
        duracion = fin - inicio
        if duracion <= duracion_maxima:
            finales.append((inicio, fin))
            continue

        cursor = inicio
        while cursor < fin:
            corte = min(cursor + duracion_maxima, fin)
            finales.append((cursor, corte))
            if corte >= fin:
                break
            cursor = max(inicio, corte - overlap)

    return finales

def obtener_fragmentos_asr(segmentos, perfil):
    if not segmentos:
        return []
    
    spans = aplicar_padding(segmentos, perfil["padding"])
    spans = fusion_por_gap(spans, perfil["join_gap"], perfil["duracion_target"], perfil["duracion_maxima"])
    spans = absorber_chunks_pequenos(spans, perfil["duracion_minima"], perfil["duracion_maxima"])
    spans = limitar_duracion(spans, perfil["duracion_target"], perfil["duracion_maxima"], perfil["overlap"])

    print(f"[INFO] {len(spans)} fragmentos ASR luego de aplicar perfil")

    return redondear_segmentos(spans)