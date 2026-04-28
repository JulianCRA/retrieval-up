def aplicar_padding(segmentos, padding):
    segmentos_con_padding = []
    for inicio, fin in segmentos:
        inicio_padded = max(0, inicio - padding)
        fin_padded = fin + padding
        segmentos_con_padding.append((round(inicio_padded, 3), round(fin_padded, 3)))
    return segmentos_con_padding

