# Benchmark: resultados en test (235 fotos, 152 cajas comerciales)

Acierto comercial y error crítico: la predicción más segura que cubre cada fresa (IoU ≥ 0,5, confianza ≥ 0,25).

| # | Modelo | Tamaño | Acierto comercial | Error crítico | mAP50 | mAP50-95 | Min |
|---|---|---|---|---|---|---|---|
| 1 | yolo11n-hsv-1024 | nano | 78.9 % | 6.6 % | 0.866 | 0.740 | 43.8 |
| 2 | yolo26n | nano | 77.0 % | 7.9 % | 0.875 | 0.744 | 19.8 |
| 3 | yolov9e | grande | 77.0 % | 10.5 % | 0.891 | 0.753 | 109.2 |
| 4 | yolo11n-1024 | nano | 75.7 % | 10.5 % | 0.890 | 0.761 | 39.6 |
| 5 | yolov10n | nano | 73.0 % | 11.2 % | 0.861 | 0.735 | 19.8 |
| 6 | yolo12x | grande | 72.4 % | 12.5 % | 0.892 | 0.764 | 147.0 |
| 7 | yolo12n | nano | 72.4 % | 13.8 % | 0.848 | 0.723 | 27.5 |
| 8 | yolo11n | nano | 72.4 % | 14.5 % | 0.869 | 0.728 | 10.5 |
| 9 | yolo11n-hsv | nano | 71.7 % | 13.8 % | 0.853 | 0.719 | 14.2 |
| 10 | yolov9t | nano | 69.1 % | 15.8 % | 0.877 | 0.748 | 27.4 |
| 11 | yolov8n | nano | 63.8 % | 19.1 % | 0.870 | 0.739 | 14.5 |

Sin terminar al 05/10 12:00 UTC: yolov8x, yolov10x, yolo11x, yolo26x.

## ¿Mejora de verdad? Pruebas pareadas sobre las mismas 152 fresas comerciales del test
McNemar exacto (¿la diferencia puede ser azar?) e intervalo de confianza del 95 % por bootstrap. En «error crítico»,
una diferencia positiva es una reducción del error.

| Comparación | Acierto: diferencia (IC 95 %), p | Error crítico: reducción (IC 95 %), p |
|---|---|---|
| yolo11n → + tono | −0,7 (−7,2 a +5,9), p = 1,00 | −0,7 (−5,9 a +4,6), p = 1,00 |
| yolo11n → + 1024 px | +3,3 (−5,3 a +11,8), p = 0,54 | +3,9 (−2,0 a +9,9), p = 0,29 |
| yolo11n 1024 → + tono | +3,3 (−3,3 a +9,9), p = 0,44 | +3,9 (0,0 a +7,9), p = 0,11 |
| **yolo11n → + tono + 1024 px** | +6,6 (−0,7 a +14,5), p = 0,13 | **+7,9 (+2,0 a +13,8), p = 0,017** |
| yolo26n → yolo11n + tono + 1024 px | +2,0 (−5,3 a +9,9), p = 0,74 | +1,3 (−3,3 a +6,6), p = 0,79 |

Lo único significativo (p < 0,05): tono + 1024 px reduce el error crítico de yolo11n. Corresponde a 17 fresas que pasan a
estar bien, contra 5 que pasan a estar mal. Ningún otro cambio, ni la diferencia con yolo26n, es distinguible del azar.
