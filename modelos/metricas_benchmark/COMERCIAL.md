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
