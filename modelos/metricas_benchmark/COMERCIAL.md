# Benchmark: acierto en las clases comerciales (test, 235 fotos, 152 cajas comerciales)

Se toma la predicción más segura que cubre cada fresa (IoU ≥ 0,5, confianza ≥ 0,25), como haría el equipo en el campo.

| # | Modelo | Acierto comercial | Error crítico (→ early-pink/overripe) | mAP50-95 | Min |
|---|---|---|---|---|---|
| 1 | yolo11n-hsv-1024 | 78.9 % | 6.6 % | 0.740 | 43.8 |
| 2 | yolo26n | 77.0 % | 7.9 % | 0.744 | 19.8 |
| 3 | yolo11n-1024 | 75.7 % | 10.5 % | 0.761 | 39.6 |
| 4 | yolov10n | 73.0 % | 11.2 % | 0.735 | 19.8 |
| 5 | yolo12n | 72.4 % | 13.8 % | 0.723 | 27.5 |
| 6 | yolo11n | 72.4 % | 14.5 % | 0.728 | 10.5 |
| 7 | yolo11n-hsv | 71.7 % | 13.8 % | 0.719 | 14.2 |
| 8 | yolov9t | 69.1 % | 15.8 % | 0.748 | 27.4 |
| 9 | yolov8n | 63.8 % | 19.1 % | 0.739 | 14.5 |
| – | yolov9e (grande) | 77.0 % | 10.5 % | 0.753 | 109.2 |

Los modelos grandes se agregan a medida que terminan. yolov9e (25 veces más pesado) no supera a yolo26n.
