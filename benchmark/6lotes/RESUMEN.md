# Benchmark YOLO · 6lotes

Actualizado 05/10/2026 09:10 (hora de Kaggle, UTC). Lotes: lote_01, lote_02, lote_03, lote_04, lote_05, lote_06. 1564 fotos. GPU: Tesla T4, Tesla T4.

**Reparto estratificado por estado y por lote** (fotos y cajas de cada estado en cada parte):

| Parte | Fotos | unripe | early-pink | commercial-basic | commercial-high | overripe | Cajas |
|---|---|---|---|---|---|---|---|
| train | 1094 (70%) | 3777 (67%) | 393 (7%) | 240 (4%) | 470 (8%) | 785 (14%) | 5665 |
| val | 235 (15%) | 844 (67%) | 84 (7%) | 52 (4%) | 101 (8%) | 174 (14%) | 1255 |
| test | 235 (15%) | 789 (66%) | 84 (7%) | 51 (4%) | 101 (8%) | 176 (15%) | 1201 |

En train, las fotos con cajas comerciales van repetidas (564 copias extra) y la pérdida de clasificación pesa más las clases escasas (cls_pw 0.5). Validación y test quedan con la distribución real.

**Resultados en test** (fotos que ningún modelo vio al entrenar ni usó para elegir su mejor época). Ordenados por **acierto comercial** = % de cajas commercial-basic/high detectadas y con su estado correcto (más es mejor); a igualdad, por menor **error crítico** = % que el modelo llamó early-pink u overripe (menos es mejor; las no detectadas no cuentan aquí, ver la tabla de abajo). «val» = mAP50-95 en validación.

| # | Modelo | Acierto com. | Error crítico | mAP50-95 | mAP50 | P | R | val | Épocas | Min | Params (M) | GFLOPs | ms/img |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | yolov8n | 42.2 % | 30.9 % | 0.739 | 0.870 | 0.78 | 0.87 | 0.7516 | 45 (mejor 25) | 14.5 | 3.01 | 8.2 | 3.88 |
| 2 | yolo26n | 36.2 % | 26.3 % | 0.744 | 0.875 | 0.84 | 0.85 | 0.7556 | 43 (mejor 23) | 19.8 | 2.51 | 5.9 | 4.94 |
| 3 | yolov10n | 35.5 % | 39.5 % | 0.735 | 0.861 | 0.82 | 0.83 | 0.7469 | 45 (mejor 25) | 19.8 | 2.71 | 8.5 | 5.69 |
| 4 | yolo12n | 33.6 % | 40.1 % | 0.723 | 0.848 | 0.82 | 0.84 | 0.7471 | 58 (mejor 49) ⏱ | 27.5 | 2.57 | 7.5 | 6.19 |
| 5 | yolo11n-hsv-1024 | 32.3 % | 34.2 % | 0.740 | 0.866 | 0.82 | 0.83 | 0.7576 | 58 (mejor 38) | 43.8 | 2.59 | 6.5 | 7.5 |
| 6 | yolo11n-1024 | 28.9 % | 38.2 % | 0.761 | 0.890 | 0.86 | 0.78 | 0.7557 | 46 (mejor 26) | 39.6 | 2.59 | 6.5 | 8.08 |
| 7 | yolo11n-hsv | 28.3 % | 34.9 % | 0.719 | 0.853 | 0.80 | 0.85 | 0.7516 | 36 (mejor 16) | 14.2 | 2.59 | 6.5 | 5.4 |
| 8 | yolov9t | 25.6 % | 38.8 % | 0.748 | 0.877 | 0.78 | 0.87 | 0.7612 | 44 (mejor 33) ⏱ | 27.4 | 2.01 | 7.9 | 6.1 |
| 9 | yolo11n | 25.0 % | 44.1 % | 0.728 | 0.869 | 0.80 | 0.80 | 0.7417 | 27 (mejor 7) | 10.5 | 2.59 | 6.5 | 4.3 |

**AP50 por clase (test)**

| Modelo | unripe | early-pink | commercial-basic | commercial-high | overripe |
|---|---|---|---|---|---|
| yolov8n | 0.98 | 0.90 | 0.74 | 0.79 | 0.94 |
| yolo26n | 0.99 | 0.91 | 0.73 | 0.79 | 0.95 |
| yolov10n | 0.98 | 0.88 | 0.68 | 0.81 | 0.95 |
| yolo12n | 0.99 | 0.90 | 0.64 | 0.77 | 0.95 |
| yolo11n-hsv-1024 | 0.99 | 0.87 | 0.73 | 0.78 | 0.96 |
| yolo11n-1024 | 0.98 | 0.89 | 0.79 | 0.83 | 0.96 |
| yolo11n-hsv | 0.98 | 0.88 | 0.64 | 0.81 | 0.96 |
| yolov9t | 0.98 | 0.89 | 0.73 | 0.83 | 0.96 |
| yolo11n | 0.98 | 0.86 | 0.75 | 0.80 | 0.96 |

**A qué se predicen las cajas comerciales del test** (% de las cajas reales de cada clase)

| Modelo | Clase real | Cajas | ✓ bien | → early-pink | → overripe | → la otra comercial | → unripe | no detectada |
|---|---|---|---|---|---|---|---|---|
| yolov8n | commercial-basic | 51 | 37.3 | **31.4** | **7.8** | 23.5 | 0.0 | 0.0 |
| yolov8n | commercial-high | 101 | 44.6 | **7.9** | **18.8** | 27.7 | 1.0 | 0.0 |
| yolo26n | commercial-basic | 51 | 27.5 | **15.7** | **9.8** | 27.5 | 19.6 | 0.0 |
| yolo26n | commercial-high | 101 | 40.6 | **5.9** | **20.8** | 23.8 | 8.9 | 0.0 |
| yolov10n | commercial-basic | 51 | 33.3 | **29.4** | **13.7** | 13.7 | 9.8 | 0.0 |
| yolov10n | commercial-high | 101 | 36.6 | **5.9** | **31.7** | 25.7 | 0.0 | 0.0 |
| yolo12n | commercial-basic | 51 | 37.3 | **23.5** | **21.6** | 13.7 | 3.9 | 0.0 |
| yolo12n | commercial-high | 101 | 31.7 | **5.0** | **32.7** | 25.7 | 5.0 | 0.0 |
| yolo11n-hsv-1024 | commercial-basic | 51 | 27.5 | **25.5** | **5.9** | 37.3 | 3.9 | 0.0 |
| yolo11n-hsv-1024 | commercial-high | 101 | 34.7 | **5.0** | **30.7** | 28.7 | 1.0 | 0.0 |
| yolo11n-1024 | commercial-basic | 51 | 31.4 | **25.5** | **17.6** | 19.6 | 5.9 | 0.0 |
| yolo11n-1024 | commercial-high | 101 | 27.7 | **5.9** | **29.7** | 28.7 | 7.9 | 0.0 |
| yolo11n-hsv | commercial-basic | 51 | 35.3 | **15.7** | **9.8** | 17.6 | 21.6 | 0.0 |
| yolo11n-hsv | commercial-high | 101 | 24.8 | **10.9** | **28.7** | 29.7 | 5.9 | 0.0 |
| yolov9t | commercial-basic | 51 | 23.5 | **9.8** | **25.5** | 23.5 | 17.6 | 0.0 |
| yolov9t | commercial-high | 101 | 26.7 | **8.9** | **31.7** | 16.8 | 15.8 | 0.0 |
| yolo11n | commercial-basic | 51 | 21.6 | **13.7** | **25.5** | 29.4 | 9.8 | 0.0 |
| yolo11n | commercial-high | 101 | 26.7 | **13.9** | **32.7** | 26.7 | 0.0 | 0.0 |

**Experimentos con yolo11n: aumento de tono y resolución** (mismo test)

| Variante | hsv_h | imgsz | Acierto com. | Error crítico | mAP50-95 | ms/img |
|---|---|---|---|---|---|---|
| yolo11n | 0.0 | 640 | 25.0 % | 44.1 % | 0.728 | 4.3 |
| yolo11n-hsv | 0.015 | 640 | 28.3 % | 34.9 % | 0.719 | 5.4 |
| yolo11n-1024 | 0.0 | 1024 | 28.9 % | 38.2 % | 0.761 | 8.08 |
| yolo11n-hsv-1024 | 0.015 | 1024 | 32.3 % | 34.2 % | 0.740 | 7.5 |

**Sin resultado todavía**

- yolov9e: entrenando
- yolo12x: entrenando
- yolov8x: pendiente
- yolov10x: pendiente
- yolo11x: pendiente
- yolo26x: pendiente

⏱ = se detuvo por el tope de tiempo; se guarda su mejor época. Pesos: release `benchmark-6lotes` del repo. Mismas condiciones para todos: ver modelos/kaggle_benchmark.py.
