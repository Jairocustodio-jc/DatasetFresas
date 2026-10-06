# Benchmark YOLO · cv5_6lotes

Actualizado 06/10/2026 19:02 (hora de Kaggle, UTC). Lotes: lote_01, lote_02, lote_03, lote_04, lote_05, lote_06. 1564 fotos. GPU: Tesla T4, Tesla T4.

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
| 1 | yolov9t-1024-f2 | 78.3 % | 9.9 % | 0.758 | 0.886 | 0.84 | 0.83 | 0.7539 | 62 (mejor 42) | 70.9 | 2.01 | 7.9 | 10.26 |
| 2 | yolov9t-1024-f1 | 75.6 % | 11.2 % | 0.758 | 0.888 | 0.84 | 0.83 | 0.7566 | 41 (mejor 21) | 46.0 | 2.01 | 7.9 | 10.28 |
| 3 | yolo26n-f2 | 73.7 % | 10.6 % | 0.747 | 0.881 | 0.84 | 0.85 | 0.744 | 54 (mejor 34) | 22.3 | 2.51 | 5.9 | 4.09 |
| 4 | yolo26n-f3 | 72.4 % | 13.8 % | 0.763 | 0.893 | 0.80 | 0.87 | 0.7439 | 34 (mejor 14) | 14.8 | 2.51 | 5.9 | 5.18 |
| 5 | yolo11n-1024-f2 | 69.7 % | 15.8 % | 0.725 | 0.861 | 0.79 | 0.85 | 0.7456 | 42 (mejor 22) | 35.6 | 2.59 | 6.5 | 7.53 |
| 6 | yolo11n-1024-f1 | 69.0 % | 17.1 % | 0.746 | 0.869 | 0.77 | 0.86 | 0.7461 | 50 (mejor 30) | 41.2 | 2.59 | 6.5 | 7.48 |
| 7 | yolo26n-f1 | 61.8 % | 23.0 % | 0.724 | 0.855 | 0.78 | 0.85 | 0.7498 | 40 (mejor 20) | 17.0 | 2.51 | 5.9 | 4.2 |

**AP50 por clase (test)**

| Modelo | unripe | early-pink | commercial-basic | commercial-high | overripe |
|---|---|---|---|---|---|
| yolov9t-1024-f2 | 0.99 | 0.89 | 0.78 | 0.82 | 0.95 |
| yolov9t-1024-f1 | 0.98 | 0.88 | 0.76 | 0.86 | 0.96 |
| yolo26n-f2 | 0.99 | 0.89 | 0.74 | 0.83 | 0.96 |
| yolo26n-f3 | 0.98 | 0.89 | 0.79 | 0.85 | 0.96 |
| yolo11n-1024-f2 | 0.99 | 0.88 | 0.67 | 0.81 | 0.95 |
| yolo11n-1024-f1 | 0.99 | 0.89 | 0.71 | 0.80 | 0.95 |
| yolo26n-f1 | 0.98 | 0.91 | 0.67 | 0.77 | 0.94 |

**A qué se predicen las cajas comerciales del test** (% de las cajas reales de cada clase)

| Modelo | Clase real | Cajas | ✓ bien | → early-pink | → overripe | → la otra comercial | → unripe | no detectada |
|---|---|---|---|---|---|---|---|---|
| yolov9t-1024-f2 | commercial-basic | 51 | 78.4 | **3.9** | **0.0** | 17.6 | 0.0 | 0.0 |
| yolov9t-1024-f2 | commercial-high | 101 | 78.2 | **1.0** | **11.9** | 8.9 | 0.0 | 0.0 |
| yolov9t-1024-f1 | commercial-basic | 51 | 72.5 | **5.9** | **0.0** | 21.6 | 0.0 | 0.0 |
| yolov9t-1024-f1 | commercial-high | 101 | 77.2 | **0.0** | **13.9** | 8.9 | 0.0 | 0.0 |
| yolo26n-f2 | commercial-basic | 51 | 62.7 | **5.9** | **0.0** | 31.4 | 0.0 | 0.0 |
| yolo26n-f2 | commercial-high | 101 | 79.2 | **1.0** | **11.9** | 7.9 | 0.0 | 0.0 |
| yolo26n-f3 | commercial-basic | 51 | 74.5 | **5.9** | **0.0** | 19.6 | 0.0 | 0.0 |
| yolo26n-f3 | commercial-high | 101 | 71.3 | **0.0** | **17.8** | 10.9 | 0.0 | 0.0 |
| yolo11n-1024-f2 | commercial-basic | 51 | 58.8 | **17.6** | **0.0** | 23.5 | 0.0 | 0.0 |
| yolo11n-1024-f2 | commercial-high | 101 | 75.2 | **1.0** | **13.9** | 9.9 | 0.0 | 0.0 |
| yolo11n-1024-f1 | commercial-basic | 51 | 72.5 | **13.7** | **0.0** | 13.7 | 0.0 | 0.0 |
| yolo11n-1024-f1 | commercial-high | 101 | 67.3 | **1.0** | **17.8** | 13.9 | 0.0 | 0.0 |
| yolo26n-f1 | commercial-basic | 51 | 66.7 | **15.7** | **0.0** | 17.6 | 0.0 | 0.0 |
| yolo26n-f1 | commercial-high | 101 | 59.4 | **0.0** | **26.7** | 13.9 | 0.0 | 0.0 |

**Validación cruzada: media ± desviación entre folds, medida en el test fijo**

| Modelo | Folds | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall | Especif. | Acierto com. | Error crít. | mAP@0.5 (val del fold) |
|---|---|---|---|---|---|---|---|---|---|
| yolov9t-1024 | 2 | 88.7 ± 0.2 | 75.8 ± 0.1 | 84.3 ± 0.1 | 82.9 ± 0.6 | 94.6 ± 0.4 | 76.9 ± 1.9 | 10.6 ± 0.9 | 88.4 ± 0.3 |
| yolo11n-1024 | 2 | 86.5 ± 0.6 | 73.6 ± 1.5 | 78.0 ± 0.8 | 85.2 ± 0.5 | 94.9 ± 0.0 | 69.3 ± 0.5 | 16.5 ± 0.9 | 87.7 ± 0.7 |
| yolo26n | 3 | 87.6 ± 2.0 | 74.5 ± 1.9 | 80.5 ± 2.9 | 85.7 ± 0.8 | 95.0 ± 0.4 | 69.3 ± 6.5 | 15.8 ± 6.4 | 87.7 ± 0.7 |

**Sin resultado todavía**

- yolov9t-1024-f3: entrenando
- yolo11n-1024-f3: entrenando
- yolov9t-1024-f4: pendiente
- yolo26n-f4: pendiente
- yolo11n-1024-f4: pendiente
- yolov9t-1024-f5: pendiente
- yolo26n-f5: pendiente
- yolo11n-1024-f5: pendiente

⏱ = se detuvo por el tope de tiempo; se guarda su mejor época. Pesos: release `benchmark-cv5_6lotes` del repo. Mismas condiciones para todos: ver modelos/kaggle_benchmark.py.
