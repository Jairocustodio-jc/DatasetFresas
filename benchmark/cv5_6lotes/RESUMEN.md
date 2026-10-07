# Benchmark YOLO · cv5_6lotes

Actualizado 07/10/2026 02:38 (hora de Kaggle, UTC). Lotes: lote_01, lote_02, lote_03, lote_04, lote_05, lote_06. 1564 fotos. GPU: Tesla T4, Tesla T4.

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
| 1 | yolov9t-1024-f5 | 79.0 % | 6.6 % | 0.758 | 0.881 | 0.81 | 0.86 | 0.7447 | 50 (mejor 30) | 58.5 | 2.01 | 7.9 | 9.54 |
| 2 | yolo12n-1024-f1 | 79.0 % | 9.2 % | 0.750 | 0.875 | 0.82 | 0.86 | 0.7629 | 48 (mejor 28) | 65.7 | 2.57 | 7.5 | 13.56 |
| 3 | yolo12n-1024-f3 | 78.9 % | 5.2 % | 0.755 | 0.882 | 0.83 | 0.85 | 0.7399 | 45 (mejor 25) | 64.1 | 2.57 | 7.5 | 14.09 |
| 4 | yolov9t-1024-f2 | 78.3 % | 9.9 % | 0.758 | 0.886 | 0.84 | 0.83 | 0.7539 | 62 (mejor 42) | 70.9 | 2.01 | 7.9 | 10.26 |
| 5 | yolov10n-1024-f3 | 77.0 % | 6.6 % | 0.749 | 0.880 | 0.80 | 0.87 | 0.7459 | 40 (mejor 20) | 41.5 | 2.71 | 8.5 | 8.02 |
| 6 | yolo11n-1024-f4 | 76.3 % | 9.9 % | 0.750 | 0.875 | 0.80 | 0.86 | 0.7596 | 55 (mejor 35) | 45.7 | 2.59 | 6.5 | 7.69 |
| 7 | yolo11n-1024-f5 | 76.3 % | 11.9 % | 0.766 | 0.892 | 0.83 | 0.84 | 0.762 | 52 (mejor 32) | 37.8 | 2.59 | 6.5 | 7.72 |
| 8 | yolov9t-1024-f1 | 75.6 % | 11.2 % | 0.758 | 0.888 | 0.84 | 0.83 | 0.7566 | 41 (mejor 21) | 46.0 | 2.01 | 7.9 | 10.28 |
| 9 | yolo12n-1024-f2 | 75.0 % | 7.9 % | 0.744 | 0.866 | 0.83 | 0.83 | 0.7591 | 63 (mejor 43) | 86.5 | 2.57 | 7.5 | 13.71 |
| 10 | yolo26n-f2 | 73.7 % | 10.6 % | 0.747 | 0.881 | 0.84 | 0.85 | 0.744 | 54 (mejor 34) | 22.3 | 2.51 | 5.9 | 4.09 |
| 11 | yolov9t-1024-f3 | 73.7 % | 13.1 % | 0.745 | 0.867 | 0.84 | 0.84 | 0.7484 | 64 (mejor 44) | 74.0 | 2.01 | 7.9 | 9.84 |
| 12 | yolo11n-1024-f3 | 72.4 % | 12.5 % | 0.766 | 0.896 | 0.82 | 0.88 | 0.7481 | 52 (mejor 32) | 44.2 | 2.59 | 6.5 | 8.25 |
| 13 | yolo26n-f3 | 72.4 % | 13.8 % | 0.763 | 0.893 | 0.80 | 0.87 | 0.7439 | 34 (mejor 14) | 14.8 | 2.51 | 5.9 | 5.18 |
| 14 | yolov9t-1024-f4 | 71.8 % | 13.8 % | 0.766 | 0.891 | 0.84 | 0.83 | 0.7601 | 56 (mejor 36) | 64.5 | 2.01 | 7.9 | 9.91 |
| 15 | yolov10n-1024-f1 | 70.4 % | 13.2 % | 0.724 | 0.855 | 0.78 | 0.84 | 0.7497 | 35 (mejor 15) | 37.6 | 2.71 | 8.5 | 7.75 |
| 16 | yolov10n-1024-f2 | 70.4 % | 15.1 % | 0.724 | 0.852 | 0.85 | 0.82 | 0.7557 | 66 (mejor 46) | 67.3 | 2.71 | 8.5 | 8.1 |
| 17 | yolo26n-f5 | 69.7 % | 13.1 % | 0.750 | 0.873 | 0.80 | 0.85 | 0.7438 | 41 (mejor 21) | 17.0 | 2.51 | 5.9 | 4.3 |
| 18 | yolo11n-1024-f2 | 69.7 % | 15.8 % | 0.725 | 0.861 | 0.79 | 0.85 | 0.7456 | 42 (mejor 22) | 35.6 | 2.59 | 6.5 | 7.53 |
| 19 | yolo11n-1024-f1 | 69.0 % | 17.1 % | 0.746 | 0.869 | 0.77 | 0.86 | 0.7461 | 50 (mejor 30) | 41.2 | 2.59 | 6.5 | 7.48 |
| 20 | yolo26n-f4 | 66.5 % | 17.1 % | 0.748 | 0.874 | 0.77 | 0.86 | 0.7316 | 43 (mejor 23) | 17.5 | 2.51 | 5.9 | 4.64 |
| 21 | yolo26n-f1 | 61.8 % | 23.0 % | 0.724 | 0.855 | 0.78 | 0.85 | 0.7498 | 40 (mejor 20) | 17.0 | 2.51 | 5.9 | 4.2 |

**AP50 por clase (test)**

| Modelo | unripe | early-pink | commercial-basic | commercial-high | overripe |
|---|---|---|---|---|---|
| yolov9t-1024-f5 | 0.99 | 0.91 | 0.73 | 0.82 | 0.96 |
| yolo12n-1024-f1 | 0.99 | 0.89 | 0.70 | 0.85 | 0.96 |
| yolo12n-1024-f3 | 0.99 | 0.90 | 0.73 | 0.84 | 0.96 |
| yolov9t-1024-f2 | 0.99 | 0.89 | 0.78 | 0.82 | 0.95 |
| yolov10n-1024-f3 | 0.99 | 0.90 | 0.77 | 0.78 | 0.96 |
| yolo11n-1024-f4 | 0.99 | 0.90 | 0.75 | 0.79 | 0.94 |
| yolo11n-1024-f5 | 0.99 | 0.88 | 0.77 | 0.86 | 0.96 |
| yolov9t-1024-f1 | 0.98 | 0.88 | 0.76 | 0.86 | 0.96 |
| yolo12n-1024-f2 | 0.98 | 0.86 | 0.69 | 0.84 | 0.97 |
| yolo26n-f2 | 0.99 | 0.89 | 0.74 | 0.83 | 0.96 |
| yolov9t-1024-f3 | 0.99 | 0.87 | 0.71 | 0.80 | 0.96 |
| yolo11n-1024-f3 | 0.99 | 0.91 | 0.81 | 0.82 | 0.96 |
| yolo26n-f3 | 0.98 | 0.89 | 0.79 | 0.85 | 0.96 |
| yolov9t-1024-f4 | 0.99 | 0.90 | 0.78 | 0.82 | 0.96 |
| yolov10n-1024-f1 | 0.98 | 0.89 | 0.69 | 0.76 | 0.95 |
| yolov10n-1024-f2 | 0.98 | 0.86 | 0.68 | 0.80 | 0.95 |
| yolo26n-f5 | 0.99 | 0.90 | 0.72 | 0.81 | 0.96 |
| yolo11n-1024-f2 | 0.99 | 0.88 | 0.67 | 0.81 | 0.95 |
| yolo11n-1024-f1 | 0.99 | 0.89 | 0.71 | 0.80 | 0.95 |
| yolo26n-f4 | 0.99 | 0.90 | 0.73 | 0.80 | 0.95 |
| yolo26n-f1 | 0.98 | 0.91 | 0.67 | 0.77 | 0.94 |

**A qué se predicen las cajas comerciales del test** (% de las cajas reales de cada clase)

| Modelo | Clase real | Cajas | ✓ bien | → early-pink | → overripe | → la otra comercial | → unripe | no detectada |
|---|---|---|---|---|---|---|---|---|
| yolov9t-1024-f5 | commercial-basic | 51 | 76.5 | **9.8** | **0.0** | 13.7 | 0.0 | 0.0 |
| yolov9t-1024-f5 | commercial-high | 101 | 80.2 | **0.0** | **5.0** | 14.9 | 0.0 | 0.0 |
| yolo12n-1024-f1 | commercial-basic | 51 | 76.5 | **7.8** | **0.0** | 15.7 | 0.0 | 0.0 |
| yolo12n-1024-f1 | commercial-high | 101 | 80.2 | **0.0** | **9.9** | 9.9 | 0.0 | 0.0 |
| yolo12n-1024-f3 | commercial-basic | 51 | 72.5 | **0.0** | **0.0** | 27.5 | 0.0 | 0.0 |
| yolo12n-1024-f3 | commercial-high | 101 | 82.2 | **0.0** | **7.9** | 9.9 | 0.0 | 0.0 |
| yolov9t-1024-f2 | commercial-basic | 51 | 78.4 | **3.9** | **0.0** | 17.6 | 0.0 | 0.0 |
| yolov9t-1024-f2 | commercial-high | 101 | 78.2 | **1.0** | **11.9** | 8.9 | 0.0 | 0.0 |
| yolov10n-1024-f3 | commercial-basic | 51 | 64.7 | **3.9** | **0.0** | 31.4 | 0.0 | 0.0 |
| yolov10n-1024-f3 | commercial-high | 101 | 83.2 | **0.0** | **7.9** | 8.9 | 0.0 | 0.0 |
| yolo11n-1024-f4 | commercial-basic | 51 | 72.5 | **2.0** | **0.0** | 25.5 | 0.0 | 0.0 |
| yolo11n-1024-f4 | commercial-high | 101 | 78.2 | **1.0** | **12.9** | 7.9 | 0.0 | 0.0 |
| yolo11n-1024-f5 | commercial-basic | 51 | 80.4 | **9.8** | **0.0** | 9.8 | 0.0 | 0.0 |
| yolo11n-1024-f5 | commercial-high | 101 | 74.3 | **1.0** | **11.9** | 12.9 | 0.0 | 0.0 |
| yolov9t-1024-f1 | commercial-basic | 51 | 72.5 | **5.9** | **0.0** | 21.6 | 0.0 | 0.0 |
| yolov9t-1024-f1 | commercial-high | 101 | 77.2 | **0.0** | **13.9** | 8.9 | 0.0 | 0.0 |
| yolo12n-1024-f2 | commercial-basic | 51 | 70.6 | **3.9** | **0.0** | 25.5 | 0.0 | 0.0 |
| yolo12n-1024-f2 | commercial-high | 101 | 77.2 | **0.0** | **9.9** | 12.9 | 0.0 | 0.0 |
| yolo26n-f2 | commercial-basic | 51 | 62.7 | **5.9** | **0.0** | 31.4 | 0.0 | 0.0 |
| yolo26n-f2 | commercial-high | 101 | 79.2 | **1.0** | **11.9** | 7.9 | 0.0 | 0.0 |
| yolov9t-1024-f3 | commercial-basic | 51 | 80.4 | **3.9** | **0.0** | 15.7 | 0.0 | 0.0 |
| yolov9t-1024-f3 | commercial-high | 101 | 70.3 | **1.0** | **16.8** | 10.9 | 0.0 | 1.0 |
| yolo11n-1024-f3 | commercial-basic | 51 | 80.4 | **2.0** | **0.0** | 17.6 | 0.0 | 0.0 |
| yolo11n-1024-f3 | commercial-high | 101 | 68.3 | **0.0** | **17.8** | 13.9 | 0.0 | 0.0 |
| yolo26n-f3 | commercial-basic | 51 | 74.5 | **5.9** | **0.0** | 19.6 | 0.0 | 0.0 |
| yolo26n-f3 | commercial-high | 101 | 71.3 | **0.0** | **17.8** | 10.9 | 0.0 | 0.0 |
| yolov9t-1024-f4 | commercial-basic | 51 | 66.7 | **13.7** | **0.0** | 19.6 | 0.0 | 0.0 |
| yolov9t-1024-f4 | commercial-high | 101 | 74.3 | **0.0** | **13.9** | 11.9 | 0.0 | 0.0 |
| yolov10n-1024-f1 | commercial-basic | 51 | 64.7 | **7.8** | **0.0** | 27.5 | 0.0 | 0.0 |
| yolov10n-1024-f1 | commercial-high | 101 | 73.3 | **1.0** | **14.9** | 10.9 | 0.0 | 0.0 |
| yolov10n-1024-f2 | commercial-basic | 51 | 68.6 | **11.8** | **0.0** | 19.6 | 0.0 | 0.0 |
| yolov10n-1024-f2 | commercial-high | 101 | 71.3 | **1.0** | **15.8** | 11.9 | 0.0 | 0.0 |
| yolo26n-f5 | commercial-basic | 51 | 70.6 | **5.9** | **0.0** | 23.5 | 0.0 | 0.0 |
| yolo26n-f5 | commercial-high | 101 | 69.3 | **0.0** | **16.8** | 13.9 | 0.0 | 0.0 |
| yolo11n-1024-f2 | commercial-basic | 51 | 58.8 | **17.6** | **0.0** | 23.5 | 0.0 | 0.0 |
| yolo11n-1024-f2 | commercial-high | 101 | 75.2 | **1.0** | **13.9** | 9.9 | 0.0 | 0.0 |
| yolo11n-1024-f1 | commercial-basic | 51 | 72.5 | **13.7** | **0.0** | 13.7 | 0.0 | 0.0 |
| yolo11n-1024-f1 | commercial-high | 101 | 67.3 | **1.0** | **17.8** | 13.9 | 0.0 | 0.0 |
| yolo26n-f4 | commercial-basic | 51 | 56.9 | **19.6** | **0.0** | 23.5 | 0.0 | 0.0 |
| yolo26n-f4 | commercial-high | 101 | 71.3 | **0.0** | **15.8** | 12.9 | 0.0 | 0.0 |
| yolo26n-f1 | commercial-basic | 51 | 66.7 | **15.7** | **0.0** | 17.6 | 0.0 | 0.0 |
| yolo26n-f1 | commercial-high | 101 | 59.4 | **0.0** | **26.7** | 13.9 | 0.0 | 0.0 |

**Validación cruzada: media ± desviación entre folds, medida en el test fijo**

| Modelo | Folds | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall | Especif. | Acierto com. | Error crít. | mAP@0.5 (val del fold) |
|---|---|---|---|---|---|---|---|---|---|
| yolo12n-1024 | 3 | 87.4 ± 0.8 | 75.0 ± 0.6 | 82.8 ± 0.8 | 84.9 ± 1.3 | 94.3 ± 0.4 | 77.6 ± 2.3 | 7.4 ± 2.0 | 88.5 ± 1.0 |
| yolov9t-1024 | 5 | 88.3 ± 0.9 | 75.7 ± 0.8 | 83.7 ± 1.3 | 83.9 ± 1.5 | 94.6 ± 0.5 | 75.7 ± 3.0 | 10.9 ± 2.9 | 88.0 ± 0.9 |
| yolo11n-1024 | 5 | 87.9 ± 1.5 | 75.1 ± 1.7 | 80.1 ± 2.2 | 85.6 ± 1.3 | 94.5 ± 0.4 | 72.7 ± 3.5 | 13.4 ± 2.9 | 88.0 ± 0.5 |
| yolov10n-1024 | 3 | 86.2 ± 1.5 | 73.2 ± 1.4 | 81.1 ± 3.3 | 84.2 ± 2.4 | 93.8 ± 0.4 | 72.6 ± 3.8 | 11.6 ± 4.5 | 87.8 ± 0.2 |
| yolo26n | 5 | 87.5 ± 1.4 | 74.7 ± 1.4 | 79.7 ± 2.5 | 85.5 ± 0.7 | 94.9 ± 0.4 | 68.8 ± 4.8 | 15.5 ± 4.8 | 87.1 ± 0.9 |

**Sin resultado todavía**

- yolov10n-1024-f4: entrenando
- yolo12n-1024-f4: entrenando
- yolov10n-1024-f5: pendiente
- yolo12n-1024-f5: pendiente

⏱ = se detuvo por el tope de tiempo; se guarda su mejor época. Pesos: release `benchmark-cv5_6lotes` del repo. Mismas condiciones para todos: ver modelos/kaggle_benchmark.py.
