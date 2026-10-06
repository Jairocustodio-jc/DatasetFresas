# Benchmark YOLO · cv5_6lotes

Actualizado 06/10/2026 17:26 (hora de Kaggle, UTC). Lotes: lote_01, lote_02, lote_03, lote_04, lote_05, lote_06. 1564 fotos. GPU: Tesla T4, Tesla T4.

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
| 1 | yolo26n-f1 | 61.8 % | 23.0 % | 0.724 | 0.855 | 0.78 | 0.85 | 0.7498 | 40 (mejor 20) | 17.0 | 2.51 | 5.9 | 4.2 |

**AP50 por clase (test)**

| Modelo | unripe | early-pink | commercial-basic | commercial-high | overripe |
|---|---|---|---|---|---|
| yolo26n-f1 | 0.98 | 0.91 | 0.67 | 0.77 | 0.94 |

**A qué se predicen las cajas comerciales del test** (% de las cajas reales de cada clase)

| Modelo | Clase real | Cajas | ✓ bien | → early-pink | → overripe | → la otra comercial | → unripe | no detectada |
|---|---|---|---|---|---|---|---|---|
| yolo26n-f1 | commercial-basic | 51 | 66.7 | **15.7** | **0.0** | 17.6 | 0.0 | 0.0 |
| yolo26n-f1 | commercial-high | 101 | 59.4 | **0.0** | **26.7** | 13.9 | 0.0 | 0.0 |

**Validación cruzada: media ± desviación entre folds, medida en el test fijo**

| Modelo | Folds | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall | Especif. | Acierto com. | Error crít. | mAP@0.5 (val del fold) |
|---|---|---|---|---|---|---|---|---|---|
| yolo26n | 1 | 85.5 | 72.4 | 77.8 | 85.5 | 94.6 | 61.8 | 23.0 | 88.3 |

**Sin resultado todavía**

- yolov9t-1024-f1: entrenando
- yolo11n-1024-f1: entrenando
- yolov9t-1024-f2: pendiente
- yolo26n-f2: pendiente
- yolo11n-1024-f2: pendiente
- yolov9t-1024-f3: pendiente
- yolo26n-f3: pendiente
- yolo11n-1024-f3: pendiente
- yolov9t-1024-f4: pendiente
- yolo26n-f4: pendiente
- yolo11n-1024-f4: pendiente
- yolov9t-1024-f5: pendiente
- yolo26n-f5: pendiente
- yolo11n-1024-f5: pendiente

⏱ = se detuvo por el tope de tiempo; se guarda su mejor época. Pesos: release `benchmark-cv5_6lotes` del repo. Mismas condiciones para todos: ver modelos/kaggle_benchmark.py.
