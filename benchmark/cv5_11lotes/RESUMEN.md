# Benchmark YOLO · cv5_11lotes

Actualizado 04/11/2026 04:02 (hora de Kaggle, UTC). Lotes: lote_01, lote_02, lote_03, lote_04, lote_05, lote_06, lote_07, lote_08, lote_09, lote_10, lote_11. 3077 fotos. GPU: Tesla T4, Tesla T4.

**Reparto estratificado por estado y por lote** (fotos y cajas de cada estado en cada parte):

| Parte | Fotos | unripe | early-pink | commercial-basic | commercial-high | overripe | Cajas |
|---|---|---|---|---|---|---|---|
| train | 2151 (70%) | 7437 (66%) | 790 (7%) | 403 (4%) | 926 (8%) | 1628 (15%) | 11184 |
| val | 463 (15%) | 1571 (65%) | 169 (7%) | 86 (4%) | 198 (8%) | 378 (16%) | 2402 |
| test | 463 (15%) | 1633 (67%) | 170 (7%) | 86 (4%) | 198 (8%) | 333 (14%) | 2420 |

En train, las fotos con cajas comerciales van repetidas (1064 copias extra) y la pérdida de clasificación pesa más las clases escasas (cls_pw 0.5). Validación y test quedan con la distribución real.

**Resultados en test** (fotos que ningún modelo vio al entrenar ni usó para elegir su mejor época). Ordenados por **acierto comercial** = % de cajas commercial-basic/high detectadas y con su estado correcto (más es mejor); a igualdad, por menor **error crítico** = % que el modelo llamó early-pink u overripe (menos es mejor; las no detectadas no cuentan aquí, ver la tabla de abajo). «val» = mAP50-95 en validación.

| # | Modelo | Acierto com. | Error crítico | mAP50-95 | mAP50 | P | R | val | Épocas | Min | Params (M) | GFLOPs | ms/img |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

**Sin resultado todavía**

- yolo12n-1024-f1: pendiente
- yolo12n-1024-f2: pendiente
- yolo12n-1024-f3: pendiente
- yolo12n-1024-f4: pendiente
- yolo12n-1024-f5: pendiente

⏱ = se detuvo por el tope de tiempo; se guarda su mejor época. Pesos: release `benchmark-cv5_11lotes` del repo. Mismas condiciones para todos: ver modelos/kaggle_benchmark.py.
