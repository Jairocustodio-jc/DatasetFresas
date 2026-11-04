# Resultados: detección y clasificación del estado de madurez de la fresa

*Estado al 12 de octubre de 2026. Resultados preliminares: dataset parcialmente revisado (lotes 1–6).*

## 1. Conjunto de datos

### 1.1 Origen y clases
Se utilizó un conjunto de 3099 imágenes de fresa en campo (1008 × 756 px), anotadas con cajas delimitadoras en cinco
estados de madurez:

| Clase | Descripción operativa |
|---|---|
| `unripe` | Fruto verde o no desarrollado |
| `early-pink` | Inicio de coloración, por debajo del umbral comercial |
| `commercial-basic` | Grado comercial intermedio |
| `commercial-high` | Grado comercial de mayor exigencia |
| `overripe` | Sobremaduro, con pérdida de firmeza |

Las clases `commercial-basic` y `commercial-high` son las de mayor interés para la aplicación: confundirlas con
`early-pink` (cosecha anticipada) o con `overripe` (cosecha tardía) tiene el mayor costo operativo.

### 1.2 Revisión humana de las anotaciones
Las anotaciones originales se revisaron manualmente en lotes de 300 imágenes, con una herramienta web desarrollada para el
proyecto que permite marcar cada imagen y corregir cajas y etiquetas desde un dispositivo móvil. La revisión del lote 1
mostró que el 41 % de las cajas etiquetadas originalmente como `commercial-basic` debían corregirse, principalmente a
`early-pink`. Las etiquetas automáticas no eran confiables precisamente en las clases comerciales.

A partir del lote 2 se incorporó un esquema *human-in-the-loop*: un detector YOLO, reentrenado con los lotes ya revisados,
propone correcciones que el revisor acepta o rechaza. La proporción de sugerencias aceptadas aumentó a medida que el modelo
se reentrenó con más datos revisados:

| Lote revisado | Modelo que sugirió | Sugerencias aceptadas |
|---|---|---|
| 2 | Entrenado con el lote 1 | 53 % |
| 3 | Entrenado con los lotes 1–2 | 57 % |
| 6 (primeras 76 imágenes) | Entrenado con los lotes 1–5 | 84 % (16 de 19) |

Al momento de este informe se dispone de **1564 imágenes revisadas** (lotes 1–5 completos y parte del lote 6).

### 1.3 Partición
Se usó una partición **70 / 15 / 15** (entrenamiento / validación / prueba) estratificada simultáneamente por clase
(número de cajas de cada estado) y por lote, mediante estratificación iterativa. Las tres partes conservan la misma
proporción de cada estado:

| Parte | Imágenes | unripe | early-pink | commercial-basic | commercial-high | overripe |
|---|---|---|---|---|---|---|
| Entrenamiento | 1094 (70 %) | 3777 (67 %) | 393 (7 %) | 240 (4 %) | 470 (8 %) | 785 (14 %) |
| Validación | 235 (15 %) | 844 (67 %) | 84 (7 %) | 52 (4 %) | 101 (8 %) | 174 (14 %) |
| Prueba | 235 (15 %) | 789 (66 %) | 84 (7 %) | 51 (4 %) | 101 (8 %) | 176 (15 %) |

El conjunto de prueba (235 imágenes, 152 fresas comerciales) se mantuvo fijo en todos los experimentos y no se usó para
entrenar ni para seleccionar épocas. Se verificó mediante *hash* perceptual que no existen imágenes casi duplicadas entre
particiones.

## 2. Protocolo experimental

### 2.1 Receta de entrenamiento (común a todos los modelos)
- Inicialización con pesos preentrenados en COCO.
- Optimizador AdamW, tasa de aprendizaje inicial 0,001, hasta 100 épocas con parada temprana (paciencia 20); se conserva
  el peso de la mejor época según la validación.
- Aumentos: mosaico, volteo horizontal, escala y traslación; saturación ±0,3 y brillo ±0,3; sin aumento de tono
  (`hsv_h = 0`) salvo en los experimentos que lo indican.
- Clases minoritarias: las imágenes de entrenamiento con cajas comerciales se duplican (sobremuestreo ×2), y la pérdida de
  clasificación pondera las clases por el inverso de su frecuencia elevado a 0,5.
- Lote de 16 imágenes a 640 px y de 8 a 1024 px, con acumulación de gradiente hasta un lote nominal de 64.
- Entrenamiento en GPU NVIDIA T4 (Kaggle).

### 2.2 Métricas
Además de las métricas estándar de detección (mAP@0.5, mAP@0.5:0.95, precisión y *recall*), se definieron:

- **Especificidad por clase:** E = TN / (TN + FP), calculada sobre una matriz de confusión en conteos (confianza ≥ 0,25,
  IoU ≥ 0,45) y promediada sobre las cinco clases.
- **Acierto comercial:** porcentaje de fresas reales `commercial-basic` o `commercial-high` del conjunto de prueba que el
  modelo detecta y clasifica con su estado correcto. Se toma, para cada fresa, la predicción de mayor confianza que la
  cubre (IoU ≥ 0,5, confianza ≥ 0,25); las fresas no detectadas cuentan como fallo.
- **Error crítico:** porcentaje de esas fresas comerciales que el modelo clasifica como `early-pink` u `overripe`.

El acierto comercial y el error crítico son las métricas principales de decisión, ya que reflejan el costo operativo del
error. El mAP se reporta, pero no discrimina entre los modelos evaluados (todos se ubican entre 86 % y 89 % de mAP@0.5).

### 2.3 Diseño en dos fases
1. **Fase exploratoria:** una corrida por configuración (21 configuraciones) para seleccionar candidatos.
2. **Fase confirmatoria:** validación cruzada de 5 *folds* estratificados sobre entrenamiento y validación (1329 imágenes),
   con el mismo conjunto de prueba fijo. Cada modelo se entrena 5 veces y se reporta media ± desviación estándar sobre el
   conjunto de prueba. Las comparaciones entre modelos usan la prueba t pareada por *fold*, con corrección de Holm por
   comparaciones múltiples.

Solo los resultados de la fase confirmatoria se consideran conclusiones.

## 3. Fase exploratoria (una corrida por modelo)

### 3.1 Familias YOLO a 640 px: modelos nano frente a modelos grandes
Se compararon seis familias (YOLOv8, v9, v10, 11, 12 y 26) en su variante más pequeña (n; t en YOLOv9) y en la más grande
(x; e en YOLOv9), todas a 640 px.

| Familia | Nano: mAP@0.5 | Nano: acierto com. | Grande: mAP@0.5 | Grande: acierto com. |
|---|---|---|---|---|
| YOLOv8 (n / x) | 87,0 % | 63,8 % | 88,6 % | 67,1 % |
| YOLOv9 (t / e) | 87,8 % | 69,1 % | 89,1 % | 77,0 % |
| YOLOv10 (n / x) | 86,1 % | 73,0 % | 86,0 % | 72,4 % |
| YOLO11 (n / x) | 86,9 % | 72,4 % | 89,1 % | 77,0 % |
| YOLO12 (n / x) | 84,8 % | 72,4 % | 89,2 % | 72,3 % |
| YOLO26 (n / x) | 87,5 % | 77,0 % | 88,9 % | 73,0 % |

Los modelos grandes, con 10 a 34 veces más parámetros, no superaron de forma consistente a los nano en las clases
comerciales. Dado que el sistema debe ejecutarse en una Raspberry Pi, la fase confirmatoria se centró en los modelos nano.
*Este resultado proviene de una corrida única por modelo y no fue confirmado con validación cruzada.*

### 3.2 Resolución y aumento de tono
Se evaluó el efecto de entrenar a 1024 px y de añadir aumento de tono (`hsv_h = 0,015`). En una corrida única con YOLO11n,
la combinación de ambos redujo el error crítico de 14,5 % a 6,6 % (prueba de McNemar sobre las mismas 152 fresas,
p = 0,017). El aumento de tono por sí solo no tuvo efecto a 640 px, y su aporte adicional sobre 1024 px no fue
significativo (p = 0,11). Estos resultados se consideran indicios: provienen de un solo modelo y de una sola corrida.

### 3.3 Análisis cualitativo de atención
Los mapas EigenCAM sobre la capa P3 de la cabeza de detección mostraron que, en sus errores críticos (`commercial-high`
predicha como `overripe`), YOLO26n a 640 px concentraba la activación en la zona central más roja del fruto. Una variante
a 1024 px con aumento de tono distribuía la activación sobre una mayor superficie del fruto y corrigió cuatro de esos seis
errores. Este análisis es cualitativo: EigenCAM muestra zonas de activación, no la causa de la decisión de clase.

## 4. Fase confirmatoria: validación cruzada de 5 folds

### 4.1 Resultados sobre el conjunto de prueba fijo

| Modelo | Resolución | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall | Especificidad | Acierto comercial | Error crítico |
|---|---|---|---|---|---|---|---|---|
| YOLO12n | 1024 | 87,5 ± 0,6 | 74,8 ± 0,5 | 81,9 ± 1,4 | 84,8 ± 1,7 | 94,1 ± 1,1 | **76,2 ± 2,7** | **9,2 ± 3,4** |
| YOLOv9t | 1024 | **88,3 ± 0,9** | **75,7 ± 0,8** | **83,7 ± 1,3** | 83,9 ± 1,5 | 94,6 ± 0,5 | 75,7 ± 3,0 | 10,9 ± 2,9 |
| YOLO11n | 1024 | 87,9 ± 1,5 | 75,1 ± 1,7 | 80,1 ± 2,2 | **85,6 ± 1,3** | 94,5 ± 0,4 | 72,7 ± 3,5 | 13,4 ± 2,9 |
| YOLO26n | 1024 | 87,8 ± 0,6 | 75,6 ± 0,9 | 81,1 ± 1,9 | 84,9 ± 1,4 | 94,5 ± 0,5 | 72,4 ± 2,9 | 13,0 ± 3,8 |
| YOLOv10n | 1024 | 86,5 ± 1,1 | 73,7 ± 1,3 | 80,8 ± 2,6 | 84,7 ± 1,9 | 94,2 ± 0,7 | 71,3 ± 3,9 | 13,8 ± 4,7 |
| YOLO26n *(referencia)* | 640 | 87,5 ± 1,4 | 74,7 ± 1,4 | 79,7 ± 2,5 | 85,5 ± 0,7 | **94,9 ± 0,4** | 68,8 ± 4,8 | 15,5 ± 4,8 |

*Valores en %, media ± desviación estándar sobre 5 folds, evaluados en el mismo conjunto de prueba (235 imágenes).*

### 4.2 Acierto comercial por fold

| Modelo (1024 px) | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|
| YOLO12n | 79,0 | 75,0 | 78,9 | 73,0 | 75,0 |
| YOLOv9t | 75,6 | 78,3 | 73,7 | 71,8 | 79,0 |
| YOLO11n | 69,0 | 69,7 | 72,4 | 76,3 | 76,3 |
| YOLO26n | 71,1 | 68,4 | 76,3 | 73,7 | 72,3 |
| YOLOv10n | 70,4 | 70,4 | 77,0 | 66,4 | 72,4 |

### 4.3 Análisis estadístico
Se realizaron las 10 comparaciones por pares entre los cinco modelos a 1024 px mediante la prueba t pareada por fold,
con corrección de Holm. **Ninguna diferencia resultó significativa** (α = 0,05), ni en acierto comercial ni en error
crítico. Las comparaciones más fuertes fueron:

| Comparación | Diferencia | p (sin corregir) | p (Holm) |
|---|---|---|---|
| YOLO12n − YOLOv10n, acierto comercial | +4,9 pts | 0,017 | 0,17 |
| YOLO12n − YOLOv10n, error crítico | −4,6 pts | 0,008 | 0,08 |
| YOLO12n − YOLO26n, acierto comercial | +3,8 pts | 0,068 | 0,62 |
| YOLO12n − YOLOv9t, acierto comercial | +0,5 pts | 0,80 | 1,00 |

### 4.4 Efecto de la resolución en YOLO26n
Con los mismos folds, YOLO26n a 1024 px obtuvo +3,5 puntos de acierto comercial respecto de 640 px (68,8 → 72,4 %), mejoró
en 4 de 5 folds y redujo su desviación (4,8 → 2,9). La diferencia no fue significativa (p = 0,23).

## 5. Discusión
- **Los modelos nano evaluados son estadísticamente equivalentes** en este conjunto de datos. YOLO12n y YOLOv9t a 1024 px
  obtuvieron las medias más altas en las clases comerciales (≈76 %), pero, con 5 folds y 152 fresas comerciales en prueba,
  las diferencias no superan la variabilidad entre folds tras la corrección por comparaciones múltiples.
- **El mAP no es una métrica adecuada para elegir modelo en esta aplicación:** todos los modelos se ubican entre 86 % y 89 %
  de mAP@0.5, mientras que el acierto comercial varía entre 69 % y 76 %.
- **Una corrida única no basta para comparar modelos.** YOLO26n a 640 px obtuvo 77,0 % de acierto comercial en la fase
  exploratoria y 68,8 ± 4,8 % en la validación cruzada; YOLOv9t a 1024 px, 79,0 % y 75,7 ± 3,0 %, respectivamente.
- **La principal fuente de error son las fronteras entre estados vecinos:** `commercial-basic` con `early-pink` y con
  `commercial-high`, y `commercial-high` con `overripe`. No se observaron confusiones entre fresas comerciales y `unripe`.
- **Meta de la aplicación:** si por oclusión solo es visible ~90 % de los frutos y se requiere clasificar correctamente el
  85 % del total, el sistema necesita ≈94 % de acierto sobre los frutos visibles. Con ≈76 %, el desempeño actual queda por
  debajo de esa meta.

## 6. Limitaciones
- Dataset parcialmente revisado (1564 de 3099 imágenes) y conjunto de prueba pequeño en las clases comerciales
  (51 `commercial-basic` y 101 `commercial-high`).
- Etiquetado realizado por una sola persona, sin medición de su consistencia.
- Una semilla por fold; receta de entrenamiento común, no ajustada por arquitectura.
- Imágenes de un conjunto de datos externo; no se ha evaluado el desempeño en el cultivo de destino (Huaral).
- No se ha medido la velocidad de inferencia en la Raspberry Pi.

## 7. Trabajo pendiente
1. Medir la velocidad de inferencia de los modelos finalistas en la Raspberry Pi 5.
2. Completar la revisión del conjunto de datos (lotes 6–11).
3. Medir la consistencia del etiquetado (re-etiquetado ciego de una muestra de fresas comerciales).
4. Validación cruzada final con el dataset completo y los modelos finalistas (de ser necesario, 3 repeticiones × 5 folds).
5. Conjunto de prueba propio con imágenes del cultivo de Huaral.
6. Evaluar un clasificador de segunda etapa sobre el recorte de cada fruto y la combinación de varias vistas del mismo fruto
   durante el recorrido del robot.
