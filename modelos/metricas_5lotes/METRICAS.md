# Métricas del modelo actual (YOLOv8n, `yolo_fresas_5lotes.pt`)

Modelo YOLOv8n reentrenado en Kaggle (GPU T4) con los lotes 1–5 revisados a mano: 1485 fotos, 1285 de train y 200 de
validación (40 por lote). Paró en la época 25 por parada temprana y se guardó la mejor, la 15. Imagen de 640 px.

Se evalúa en dos conjuntos:
- **Validación (200 fotos, 1000 cajas):** se usó para elegir la mejor época. Se reconstruyó con el estado exacto de los
  datos de la corrida, y reproduce las métricas que reportó Kaggle.
- **Test, lote 6 (76 fotos, 388 cajas):** revisado a mano después de entrenar; el modelo nunca vio estas fotos. Es la
  medida honesta.

## Métricas globales

| Conjunto | Fotos | Precisión | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Validación | 200 | 0,778 | 0,835 | 0,880 | 0,743 |
| **Test (lote 6)** | 76 | **0,778** | **0,850** | **0,856** | **0,726** |

En fotos nuevas el rendimiento casi no baja (mAP50 0,88 → 0,86): el modelo generaliza a lotes que no vio.

## Por clase (test, lote 6)

| Clase | Precisión | Recall | AP50 | AP50-95 |
|---|---|---|---|---|
| unripe | 0,88 | 0,95 | 0,96 | 0,75 |
| early-pink | 0,82 | 0,83 | 0,89 | 0,77 |
| **commercial-basic** | **0,45** | 0,67 | **0,53** | 0,45 |
| commercial-high | 0,88 | 0,85 | 0,92 | 0,82 |
| overripe | 0,86 | 0,96 | 0,99 | 0,84 |

En validación (200 fotos), commercial-basic y commercial-high son también las dos clases más débiles (AP50 0,76 y 0,78;
el resto ≥ 0,93).

## Clases comerciales: a qué se predicen (% de las cajas reales)

| Conjunto | Clase real | Cajas | ✓ Bien | → early-pink | → overripe | → la otra comercial |
|---|---|---|---|---|---|---|
| Validación | commercial-basic | 40 | 52,5 | **20,0** | 0 | 27,5 |
| Validación | commercial-high | 73 | 68,5 | 1,4 | **20,5** | 8,2 |
| Test (lote 6) | commercial-basic | 15 | 60,0 | **20,0** | 0 | 20,0 |
| Test (lote 6) | commercial-high | 35 | 82,9 | 0 | **5,7** | 11,4 |

| Conjunto | Acierto comercial | Error crítico (→ early-pink u overripe) |
|---|---|---|
| Validación | 62,8 % | 21,2 % |
| Test (lote 6) | 76,0 % | 10,0 % |

Matriz de confusión: con confianza ≥ 0,25 e IoU ≥ 0,45. Los errores se concentran en las fronteras vecinas:
commercial-basic ↔ early-pink y commercial-high ↔ overripe. Nunca se confunde una fresa comercial con unripe. Los mismos
pares son los que más corrige la persona al revisar.

## Etiquetado asistido (human-in-the-loop)

| Lote | Modelo que sugirió | Sugerencias de estado aceptadas por la persona |
|---|---|---|
| 2 | lote 1 | 53 % |
| 3 | lotes 1–2 | 57 % |
| 6 (76 fotos revisadas) | lotes 1–5 | **84 %** (16 de 19) |

Al reentrenar con más lotes revisados, las sugerencias del modelo coinciden cada vez más con el criterio humano. En el
lote 6, el 84 % se aceptó, sobre todo commercial-basic → early-pink (8 de 9) y commercial-basic → commercial-high (5 de 6).

## Velocidad
41–44 ms por foto en CPU (Xeon, sin GPU, 640 px). En una Raspberry Pi falta medirla.

## Cautela al presentar
- El test es pequeño: 76 fotos, con solo 15 cajas commercial-basic y 35 commercial-high. Un acierto más o menos mueve
  varios puntos. El benchmark de esta noche usa un test estratificado más grande (235 fotos, unas 150 cajas comerciales).
- Las etiquetas son de una sola persona, así que no se ha medido el acuerdo entre etiquetadores.

Archivos: `confusion_*.png` (matrices normalizadas), `curva_PR_*.png` y `metricas.json` (todos los números).
