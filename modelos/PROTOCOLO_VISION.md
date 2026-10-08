# Protocolo de evaluación (visión por computadora de la tesis)

Reglas para comparar modelos y sacar conclusiones. Cualquiera (persona o asistente) que trabaje en esta parte las sigue y
revisa el registro de abajo antes de afirmar algo.

## Reglas
1. **Igualdad de condiciones.** Solo se comparan modelos con la misma resolución, la misma partición y la misma receta.
   Lo que no cumpla eso se reporta aparte, marcado como referencia.
2. **Exploración ≠ confirmación.** Una corrida única (benchmark) sirve para elegir candidatos. Las conclusiones salen de la
   validación cruzada de 5 folds (mismos folds: `benchmark/cv5_6lotes/particion.json`, huella `42a74935bb54`).
3. **No descartar con evidencia débil.** Un modelo o una variante (resolución, aumento) no se descarta por una corrida única
   ni por diferencias menores que la variación entre folds. Para descartar: validación cruzada completa, en igualdad de
   condiciones y con corrección por comparaciones múltiples (Holm).
4. **Reportar todos los folds**, no solo media ± desviación, para que se vean los casos atípicos.
5. **Decir lo que no se sabe.** Hoy: una semilla por fold; receta fija (AdamW, lr 0,001) no ajustada por arquitectura;
   test de 235 fotos y 152 fresas comerciales; un solo etiquetador.
6. **Métrica principal:** acierto comercial y error crítico (fresas commercial-basic/high, predicción más segura por fresa).
   mAP se reporta, pero no decide entre modelos.

## Registro de decisiones tomadas con evidencia débil (pendientes de confirmar)
| Decisión | Se basó en | Estado |
|---|---|---|
| YOLO26n evaluado a 640 px en la validación cruzada, no a 1024 | Una corrida: 77,0 % (640) contra 75,0 % (1024) | Pendiente: CV de `yolo26n-1024` |
| «1024 px no mejora de forma consistente» | Corridas únicas por modelo | Sin confirmar; en CV los mejores son a 1024 |
| «YOLO26n descartado por precisión» | CV a 640 contra rivales a 1024; sin corrección múltiple | Retirada: comparación no equivalente |
| Aumento de tono (hsv_h 0,015) sin efecto propio | Un modelo, una corrida (p = 0,11 en el aporte del tono) | Sin confirmar |
| Grandes (x/e) no superan a los nano | Una corrida por modelo, a 640 px | Indicio; sin CV |

## Resultados confirmados (validación cruzada, 5 folds, test fijo)
| Modelo | Acierto com. | Error crít. | mAP@0.5 |
|---|---|---|---|
| YOLO12n · 1024 | 76,2 ± 2,7 | 9,2 ± 3,4 | 87,5 ± 0,6 |
| YOLOv9t · 1024 | 75,7 ± 3,0 | 10,9 ± 2,9 | 88,3 ± 0,9 |
| YOLO11n · 1024 | 72,7 ± 3,5 | 13,4 ± 2,9 | 87,9 ± 1,5 |
| YOLOv10n · 1024 | 71,3 ± 3,9 | 13,8 ± 4,7 | 86,5 ± 1,1 |
| YOLO26n · 640 (referencia, otra resolución) | 68,8 ± 4,8 | 15,5 ± 4,8 | 87,5 ± 1,4 |

YOLO12n-1024 y YOLOv9t-1024 están empatados (sin diferencia significativa). Pendiente: la corrección de Holm sobre todas
las comparaciones antes de afirmar otras diferencias.
