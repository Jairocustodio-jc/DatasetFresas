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
| YOLO26n evaluado a 640 px en la validación cruzada, no a 1024 | Una corrida: 77,0 % (640) contra 75,0 % (1024) | Corregido: CV de `yolo26n-1024` hecha (72,4 %); a 1024 rinde +3,5 pts que a 640, sin significancia (p = 0,23) |
| «1024 px no mejora de forma consistente» | Corridas únicas por modelo | Sin confirmar; en CV los mejores son a 1024 |
| «YOLO26n descartado por precisión» | CV a 640 contra rivales a 1024; sin corrección múltiple | Retirada: comparación no equivalente |
| Aumento de tono (hsv_h 0,015) sin efecto propio | Un modelo, una corrida (p = 0,11 en el aporte del tono) | Sin confirmar |
| Grandes (x/e) no superan a los nano | Una corrida por modelo, a 640 px | Indicio; sin CV |

## Resultados de la validación cruzada (5 folds, test fijo, todos a 1024 px salvo indicación)
| Modelo | Acierto com. | Error crít. | mAP@0.5 | Acierto por fold (f1–f5) |
|---|---|---|---|---|
| YOLO12n · 1024 | 76,2 ± 2,7 | 9,2 ± 3,4 | 87,5 ± 0,6 | 79,0 · 75,0 · 78,9 · 73,0 · 75,0 |
| YOLOv9t · 1024 | 75,7 ± 3,0 | 10,9 ± 2,9 | 88,3 ± 0,9 | 75,6 · 78,3 · 73,7 · 71,8 · 79,0 |
| YOLO11n · 1024 | 72,7 ± 3,5 | 13,4 ± 2,9 | 87,9 ± 1,5 | 69,0 · 69,7 · 72,4 · 76,3 · 76,3 |
| YOLO26n · 1024 | 72,4 ± 2,9 | 13,0 ± 3,8 | 87,8 ± 0,6 | 71,1 · 68,4 · 76,3 · 73,7 · 72,3 |
| YOLOv10n · 1024 | 71,3 ± 3,9 | 13,8 ± 4,7 | 86,5 ± 1,1 | 70,4 · 70,4 · 77,0 · 66,4 · 72,4 |
| YOLO26n · 640 (referencia) | 68,8 ± 4,8 | 15,5 ± 4,8 | 87,5 ± 1,4 | 61,8 · 73,7 · 72,4 · 66,5 · 69,7 |

**Con corrección de Holm (10 comparaciones entre los 5 modelos a 1024 px), ninguna diferencia es significativa**, ni en
acierto comercial ni en error crítico. La más fuerte, YOLO12n contra YOLOv10n, pasa de p = 0,017 a p_Holm = 0,17
(error crítico: de 0,008 a 0,08). Conclusión defendible hoy: **YOLO12n y YOLOv9t tienen las medias más altas, pero con 5
folds y este test no se puede afirmar que ninguno de los cinco sea mejor que otro.** La elección final se decide por
velocidad en la Raspberry y se confirma con la validación cruzada del dataset completo (más fotos, y si hace falta 3 × 5 folds).
Resolución en YOLO26n (mismos folds): 1024 px aporta +3,5 pts de acierto comercial (p = 0,23, no significativo).
