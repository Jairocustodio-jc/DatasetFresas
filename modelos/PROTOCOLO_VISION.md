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
7. **Los hiperparámetros se eligen con la validación, nunca con el test.** `kaggle_benchmark.py --hiper` entrena cada
   variante de la receta (una cosa a la vez respecto de «base») en los mismos folds de la CV y la compara contra base por
   el acierto comercial en la validación de cada fold (t pareada + Holm). Una variante solo se adopta si gana con
   p_Holm < 0,05 y en la mayoría de los folds; si ninguna lo hace, se mantiene la receta base. El test fijo se reporta de
   referencia y no participa en la elección. Hasta hoy no se ha hecho: la receta actual se fijó a mano.

## Metas del sistema y su justificación (borrador para el asesor, 27/10/2026)
El acierto del sistema se descompone en dos factores que se justifican y se miden por separado:

**acierto del sistema = recall en campo (fresas que el robot llega a ver y detectar) × acierto sobre las fresas visibles.**

1. **Acierto sobre las fresas visibles: se justifica con las normas de calidad.** Las normas de comercialización fijan
   cuánto error tolera un lote, y eso se traduce en una precisión mínima sobre lo que el robot cosecha:
   - UNECE FFV-35 (fresas; adoptada por la UE en el Reglamento 543/2011, anexo I parte B.7): Categoría «Extra» tolera
     **5 %** por número o peso de fresas que no cumplen la categoría; Categorías I y II, **10 %**, con no más de 2 % que no
     cumpla los requisitos mínimos o con pudrición. Tamaño mínimo 25 mm (Extra) y 18 mm (I y II).
   - USDA U.S. No. 1 (7 CFR 51.3115 y ss.): **10 %** por conteo de fresas que no cumplen el grado, con sublímites para
     daño serio y 2 % para pudrición.
   - Perú: **NTP 011.011:2019 «Fresas. Requisitos»** (INACAL; reemplaza a la NTP 011.011:1975). Categorías Extra, Primera
     y Segunda. Tolerancias de calidad: **pendiente de consultar el texto en INACAL** (no está disponible en línea); un
     artículo que la cita reproduce diámetros mínimos de 38 / 26 / 20 mm y 10 % de tolerancia de tamaño.
   Meta derivada: un lote cosechado debe cumplir Categoría I, es decir, **≥ 90 % de las fresas cosechadas realmente
   comerciales** (≥ 95 % para Extra). La confusión commercial-basic ↔ commercial-high no saca al lote de categoría; el
   **error crítico** (early-pink u overripe cosechada) sí. Por eso el error crítico es la métrica principal: hoy ≈ 10 %
   en el test, justo en el límite de Categoría I y lejos del 5 % de Extra.
   Argumento biológico adicional: la fresa es un fruto no climatérico, no madura después de cosechada (Perkins-Veazie,
   *Horticultural Reviews* 17, 1995; manuales de poscosecha de Kader, UC Davis). Una early-pink cosechada es fruta perdida;
   una overripe, merma por vida útil.
2. **Recall en campo: no se puede justificar con literatura; se fija como requisito y se mide en Huaral.** Lo publicado
   para robots de fresa en campo está por debajo o alrededor del 90 % y varía mucho con la oclusión: Xiong et al. 2020
   (J. Field Robotics) 50–97 % de éxito al primer intento según condiciones; Parsa et al. 2024 (J. Field Robotics) 87 % de
   las detectadas cosechadas; He et al. 2025 (Comput. Electron. Agric.) 58 % → 74 % de cosecha al apartar hojas con
   ventilador, detección correcta ≈ 80 %, visible/oculta clasificada ≈ 93 %; detectores en campo con recall 81–90 %
   (RTF-YOLO 2024, DHN-YOLO 2025, graduación en invernadero *Agriculture* 2026: 1964 de 2175 instancias detectadas).
   Un 90 % es alcanzable solo con varios pases o varios ángulos por planta; hay que medirlo contando las fresas que una
   persona ve en la planta contra las que el robot detecta en su trayectoria.
3. **Techo humano (pendiente, aporte propio):** doble etiquetado ciego de ~300 fresas comerciales y kappa ponderado
   cuadrático. No hay kappa publicado para fresa; si dos personas coinciden en el 90 %, pedirle 94 % al modelo no tiene
   sentido y la meta se recalibra.
4. **Costo por tipo de error:** tabla con la consecuencia de cada confusión (fruta perdida, merma, retraso al siguiente
   pase) y, con precios locales, una función de costo que justifique el peso del error crítico frente a basic ↔ high.

Todas las cifras de normas y robots se verificaron solo en extractos y textos derivados (UE, ASEAN, Tanzania para
FFV-35); antes de citarlas en la tesis hay que leer el texto original de cada norma y artículo.

### Justificación estadística del 94 % (muestreo de aceptación)
La meta «detectar 90 de 100 y clasificar bien 85 de 100» equivale a un acierto de 85/90 = 94,4 % sobre las fresas visibles.
No hay literatura que fije ese número, pero sale de la norma con un argumento estadístico estándar (muestreo de aceptación
por atributos, el mismo de ISO 2859 / MIL-STD-105): la norma tolera hasta el 10 % de fresas fuera de categoría en la
**muestra inspeccionada** de un lote. Si el robot comete un error por fresa con probabilidad p, el número de fresas fuera de
categoría en una muestra de n sigue una binomial(n, p), y el lote pasa cuando ese número es ≤ 0,10·n. Para que el lote pase
con probabilidad ≥ 95 % (riesgo del productor ≤ 5 %), p tiene que quedar por debajo del 10 % con margen:

| Fresas inspeccionadas por lote | Error máximo por fresa (lote pasa con 95 %) | Acierto mínimo | Error máximo con 90 % |
|---|---|---|---|
| 50 | 5,4 % | 94,6 % | 6,4 % |
| 100 | 6,3 % | **93,7 %** | 7,1 % |
| 200 | 7,1 % | 92,9 % | 7,8 % |
| 500 | 8,1 % | 91,9 % | 8,5 % |
| Categoría Extra (5 %), 100 fresas | 2,6 % | 97,4 % | — |

Con una muestra de inspección de 100 fresas (el orden de magnitud de una bandeja o caja de mercado), el acierto mínimo es
**≈ 94 %**: coincide con el 94,4 % de la meta. Así el 85/90 deja de ser una cifra elegida y pasa a ser la consecuencia de
la tolerancia de la norma y de un riesgo del productor del 5 %. Dos matices que hay que escribir:
- La tolerancia de la norma cubre **todos** los defectos (tamaño, daño, pudrición, madurez). Si se reserva solo la mitad
  para el error de madurez, el requisito sube a ≈ 97 % (p ≤ 2,6 % con 100 fresas). La versión «todo el presupuesto para la
  madurez» es la menos exigente y es la que da el 94 %.
- La magnitud que entra en la norma es la **precisión de la decisión de cosechar**: de las fresas que el robot decide
  cosechar, cuántas son realmente comerciales. No es el acierto comercial (que mira las fresas comerciales reales). Hoy, en
  el test (YOLOv9t-1024, benchmark 6 lotes): de 200 cajas que el modelo llamó comerciales, 136 eran comerciales, 26 eran
  early-pink u overripe y 38 no tenían caja real (detecciones de más). Entre las fresas reales cosechadas, el error de
  madurez es 26/162 = **16 %**, contra el ≤ 6,3 % que exige el argumento. La brecha es de unos 10 puntos de error, no de
  18 puntos de acierto; y el error crítico (10 %) es la parte de esa brecha que importa a la norma.

### Justificación del 90 % de detección y de los tamaños de muestra
El 90 % de recall no es de calidad sino de **rendimiento**: lo que no se detecta no se cosecha en ese pase. No se justifica
con la norma ni con literatura (los robots publicados están entre 50 y 87 %); se fija como requisito y se mide en Huaral.
Lo que sí da la estadística es cuánto hay que medir:
- Para estimar el recall en campo con ± 5 puntos (IC 95 %) alrededor de 90 %: n ≈ 1,96² · 0,9 · 0,1 / 0,05² ≈ **140 fresas**
  contadas a mano en la planta y comparadas con lo que el robot detecta.
- Para verificar que la precisión de cosecha es ≥ 94 % contra la alternativa de 90 % (una cola, α = 5 %, potencia 80 %):
  n ≈ **360 fresas cosechadas**. Con las 152 fresas comerciales del test actual el IC de la precisión es de ± 6 puntos:
  sirve para decir que hoy no se cumple, no para certificar que se cumple. Por eso la verificación final va con la CV del
  dataset completo (≈ 300 fresas comerciales en test) o con un conjunto de prueba de Huaral.

## Orden de trabajo cuando se termine de etiquetar (lotes 1–11)
1. `--cv 5 --nueva-particion` con los finalistas (YOLO12n, YOLOv9t y YOLO26n a 1024 px; v10n y 11n si hay tiempo):
   partición y test fijo nuevos con todas las fotos; esta es la comparación que va a la tesis. Ningún resultado de la
   partición de 6 lotes se mezcla con esta.
2. Con el modelo que gane (o los dos primeros si empatan con Holm): `--hiper --particion-de 11lotes`, primero
   `--folds 1 2 3` y después el resto. Si una variante gana según la regla 7, se repite la CV del paso 1 con ella.
3. Velocidad en la Raspberry Pi 5 de los finalistas (640 y 1024 px) antes de decidir el modelo final.

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
