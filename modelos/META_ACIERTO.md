# De dónde sale la meta de acierto

Nota escrita el 7-11-2026. Recoge la derivación que respalda las tres últimas
diapositivas de `presentacion/PFC2_presentacion_previa_7min_resultados.pptx`.
La fuente de los supuestos es `modelos/PROTOCOLO_VISION.md`.

## 1. La meta no es el acierto por fresa, es el error del conteo

El requisito operativo es estimar las fresas comerciales por surco con un error
relativo de ± 10 % al 95 % de confianza, que es la precisión que exige planificar
la cosecha. La estimación corregida es N = C / k, donde C es el conteo del
detector y k el factor de calibración medido contra conteo manual.

El error relativo de N tiene dos componentes que se suman en cuadratura, la
incertidumbre de la calibración y la variación del conteo entre unidades, y esta
última decrece como 1/√n:

    (σN/N)² = (σk/k)² + (σC/C)²        σC/C ∝ 1/√n
    z · σ = 10 %   ⇒   σ ≤ 10 / 1,96 = 5,1 %

Medido en el test, unidades de 40 fotos con unas 26 fresas comerciales dan
σ = 10 % tras calibrar. Por tanto n = 26 · (10 / 5,1)² ≈ 100 fresas comerciales
por unidad, es decir unas 160 fotos o varias vistas por planta. La verificación
en campo es una prueba de equivalencia TOST con margen ± 10 %.

## 2. La meta traducida a métrica del detector

El sesgo del conteo es exactamente el cociente entre recall y precisión de la
clase comercial, porque las fresas contadas son TP/P y las reales son TP/R:

    k = R / P

De ahí sale la meta en métrica, sin ninguna constante arbitraria:

| Situación | Valores | k |
|---|---|---|
| Hoy, fold 1 de la validación cruzada | R = 85 %, P = 70 % | 1,22 |
| Meta de ± 10 % sin calibrar | R = 85 %, **P ≥ 78 %** | ≤ 1,10 |
| Si se quisiera ± 5 % | R = 85 %, P ≥ 81 % | ≤ 1,05 |

Faltan ocho puntos de precisión, y se ganan quitando detecciones de más, no
detectando más fruta. Calibrando el sesgo, en cambio, la precisión actual ya
cumple siempre que la unidad reúna unas 100 fresas comerciales.

Advertencia sobre estos números. El recall del 85 % es el promedio de las cinco
clases de la validación cruzada y no el de la clase comercial sola, de modo que
el 70 % de precisión es una deducción a partir de k y no una medida directa.
Además k = 1,22 viene del fold 1. Cuando terminen los cinco folds, ambos números
se recalculan y la meta se ajusta sola, porque la relación k = R/P no cambia.

## 3. La meta si se midiera fruto a fruto

Si la meta se expresara como clasificar bien el 85 % de los frutos del surco,
primero hay que descontar la oclusión:

    accuracy total = visibilidad × accuracy sobre lo visible
    Meta:   0,85 = 0,90 × 0,944
    Hoy:    0,90 × 0,757 = 0,68 del surco

Es decir, con el 90 % de los frutos visibles la accuracy sobre lo visible debe
llegar a 94,4 %, y hoy es 76 %. Viendo cada fruto varias veces, el voto
mayoritario sube a 85,2 % con tres vistas y 90,4 % con cinco bajo independencia,
todavía por debajo de 94,4 %. Esta es la razón de fondo para preferir la meta de
conteo calibrado sobre la meta por fruto.

## Referencias

G. Forman, «Quantifying counts and costs via classification», Data Min. Knowl.
Discov. 17(2), 2008. P. González et al., «A review on quantification learning»,
ACM Comput. Surv. 50(5), 2017.
