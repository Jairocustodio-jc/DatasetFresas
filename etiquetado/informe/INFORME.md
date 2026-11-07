# Corrección del etiquetado en 5 estados de madurez

**Dataset:** `DatasetId_360753_1652783343` — 3 099 imágenes (756 × 1008), 16 085 bounding boxes.
**Punto de partida:** `data_anylabeling_2/` (la carpeta con la que estabas trabajando en AnyLabeling).
**Resultado:** `dataset_5estados/` — mismas cajas, mismas coordenadas, **solo cambian etiquetas**,
más un juego de **atributos por caja** (oclusión, superficie visible, tamaño, nitidez, daño…).

---

## 1. Los 5 estados

| id | etiqueta (la que va en el JSON) | criterio: % de la superficie visible del fruto con color rosa/rojo |
|----|--------------------------------|--------------------------------------------------------------------|
| 0 | `verde_inmadura` | 0 % (verde, blanco o crema, sin pigmento) |
| 1 | `rosa_temprano` | > 0 % y < 50 % |
| 2 | `comercial_basico` | 50 % – 75 % |
| 3 | `comercial_alto` | 75 % – 90 % |
| 4 | `madurez_plena` | ≥ 90 % **y rojo intenso y uniforme** |

Las etiquetas están en español desde el 3-11-2026 (sección 10.1). Los nombres
anteriores eran `unripe`, `early-pink`, `commercial-basic`, `commercial-high` y
`overripe`; `cambios.csv` conserva esa nomenclatura porque documenta lo ocurrido
antes del renombrado.

El denominador es **la superficie del fruto que se ve**: no cuentan las hojas, el
cáliz, el pedúnculo ni los brillos especulares que lo tapan, ni los frutos vecinos
que caen dentro del bbox.

---

## 2. Por qué el resultado anterior no servía

El script previo (`relabel_chromatic_5stages.py`) medía el porcentaje de rojo sobre
una máscara `S > 38`. Eso falla en tres sitios a la vez:

1. **El denominador estaba mal.** `S > 38` incluye hojas y excluye las zonas
   blancas/cremas del propio fruto, así que el porcentaje no era "% de la superficie
   del fruto".
2. **Contaba frutos vecinos.** Cualquier fresa roja que asomara dentro del bbox
   sumaba al porcentaje: de ahí los cientos de frutos verdes etiquetados
   `early-pink` (es justo lo que estabas corrigiendo a mano: 194 de tus 397
   correcciones manuales eran `early-pink → unripe`).
3. **No distinguía crema de rosa pálido**, con lo que el límite 0/1 —el más
   frecuente del dataset— quedaba al azar.

---

## 3. Método

### 3.1 Medición por fruto

1. **Recorte** con 4 px de margen.
2. **Segmentación** con GrabCut inicializado por máscara (borde del recorte = fondo
   seguro, elipse inscrita = fruto probable, núcleo = fruto seguro), componente conexa
   central y relleno de agujeros. 16 084 de 16 085 cajas se segmentan con GrabCut.
3. **Se descuenta lo que no es superficie de fruto**: hoja/cáliz/pedúnculo (verde con
   tono H ≥ 38; medido sobre el propio dataset, el fruto verde vive en H 28-35 y el
   follaje por encima de 38), brillos especulares (V ≥ 244, S ≤ 34) y sombra profunda
   (V < 80).
4. **Pigmento por histéresis** sobre a\* de CIE Lab suavizado: semillas de rojo franco
   (a\* ≥ 150) que crecen por conectividad hasta a\* ≥ 134, más manchas de rubor pálido
   (a\* ≥ 139) que ocupen al menos el 1,5 % de la superficie.
5. **Filtro anti-fruto-vecino**: si una mancha pequeña de color continúa fuera de la
   máscara del fruto y la parte exterior es mayor que la interior, el color es de otra
   fresa y se descarta.
6. **Métricas de intensidad y sanidad**: % de rojo intenso (a\* ≥ 168), % de rojo
   granate (a\* ≥ 162 y L ≤ 118), % de color pálido, dispersión del a\*, manchas
   oscuras no rojas (candidatas a daño) y señales de fiabilidad (oclusión, superficie
   visible, reflejo, sombra, nitidez, tamaño, color de fruto vecino).

**Reproducibilidad:** GrabCut inicializa sus GMM con k-means++ y el RNG global de
OpenCV. Sin semilla fija, repetir la medición cambiaba el resultado en un 6 % de las
cajas (hasta 4 puntos de porcentaje). Con `cv2.setRNGSeed` fijo la medición es idéntica
entre ejecuciones (verificado: 0 diferencias en 333 cajas medidas dos veces).

### 3.2 Calibración

Etiqueté a ojo 120 frutos tomados al azar en 6 estratos de porcentaje, sin ver el valor
medido. La medición automática y las etiquetas que ya tenías acertaban prácticamente lo
mismo (≈ 68 % de coincidencia exacta con mi lectura visual). Conclusión: **re-etiquetar
todo con el script no habría mejorado nada**. Por eso la medición se usa solo para
*detectar sospechosos* y la decisión final la tomé mirando los recortes.

### 3.3 Auditoría visual (lo que decide las etiquetas)

| grupo | qué es | cuántos | cómo se revisó |
|-------|--------|---------|----------------|
| A | etiquetadas con color pero con 0 % medido | 1 174 | 34 hojas de contacto, una a una |
| C+B | frontera entre estados o contradicción grande | 371 | 19 hojas, asignando estado 0-4 |
| D | seguían como `early-pink` con < 12 % medido | 351 | 15 hojas, "¿hay rosa o no?" |
| R | casos dudosos | 25 | caja dibujada sobre la imagen completa |
| E | etiquetadas `overripe` pero de color pálido | 11 | 1 hoja |
| G | `commercial-high` con rojo granate | 93 | 5 hojas |
| F | manchas oscuras (posible daño) | 83 | 4 hojas, con la zona detectada resaltada |
| H | fronterizos nuevos tras la remedición | 11 | 1 hoja |

**2 119 recortes revisados uno a uno.** Cuando la lectura visual y la medición no
coinciden, manda la lectura visual; cuando ninguna es concluyente, se deja la etiqueta
como estaba y el caso queda en `revisar.csv`.

---

## 4. Criterios usados además del color

| criterio | qué aporta | cómo se aplicó |
|---|---|---|
| **Intensidad del rojo** (a\*, L\*) | separa "cubierto de naranja pálido" de "rojo intenso" | los 11 frutos pálidos etiquetados `overripe` bajaron a `commercial-high`; 15 frutos granates uniformes subieron de `commercial-high` a `overripe` |
| **Uniformidad del color** | un fruto con rojo granate uniforme es sobremaduro; con hombro crema visible, no | se exigió ausencia de zona pálida > 10 % para el estado 4 |
| **Sanidad (daño/pudrición)** | un fruto podrido puede tener 95 % de rojo y no ser comercial | atributo `dano`, solo en los **10 casos confirmados a ojo** |
| **Oclusión y superficie visible** | si se ve media fresa, cualquier porcentaje es poco fiable | atributos `oclusion_pct`, `superficie_visible_pct` y bandera `ocluido` |
| **Tamaño de la caja** | por debajo de ~32 px no se distingue rosa pálido de crema | atributo `lado_px` y bandera `muy_pequeno` |
| **Nitidez y reflejo** | el desenfoque y el brillo especular falsean el color | atributos `nitidez`, banderas `borroso` y `reflejo_fuerte` |
| **Fruto vecino dentro de la caja** | es la causa principal de los falsos `early-pink` | bandera `fruto_vecino_en_caja` |

Cada shape del JSON lleva ahora:

```json
"difficult": false,
"flags": {"ocluido": false, "borroso": false, "reflejo_fuerte": false,
          "fruto_vecino_en_caja": false, "muy_pequeno": false, "dano": false,
          "dificil": false},
"attributes": {"color_pct": 82.2, "rojo_intenso_pct": 40.2, "rojo_granate_pct": 0.0,
               "color_palido_pct": 21.5, "superficie_visible_pct": 90,
               "oclusion_pct": 10, "lado_px": 119, "nitidez": 535.0}
```

Banderas activas sobre las 16 085 cajas:

| bandera | cajas | % |
|---|---|---|
| `dificil` (cualquier motivo, = `difficult`) | 3 815 | 23,7 % |
| `ocluido` | 3 123 | 19,4 % |
| `muy_pequeno` (< 32 px) | 1 082 | 6,7 % |
| `reflejo_fuerte` | 778 | 4,8 % |
| `fruto_vecino_en_caja` | 87 | 0,5 % |
| `borroso` | 51 | 0,3 % |
| `dano` (confirmado a ojo) | 10 | 0,1 % |

Con esto puedes entrenar con todo y reportar aparte la métrica sobre el subconjunto
"fácil", que es lo que se suele pedir en una tesis.

---

## 5. Resultado

**1 445 etiquetas corregidas de 16 085 (9,0 %).** Las 14 640 restantes se dejaron
intactas. Ninguna coordenada se tocó (verificado caja por caja).

| antes → después | nº |
|---|---|
| early-pink → unripe | 1 155 |
| commercial-basic → early-pink | 58 |
| commercial-high → overripe | 47 |
| overripe → commercial-high | 44 |
| early-pink → commercial-basic | 37 |
| commercial-high → commercial-basic | 32 |
| commercial-basic → overripe | 22 |
| commercial-basic → commercial-high | 16 |
| commercial-basic → unripe | 8 |
| commercial-high → unripe | 6 |
| early-pink → overripe | 5 |
| early-pink → commercial-high | 4 |
| overripe → unripe | 4 |
| overripe → early-pink | 3 |
| unripe → overripe | 2 |
| overripe → commercial-basic | 1 |
| commercial-high → early-pink | 1 |

Distribución por estado:

| estado | antes | después |
|--------|-------|---------|
| 0 unripe | 9 483 | 10 654 |
| 1 early-pink | 1 955 | 816 |
| 2 commercial-basic | 1 089 | 1 055 |
| 3 commercial-high | 1 344 | 1 322 |
| 4 overripe | 2 214 | 2 238 |

El grueso del cambio es el estado 1: casi dos tercios de los `early-pink` eran frutos
completamente verdes o blancos.

> Estas cifras son las de esta pasada automática + auditoría. El estado **final** del
> dataset, después de la revisión en el celular, está en la sección 10.

---

## 6. Qué hay en cada carpeta

```
dataset_5estados/
├── <n>.json            JSON AnyLabeling corregido, con flags y attributes
├── <n>.png             imagen
└── yolo/
    ├── images/         copia de las imágenes
    ├── labels/         <n>.txt formato YOLO (clase cx cy w h normalizados)
    └── classes.txt     unripe, early-pink, commercial-basic, commercial-high, overripe

informe/
├── INFORME.md          este documento
├── cambios.csv         las 1 445 correcciones, con motivo y % medido
├── atributos.csv       los atributos de las 16 085 cajas (para filtrar al entrenar)
├── revisar.csv         14 casos para mirar a mano (casi todos son cajas mal puestas)
├── revisar/            los recortes de esos 14 casos
└── muestras_estado_*.png   24 frutos de cada estado en el dataset final

_work/                  código y datos intermedios
├── pigment.py          análisis cromático por fruto (semilla fija, reproducible)
├── extract.py          extracción masiva (8 procesos, ~10 min)
├── candidates.py       selección de sospechosos
├── refine_sets.py      conjuntos de afinado por intensidad y sanidad
├── apply.py            aplica las decisiones y escribe el dataset
├── verify.py           comprueba que solo cambiaron etiquetas
├── features.csv        36 métricas de las 16 085 cajas
├── audit/              todas las hojas de contacto de la auditoría
└── audit_*_result.txt  mis decisiones visuales, caja por caja
```

Las carpetas anteriores (`data_anylabeling_2`, `dataset_relabeled`,
`dataset_relabeled_v2`, `dataset_anylabeling`, `Images_AnyLabeling_5cls`) **no se han
tocado**.

---

## 7. Los 14 casos pendientes

Están en `informe/revisar.csv` con su recorte al lado. No son dudas de color: son
**cajas mal puestas** en el etiquetado original (una caja sobre un palo, otra sobre
suelo desnudo, otra a caballo entre dos frutos, franjas de 19 px…). Lo razonable es
borrarlas o reajustarlas en AnyLabeling; por eso las dejé con su etiqueta original en
vez de inventar un estado. Ejemplos: `1081.png` tiene tres cajas sin fruto dentro;
`945.png#1` encierra una hoja seca; `544.png#12` está a caballo entre una fresa roja y
una verde.

---

## 8. Límites conocidos

- **Daño y pudrición**: el detector automático de manchas oscuras tiene mucha sombra
  lateral como falso positivo (de 83 candidatos grandes revisados, solo 6 eran daño
  real) y apenas detecta daño sobre fruto rojo. Por eso el atributo `dano` solo marca
  los 10 casos confirmados a ojo: **no es un censo completo de fruta dañada**. Para eso
  haría falta anotación manual específica o un modelo entrenado para ello.
- **Deformidad y albinismo** no se anotaron: requieren criterio manual.
- Quedan ~1 900 cajas cuya medición cae a menos de 6 puntos del corte de su clase. Se
  dejaron como estaban porque dentro de ese margen ni la medición ni la etiqueta
  original son más fiables que la otra. Auditarlas visualmente serían unas 95 hojas más.
- El porcentaje se juzga sobre **la cara visible** del fruto; en una sola foto nunca se
  ve el otro lado. Conviene decirlo explícitamente en la memoria de la tesis.
- Si en algún punto afirmas que los estados equivalen a madurez comestible o comercial,
  el color no lo sostiene por sí solo: con 30-50 frutos por estado medidos con
  refractómetro (°Brix), penetrómetro (firmeza) y colorímetro (L\*a\*b\*) tendrías la
  validación que suele pedir un jurado.

## 9. Cómo repetir el proceso

```bash
python _work/extract.py       # mide las cajas -> _work/features.csv
python _work/candidates.py    # marca los sospechosos -> _work/candidatos.csv
python _work/refine_sets.py   # conjuntos de intensidad y sanidad
python _work/make_audit.py    # genera las hojas de contacto para revisar
python _work/apply.py         # reconstruye dataset_5estados/ desde data_anylabeling_2
python _work/verify.py        # comprueba integridad
```

> ⚠️ **`apply.py` ya no debe ejecutarse.** Reconstruye el dataset desde
> `data_anylabeling_2/` y borraría las correcciones hechas desde el celular
> (sección 10). A partir de aquí, cualquier ajuste se hace **in situ** sobre
> `dataset_5estados/`, como hace `_work/cierre.py`.

---

## 10. Estado final: revisión en el celular y cierre técnico

Tras la pasada automática, el dataset se revisó foto a foto desde el celular con la
herramienta de `Jairocustodio-jc/DatasetFresas` (11 lotes, 3 099 imágenes).

**Aporte de esa revisión** (171 imágenes con `.bak`, es decir con cajas o etiquetas
tocadas a mano):

- **184 cambios de etiqueta**, sobre todo en las clases con color: `commercial-basic →
  early-pink` (51), `overripe → commercial-high` (33), `commercial-high → overripe`
  (25), `commercial-basic → commercial-high` (25).
- **4 cajas nuevas** (1 004 → 1 008 en esos archivos).
- Los metadatos por caja sobrevivieron: 964 de 1 008 conservaron sus `attributes`; los
  44 que los perdieron eran justamente cajas movidas o creadas, cuyas métricas ya no
  eran válidas.

**Cierre técnico** (`_work/cierre.py`, trabaja in situ):

1. Métricas recalculadas para las **1 008 cajas** de los 171 archivos tocados, de modo
   que `attributes`, `flags` y `difficult` corresponden a la geometría actual.
2. **114 cajas recortadas al lienzo**: se salían de la imagen por menos de 1 px (el peor
   caso, 0,92 px), defecto heredado de la anotación original.
3. Export YOLO verificado **caja por caja** contra los JSON y regenerado donde hacía
   falta.
4. `informe/atributos.csv` reescrito con las 16 089 cajas del estado final.
5. Verificación de integridad: **0 problemas** (etiquetas válidas, geometría dentro de
   la imagen, JSON y YOLO coincidentes).

**Distribución final:**

| estado | cajas | % |
|---|---:|---:|
| 0 `unripe` | 10 670 | 66,3 % |
| 1 `early-pink` | 862 | 5,4 % |
| 2 `commercial-basic` | 998 | 6,2 % |
| 3 `commercial-high` | 1 328 | 8,3 % |
| 4 `overripe` | 2 231 | 13,9 % |
| **total** | **16 089** | |

Cajas marcadas `difficult`: **3 824 (23,8 %)**. Todas las cajas tienen métricas.

### 10.1 Etiquetas en español y el estado 4

Las cinco etiquetas pasaron al español el 3-11-2026, **sin cambiar ninguna
clasificación**: solo el texto del campo `label`, `yolo/classes.txt` y la columna
`etiqueta` de los CSV. Los índices YOLO 0-4 son los mismos, así que cualquier modelo
o anotación que use números no se ve afectado.

| id | antes | ahora |
|----|-------|-------|
| 0 | `unripe` | `verde_inmadura` |
| 1 | `early-pink` | `rosa_temprano` |
| 2 | `commercial-basic` | `comercial_basico` |
| 3 | `commercial-high` | `comercial_alto` |
| 4 | `overripe` | `madurez_plena` |

El cambio del estado 4 no es cosmético. Tanto NY/T 2787-2015 como USDA reservan
"sobremadura" para una **condición** del fruto —blando, de color apagado, oscurecido—
que no puede juzgarse en una fotografía, mientras que lo que esta clase mide es
**cobertura de color ≥ 90 %**, que en ambas normas corresponde a la **madurez plena**
(*fully-ripe*). Mantener el nombre anterior habría obligado a defender en la memoria
una equivalencia que las normas no respaldan.

El corte entre `comercial_basico` y `comercial_alto` se mantiene en **75 %**, el de
USDA No. 1, y no en el 80 % de NY/T 2787-2015. La razón es metodológica: las etiquetas
finales no salen de un umbral automático sino de revisión visual, de modo que
recortar la franja 75-80 % de forma mecánica sustituiría ese criterio por la medición
automática justo en la zona más dudosa. Hay 362 cajas medidas en esa franja; si en
algún momento se quiere alinear con NY/T, el camino correcto es revisarlas a ojo, no
aplicar un filtro.
