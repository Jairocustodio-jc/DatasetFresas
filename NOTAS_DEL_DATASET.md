# Notas para el agente que mantiene `revisar_fresas.py`

Escrito por el agente que trabaja en la PC sobre el dataset de origen
(`D:\新草莓\DatasetId_360753_1652783343`). Aquí va lo que cambió en
`dataset_5estados` y tres cosas del script de revisión que conviene ajustar antes de
seguir generando lotes.

---

## 1. Qué es ahora `dataset_5estados`

Se construye desde `data_anylabeling_2/` (la carpeta que el usuario editaba a mano) y
contiene **las mismas 16 085 cajas, con las mismas coordenadas**. Lo único que se
cambió fueron etiquetas: **1 445 correcciones (9,0 %)**, decididas revisando 2 119
recortes uno a uno, no con un umbral automático.

| estado | id YOLO | cajas |
|---|---|---|
| `unripe` | 0 | 10 654 |
| `early-pink` | 1 | 816 |
| `commercial-basic` | 2 | 1 055 |
| `commercial-high` | 3 | 1 322 |
| `overripe` | 4 | 2 238 |

Criterio: % de la superficie **visible** del fruto con color rosa/rojo
(0 / <50 / 50-75 / 75-90 / ≥90), y para el límite 3↔4 también intensidad y
uniformidad del rojo (un fruto cubierto de naranja pálido es 3, no 4).

Cada `shape` del JSON lleva ahora metadatos nuevos:

```json
{
  "label": "commercial-high",
  "score": null,
  "points": [[x1, y1], [x2, y2]],
  "group_id": null,
  "description": "",
  "difficult": false,
  "shape_type": "rectangle",
  "flags": {"ocluido": false, "borroso": false, "reflejo_fuerte": false,
            "fruto_vecino_en_caja": false, "muy_pequeno": false,
            "dano": false, "dificil": false},
  "attributes": {"color_pct": 82.2, "rojo_intenso_pct": 40.2, "rojo_granate_pct": 0.0,
                 "color_palido_pct": 21.5, "superficie_visible_pct": 90,
                 "oclusion_pct": 10, "lado_px": 119, "nitidez": 535.0},
  "kie_linking": []
}
```

`difficult` está en **3 815 cajas (23,7 %)**: oclusión > 25 %, lado < 32 px, desenfoque
o reflejo fuerte. No se borró ninguna caja; es solo una marca para poder filtrar al
entrenar. El 93 % de esas cajas son de la clase `unripe`.

Informes útiles en `D:\新草莓\DatasetId_360753_1652783343\informe\`:

| archivo | contenido |
|---|---|
| `cambios.csv` | las 1 445 correcciones (`imagen`, `bbox`, `antes`, `despues`, motivo) |
| `atributos.csv` | las 16 085 cajas con sus métricas y banderas |
| `revisar.csv` | 14 cajas que sé que están mal puestas (geometría, no color) |
| `INFORME.md` | método completo |

---

## 2. Tres cosas del script que conviene arreglar

### 2.1 Cada foto se está listando dos veces (6 198 en vez de 3 099)

`armar_lotes` usa `origen.rglob("*")`, y `dataset_5estados` contiene:

```
dataset_5estados/
├── 10.png, 10.json, …        3 099 imágenes + 3 099 JSON  ← las buenas
└── yolo/
    ├── images/   3 099 PNG   ← LA MISMA FOTO, copiada para el export YOLO
    ├── labels/   3 099 TXT
    └── classes.txt
```

Por eso salen 6 198 imágenes y 21 lotes: **son 3 099 fotos duplicadas**. Además, la
copia de `yolo/images/` entra por la rama `t.exists() and not j.exists()` de
`aplicar_correcciones`, así que la misma foto se revisaría dos veces y se escribiría en
dos sitios distintos (JSON y TXT), que luego no coinciden.

Arreglo mínimo:

```python
imgs = sorted(p for p in origen.rglob("*") if es_imagen(p) and "yolo" not in p.parts)
```

o, mejor, no recursivo para esta carpeta:

```python
imgs = sorted(p for p in origen.iterdir() if es_imagen(p))
```

Con eso quedan **3 099 fotos → 11 lotes** (10 de 300 + 1 de 99), no 21.

### 2.2 `--aplicar` borra los metadatos de cada caja

En la rama JSON, `aplicar_correcciones` reconstruye la lista entera:

```python
d["shapes"] = [{"label": label, "points": [...], "group_id": None,
                "description": "", "shape_type": "rectangle", "flags": {}} ...]
```

Eso pierde, en cada caja tocada: `difficult`, `attributes`, el contenido de `flags`,
`score` y `kie_linking`. Si se aplica sobre las 3 099 imágenes, desaparece el trabajo de
atributos (3 815 `difficult`, oclusión, tamaño, nitidez, daño…).

Sugerencia: emparejar por posición con las cajas existentes y tocar solo `label` y
`points`:

```python
previas = d.get("shapes", [])
nuevas = []
for k, (label, x1, y1, x2, y2) in enumerate(cajas):
    base = dict(previas[k]) if k < len(previas) else {
        "score": None, "group_id": None, "description": "", "difficult": False,
        "shape_type": "rectangle", "flags": {}, "attributes": {}, "kie_linking": []}
    base["label"] = label
    base["points"] = [[x1 * w, y1 * h], [x2 * w, y2 * h]]
    base["shape_type"] = "rectangle"
    if k >= len(previas):           # caja nueva: sus métricas ya no valen
        base["attributes"] = {}
    nuevas.append(base)
d["shapes"] = nuevas
```

Si el editor del celular puede reordenar o borrar cajas, mejor mandar desde la página el
índice original de cada caja (`idx`) junto con `pts`, y emparejar por ese índice; con
emparejar por posición basta mientras no se reordenen.

Detalle menor: al crear un JSON desde cero se escribe `"version": "5.0.1"`, pero los del
dataset son `4.0.0-beta.7` (AnyLabeling). Conviene conservar la versión del archivo que
ya existe (el código ya lo hace) y usar `4.0.0-beta.7` para los nuevos.

### 2.3 JSON y `yolo/labels` se van a desincronizar

`dataset_5estados/yolo/labels/*.txt` es un **export** generado desde los JSON. Si el
celular corrige una caja, se escribe en el JSON y el TXT queda con el valor viejo.
Lo más simple: revisar solo la raíz (punto 2.1) y, cuando termine una tanda de
`--aplicar`, regenerar el export. Es un bucle corto:

```python
# por cada <n>.json -> yolo/labels/<n>.txt
cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
w_, h_ = (x2 - x1) / W, (y2 - y1) / H
f"{CLASES.index(label)} {cx:.6f} {cy:.6f} {w_:.6f} {h_:.6f}"
```

(CLASES = el orden de `yolo/classes.txt`: unripe, early-pink, commercial-basic,
commercial-high, overripe.)

---

## 3. Los lotes 1-3 ya publicados están un poco desfasados

Se generaron hoy a las 09:31; el dataset se reconstruyó después (18:43) con el afinado
por intensidad del rojo. Comparando las 900 fotos publicadas con los JSON actuales hay
**12 fotos con etiquetas distintas**:

- `commercial-high → overripe`: 7
- `overripe → commercial-high`: 4
- `early-pink → commercial-basic`: 1

Las cajas (coordenadas) son idénticas, así que no es grave, pero si se regeneran esos
tres lotes el celular mostrará el estado correcto. Lo que el usuario ya haya marcado
sigue siendo válido: el desfase afecta a 12 de 900 fotos.

---

## 4. Sugerencia: revisar por prioridad, no en orden de archivo

Revisar 3 099 fotos en orden alfabético reparte el esfuerzo por igual entre lo que ya
está bien y lo que no. Con los CSV se puede ordenar por valor:

1. **`cambios.csv` (1 445 cajas)** — lo que yo cambié. Si algo está mal, está aquí.
   Son ~1 100 fotos distintas.
2. **`revisar.csv` (14 cajas)** — cajas mal puestas (sobre un palo, sobre suelo, a
   caballo entre dos frutos). Estas sí necesitan el editor de cajas: hay que borrarlas
   o reajustarlas.
3. **Frontera de clase** — en `atributos.csv`, las cajas con `color_pct` a menos de 6
   puntos de 50, 75 o 90 (~1 900). Son las dudosas que dejé como estaban.
4. El resto.

Bastaría con aceptar una lista de nombres de imagen (`--lista archivo.txt`) al armar los
lotes para poder generar "lote de cambios", "lote de frontera", etc.

---

## 5. Contrato de datos, para no pisarnos

- Las coordenadas de las cajas del dataset **no las toco yo**: si el usuario las corrige
  desde el celular, esa es la versión buena.
- Los campos `attributes`, `flags` y `difficult` los genero yo desde la medición
  (`_work/apply.py`). Si una caja se mueve o se crea desde el celular, sus métricas
  dejan de ser válidas: lo correcto es vaciar `attributes` en esa caja y avisarme para
  recalcularlas (vuelvo a correr `extract.py` + `apply.py`).
- El `label` puede cambiarlo cualquiera de los dos; lo que mande es la revisión humana.
- Yo no borro cajas; si el celular borra alguna, se pierde la correspondencia por
  posición con `atributos.csv` (que usa `imagen` + índice de bbox). Si vais a permitir
  borrados, conviene que `--aplicar` escriba también un `mapa_indices.json` con la
  correspondencia vieja → nueva, y lo recalculo sin problema.
