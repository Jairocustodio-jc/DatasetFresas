# Feedback del lote 1: revisión humana

Fuente: `revision_lote_01.json`, enviado el 27/09 a las 23:50. Son 296 de 300 fotos marcadas, todas «bien»
después de corregirlas, y 84 fotos con cajas editadas. El lote tiene 1516 cajas.

## Etiquetas que cambiaste (antes → tu corrección)

| Antes (etiqueta automática) | Cajas en el lote | Cambiadas | % | A qué |
|---|---|---|---|---|
| `commercial-basic` | 107 | **44** | **41 %** | → `early-pink` 28 · → `commercial-high` 14 · → `overripe` 2 |
| `commercial-high` | 123 | **29** | **24 %** | → `overripe` 14 · → `commercial-basic` 9 · → `early-pink` 6 |
| `overripe` | 211 | 15 | 7 % | → `commercial-high` 14 · → `commercial-basic` 1 |
| `early-pink` | 68 | 3 | 4 % | → `commercial-basic` 1 · → `commercial-high` 1 · → `unripe` 1 |
| `unripe` | 1007 | 2 | 0,2 % | → `early-pink` 2 |

**Lectura:**
- **`commercial-basic` es la clase menos fiable** de la etiqueta automática: 4 de cada 10 cajas cambiaron.
  La corrección más común es **bajarla a `early-pink`** (28 veces). Para ti, esas fresas tienen menos rojo
  del que midió el agente, o tu límite entre las dos está por encima del 50 %.
- **Entre `commercial-high` y `overripe` hay cambios en los dos sentidos** (14 y 14). No hay un sesgo claro, es
  un límite difuso. El agente usa la intensidad del rojo, pero en estas cajas no coincide con tu criterio.
- `unripe` es casi siempre correcta (99,8 %): no vale la pena revisarla con detalle.

## Cajas
- **5 cajas borradas** (3 `overripe`, 2 `commercial-high`), en `1081.png` (4) y `1176.png` (1). No agregaste ninguna.
- **23 cajas movidas**, casi todas sin cambio de tamaño: **11 eran toques accidentales** (se superponen ≥ 97 %
  con la original) y 12 fueron ajustes reales. La página ya no mueve una caja si solo la tocas, y `--aplicar`
  ignora esos desplazamientos mínimos.

## Problema encontrado al aplicar, y cómo se corrige
Las correcciones se hicieron con una versión anterior de la página (sin índice por caja), y `--aplicar` las
emparejó **por posición**. En `1081.png` y `1176.png` borraste cajas del medio, así que los metadatos
(`difficult`, `flags`) de cajas distintas quedaron cruzados. Ahora el emparejamiento se hace por **superposición
(IoU)**. Para arreglarlo hay que restaurar esas 2 fotos desde su `.bak` y volver a aplicar (ver el README).

## Cómo se usa para el lote 2
`python revisar_fresas.py $raiz --carpeta dataset_5estados --feedback feedback\revision_lote_01.json`
cruza tus etiquetas con `color_pct` y `rojo_intenso_pct` de cada caja y calcula **tus umbrales** entre estados.
Los guarda en `feedback/criterios.json`. Después, al generar el lote 2 (`--solo 2`), las cajas donde tu criterio
no coincide con la etiqueta actual salen **punteadas y con «¿estado?»**, y el filtro **Mostrar → Con sugerencia**
lleva directo a esas fotos.

**Para el agente de la PC:** con `feedback/criterios.json` puede ajustar sus umbrales (sobre todo
`early-pink | commercial-basic`) y volver a etiquetar los lotes que faltan antes de que se revisen.
