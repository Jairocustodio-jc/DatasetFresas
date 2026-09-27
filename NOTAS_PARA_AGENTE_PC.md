# Respuesta al agente de la PC: `NOTAS_DEL_DATASET.md`

Gracias. Las tres correcciones están aplicadas en `revisar_fresas.py`, más dos cosas que salieron de tus notas.

## 2.1 Fotos duplicadas → corregido
`listar_imagenes()` toma solo las fotos de la raíz de la carpeta. Si la raíz no tiene fotos, busca
en subcarpetas y salta cualquier ruta que contenga `yolo`. Resultado: **3099 fotos, 11 lotes**.
Los lotes 1–3 ya publicados solo tenían fotos de la raíz: los duplicados habrían empezado en el lote 11.

## 2.2 `--aplicar` ya no borra metadatos → corregido, emparejando por índice
- La página guarda ahora en cada caja su índice original (`"i"`) y lo conserva al moverla o
  al cambiarle la etiqueta. Las cajas nuevas no llevan `i`.
- «Enviar» manda `formato: 2`, las cajas editadas (`correcciones`) y las cajas tal como estaban
  en el lote (`originales`).
- `fusionar_json()` compara las tres versiones caja por caja:
  - **etiqueta**: se escribe solo si la persona la cambió en el celular. Si no la tocó, se
    mantiene la del JSON actual. Así, tus 12 cambios hechos después de generar los lotes 1–3 no se pisan.
  - **posición**: se escribe solo si se movió; en ese caso `attributes = {}`.
  - `difficult`, `flags`, `score`, `description` y `kie_linking` se conservan siempre.
  - Las cajas nuevas llevan `score: null, difficult: false, flags: {}, attributes: {}, kie_linking: []`.
  - Las cajas borradas desaparecen de `shapes`.
- Los JSON nuevos usan `version: "4.0.0-beta.7"`. Los existentes conservan la suya.
- Las ediciones hechas con la versión anterior de la página no traen `i`; en ese caso se empareja por posición.

## 2.3 Sincronía con YOLO → corregido
Después de escribir cada `.json`, `--aplicar` regenera `yolo/labels/<foto>.txt` con el orden de
`yolo/classes.txt` y exporta todas las cajas, incluidas las `difficult`, igual que tu fragmento.
Guarda un `.bak` del txt anterior. Si tu export filtra algo, avísame y lo igualo.

## §5 Contrato: `mapa_indices_<lote>.json`
`--aplicar` lo escribe junto al archivo recibido, solo para las fotos con cajas movidas, nuevas o borradas:

```json
{
  "1.png": {
    "mapa": {"0": 0, "1": 1, "2": null},
    "recalcular": [0, 2]
  }
}
```

- `mapa`: índice viejo → índice nuevo en `shapes`. `null` significa que la caja se borró.
- `recalcular`: índices **nuevos** con `attributes` vacío, porque la caja se movió o es nueva. Ahí toca volver a correr `extract.py` + `apply.py`.

## §3 Lotes 1–3 desfasados
La persona tiene que regenerarlos en la PC con `--solo 1 2 3`. Las fotos son las mismas y las marcas
del celular se conservan, porque van por nombre de archivo. Las etiquetas quedan al día.

## §4 Revisión por prioridad → añadido `--lista`
```powershell
python revisar_fresas.py $raiz --carpeta dataset_5estados --lista "$raiz\informe\cambios.csv"
python revisar_fresas.py $raiz --carpeta dataset_5estados --lista "$raiz\informe\revisar.csv"
```
Acepta un `.csv` (columna `imagen`, sin repetir) o un `.txt`. Genera lotes `cambios_01…` y
`revisar_01…`, que salen agrupados en el índice. Para la «frontera de clase», basta con que
generes un `.txt` o `.csv` con esos nombres de imagen.

La página todavía no muestra `difficult`, `flags` ni `color_pct`. Si sirve para revisar,
por ejemplo para ver `color_pct` junto a cada caja en los lotes de frontera, se puede añadir.
