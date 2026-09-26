# Dataset de fresas: revisión en lotes

## Rápido: subir las primeras 300 de `dataset_5estados`

En tu PC (PowerShell o CMD), dentro de este repositorio clonado:

```bat
pip install pillow
python revisar_fresas.py "D:\新草莓\DatasetId_360753_1652783343" --carpeta dataset_5estados --solo 1
git add revision
git commit -m "Lote 1 (300 imágenes) para revisión"
git push
```

Después activa GitHub Pages (Settings → Pages → la rama donde subiste el lote, carpeta `/root`) y abre desde el celular:
`https://jairocustodio-jc.github.io/DatasetFresas/revision/lote_01/index.html`

Antes de cada lote nuevo corre `git pull`. Para los siguientes lotes usa `--solo 2`, `--solo 3`, …, y repite el `git add`, `commit` y `push`.

Si cambia el script, `python revisar_fresas.py --rehacer-html` actualiza la página de los lotes ya generados sin volver a procesar las fotos.

`revisar_fresas.py` identifica la carpeta del dataset que se arregló y la divide en lotes de 300 imágenes. Cada lote trae una página web para marcar cada foto como ✅ bien o ❌ mal desde el celular.

## 1. Encontrar la carpeta arreglada

```bat
pip install pillow
python revisar_fresas.py "D:\新草莓\DatasetId_360753_1652783343"
```

El script muestra cada subcarpeta con su cantidad de imágenes, de anotaciones (`.json` y `.txt`) y la fecha de su última modificación. La arreglada suele ser la más reciente y la que tiene unas 3000 imágenes con sus anotaciones.

## 2. Armar los lotes

```bat
python revisar_fresas.py "D:\新草莓\DatasetId_360753_1652783343" --carpeta NOMBRE_SUBCARPETA
```

El script crea `revision\` con `lote_01` … `lote_10` (300 fotos cada uno, reducidas a 1280 px para que pesen poco) y un `index.html` general.

## 3. Revisar desde el celular

- Toca una foto para abrirla grande y usa los botones ✅, ❌ y ◀ ▶ para avanzar.
- Las anotaciones (cajas o polígonos de LabelMe `.json` o YOLO `.txt`) se dibujan en amarillo. El botón «Anotaciones» las oculta o las muestra.
- Las marcas se guardan en el navegador del celular, así que puedes cerrar la página y seguir después.
- Al terminar cada lote, usa «Descargar CSV» para obtener `resultado_lote_XX.csv` con el estado de cada foto: `ok`, `mal` o `sin_revisar`.

Para abrirlo en el celular, copia la carpeta `revision` a este repositorio (o a Google Drive o a la memoria del teléfono) y abre `index.html`. Con GitHub Pages activado queda un enlace que funciona desde cualquier lugar.
