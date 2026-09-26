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

## Corregir cajas desde el celular

1. Abre una foto y toca ✎ (arriba a la derecha) para entrar en modo edición.
2. Toca una caja para seleccionarla: aparecen 4 círculos en sus esquinas. Arrastra una esquina para cambiar el tamaño o arrastra el centro para mover la caja.
3. Los estados de abajo cambian la etiqueta de la caja seleccionada. ＋ agrega una caja, ⌫ borra la seleccionada y ↺ vuelve a las cajas originales de esa foto.
4. El botón 1×/2×/3× hace zoom. Con zoom, arrastrar sobre una zona vacía mueve la imagen.
5. Al terminar el lote, abre el menú **⋯** y pulsa **Exportar correcciones**. Se descarga o comparte `correcciones_lote_XX.json`; pásalo a la PC por WhatsApp, correo o Drive.
6. En la PC:
   ```powershell
   python revisar_fresas.py $raiz --carpeta dataset_5estados --aplicar correcciones_lote_01.json
   ```
   El script escribe las cajas nuevas en los `.json` (LabelMe) o `.txt` (YOLO) originales y guarda una copia `.bak` del archivo anterior.

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

- Pulsa **Empezar** (o **Continuar**) para abrir la primera foto sin revisar. Marca **Bien** o **Mal** y pasa sola a la siguiente; ‹ › avanzan sin marcar.
- Toca la foto para ocultar o mostrar las cajas.
- En la cuadrícula, el punto verde o rojo indica el estado y ✎ indica que la foto tiene cajas editadas.
- El menú **⋯** tiene la leyenda de estados, la descarga del CSV (`resultado_lote_XX.csv`), la exportación de correcciones y la navegación entre lotes.
- Las marcas se guardan en el navegador del celular.

Para abrirlo en el celular, copia la carpeta `revision` a este repositorio (o a Google Drive o a la memoria del teléfono) y abre `index.html`. Con GitHub Pages activado queda un enlace que funciona desde cualquier lugar.
