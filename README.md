# Dataset de fresas: revisión desde el celular

**Página:** https://jairocustodio-jc.github.io/DatasetFresas/revision/

`revisar_fresas.py` divide `dataset_5estados` (3099 fotos) en lotes de 300 con una página
web para revisar cada foto (**Bien / Mal / Evaluar**) y corregir sus cajas desde el celular.
Lo revisado vuelve a la PC con **Enviar** y `--aplicar`.

## En la PC (PowerShell, en `D:\DatasetFresas`)

Primero, siempre:

```powershell
cd D:\DatasetFresas
git pull --no-edit
$raiz = (Get-ChildItem "D:\*\DatasetId_360753_1652783343").FullName
```

| Qué | Comando |
|---|---|
| Ver las carpetas del dataset | `python revisar_fresas.py $raiz` |
| Generar el lote N (1…11) | `python revisar_fresas.py $raiz --carpeta dataset_5estados --solo N` |
| Generar lotes por prioridad | `python revisar_fresas.py $raiz --carpeta dataset_5estados --lista "$raiz\informe\cambios.csv"` |
| Subir a la página | `git add -A` · `git commit -m "Lote N"` · `git push` |
| Aplicar lo enviado desde el celular | `python revisar_fresas.py $raiz --carpeta dataset_5estados --aplicar revision_lote_01.json` |
| Actualizar las páginas tras cambiar el script | `python revisar_fresas.py --rehacer-html` |

- **Fotos:** solo se toman las de la raíz de la carpeta. La copia de `yolo/images/` se ignora.
- **`--lista`:** acepta un `.txt` con un nombre por línea o un `.csv` con la columna `imagen`, como `cambios.csv` o `revisar.csv` del informe. Los lotes toman el nombre del archivo (`cambios_01`, …); `--prefijo` lo cambia.
- **`--aplicar`:**
  - crea `resultado_<lote>.csv` con el estado de cada foto;
  - en el `.json` solo escribe lo que se cambió en el celular: la etiqueta si se cambió, la posición si se movió;
  - conserva `difficult`, `flags`, `attributes` y `score`, y solo vacía `attributes` de las cajas movidas o nuevas;
  - regenera `yolo/labels/<foto>.txt` de las fotos tocadas;
  - escribe `mapa_indices_<lote>.json` con las cajas movidas, nuevas o borradas, para recalcular sus métricas;
  - guarda un `.bak` de cada archivo que reescribe.

## En el celular

- **Empezar / Continuar** abre la primera foto pendiente. **Mal · Evaluar · Bien** la marca y pasa a la siguiente.
- **Zoom:** pellizcar o doble toque. Con zoom, un dedo mueve la foto. Sin zoom, deslizar ← → cambia de foto. Un toque oculta o muestra las cajas.
- **✎ Editar cajas:**
  - tocar una caja la selecciona;
  - arrastrar una esquina cambia el tamaño; arrastrar el centro la mueve;
  - las pastillas de colores cambian su estado;
  - **＋** crea una caja nueva, **⌫** borra la seleccionada y **↺** vuelve a las cajas originales.
- **Menú ⋯:** leyenda, filtro **Mostrar** (Todas / A evaluar / Mal / Bien / Sin revisar), Enviar, CSV y cambio de lote.
- **Guardado:** todo se guarda al instante **en ese navegador de ese celular** («✓ Guardado»). Arriba del lote se ve cuántas fotos faltan enviar.
- **Enviar** genera `revision_<lote>.json`, que compartes a la PC por WhatsApp, correo o Drive. Conviene enviar seguido: si se borran los datos del navegador, lo no enviado se pierde.

Estados: `unripe` (0), `early-pink` (1), `commercial-basic` (2), `commercial-high` (3), `overripe` (4).
