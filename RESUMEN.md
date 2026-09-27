# Revisión del dataset de fresas: resumen del trabajo

**Link para revisar desde el celular:** https://jairocustodio-jc.github.io/DatasetFresas/revision/

**Repositorio:** `Jairocustodio-jc/DatasetFresas`, rama `claude/jolly-heisenberg-qa6cj9`

---

## 1. El objetivo

Revisar desde el celular (por ejemplo, en el carro) el dataset de fresas guardado en la PC, en
`D:\新草莓\DatasetId_360753_1652783343\dataset_5estados`:

- marcar cada foto como **Bien**, **Mal** o **Evaluar**;
- corregir las cajas (bounding boxes) cuando estén mal;
- trabajar en lotes de **300 imágenes**.

## 2. El dataset

| Dato | Valor |
|---|---|
| Carpeta | `dataset_5estados` |
| Imágenes | **6198** (PNG, 1008 × 756 px) |
| Lotes | **21**: del 1 al 20 con 300 fotos cada uno y el 21 con 198 |
| Anotaciones | LabelMe (`.json`), cajas rectangulares |
| Estados (clases) | `unripe`, `early-pink`, `commercial-basic`, `commercial-high`, `overripe` |

Al principio se esperaban 3000 imágenes, pero hay 6198. Se decidió revisar todas.

## 3. Cómo funciona

```
PC (D:\...)                         GitHub                          Celular
───────────                         ──────                          ───────
revisar_fresas.py --solo N  ──push──►  revision/lote_NN/  ──Pages──►  página web
                                                                     │  marcar / editar
                                                                     │  (se guarda en el celular)
revisar_fresas.py --aplicar  ◄──── revision_lote_NN.json ◄── Enviar ─┘
(CSV + anotaciones corregidas)       (WhatsApp / correo / Drive)
```

1. En la PC, el script copia 300 fotos reducidas a la carpeta `revision/lote_NN/` y genera la página.
2. Con `git push` se suben a GitHub, y GitHub Pages las publica en el link.
3. En el celular se revisa. Todo queda guardado en el navegador del celular.
4. Con **Enviar** se genera un archivo, que se pasa a la PC.
5. En la PC, `--aplicar` crea la tabla de resultados y escribe las cajas corregidas en las anotaciones originales.

## 4. Lo que hace la página

### Índice de lotes
- Lista de lotes con una barra de avance (verde para Bien, amarillo para Evaluar, rojo para Mal).
- Aviso **«sin enviar»** en los lotes con cambios que aún no llegaron a la PC.

### Cuadrícula del lote
- Miniaturas limpias, 3 por fila.
- Punto de color según el estado; **✎** si la foto tiene cajas editadas.
- Línea de estado: *«✓ 13 guardadas · 13 sin enviar»* y el botón **Enviar**.
- Botón **Empezar / Continuar**, que lleva a la primera foto sin revisar.
- Menú **⋯**:
  - leyenda de estados;
  - cajas en las miniaturas (se activan o desactivan);
  - filtro **Mostrar**: Todas / A evaluar / Mal / Bien / Sin revisar;
  - Enviar;
  - descargar el CSV;
  - cambiar de lote.

### Visor de foto
- Pantalla completa, con las cajas de colores y el nombre del estado encima.
- Botones **Mal · Evaluar · Bien**. Al marcar, pasa sola a la siguiente foto y muestra el aviso *«✓ Guardado: Bien»*.
- Etiqueta arriba con el estado de la foto (*Sin marcar*, Bien, Mal o Evaluar, más ✎ si tiene cajas editadas).
- **Zoom táctil:**

| Gesto | Acción |
|---|---|
| Pellizcar con dos dedos | Zoom hasta 8×, centrado donde están los dedos |
| Doble toque | Acerca 2.5× en ese punto; otro doble toque vuelve a la foto completa |
| Un dedo con zoom | Mueve la foto, sin que se salga de la pantalla |
| Deslizar ← → sin zoom | Foto siguiente o anterior |
| Un toque | Oculta o muestra las cajas |

### Edición de cajas (botón ✎)
- Tocar una caja la selecciona y muestra sus 4 esquinas.
- Arrastrar una esquina cambia el tamaño; arrastrar el centro mueve la caja.
- Las pastillas de colores cambian el estado de la caja seleccionada.
- **＋** agrega una caja nueva, **⌫** borra la seleccionada y **↺** vuelve a las cajas originales.
- La edición funciona con zoom, y las esquinas y etiquetas siempre se ven nítidas.

## 5. Comandos (PowerShell, en `D:\DatasetFresas`)

Primero se ubica la carpeta del dataset. Se hace así para no tener que escribir los caracteres chinos:
```powershell
cd D:\DatasetFresas
git pull --no-edit
$raiz = (Get-ChildItem "D:\*\DatasetId_360753_1652783343").FullName
```

| Qué | Comando |
|---|---|
| Ver las carpetas del dataset | `python revisar_fresas.py $raiz` |
| Generar el lote N | `python revisar_fresas.py $raiz --carpeta dataset_5estados --solo N` |
| Subir el lote | `git add -A` → `git commit -m "Lote N"` → `git push` |
| Aplicar lo revisado en el celular | `python revisar_fresas.py $raiz --carpeta dataset_5estados --aplicar revision_lote_NN.json` |
| Actualizar las páginas tras un cambio del script | `python revisar_fresas.py --rehacer-html` |

Qué hace `--aplicar`:
- crea `resultado_lote_NN.csv` con las columnas `archivo, estado (bien/mal/evaluar), editada`;
- reescribe las cajas corregidas en el `.json` (LabelMe) o el `.txt` (YOLO) original;
- guarda una copia **`.bak`** de cada archivo que modifica.

## 6. Problemas resueltos en el camino

| Problema | Solución |
|---|---|
| Claude trabaja en la nube y no puede leer el disco `D:\` | El script se ejecuta en la PC y las fotos se suben a GitHub |
| `fatal: not a git repository` | Clonar primero el repo en `D:\DatasetFresas` |
| PowerShell rompía `新草莓` y lo cambiaba por `???` | Buscar la ruta con `Get-ChildItem "D:\*\DatasetId_..."` |
| `Author identity unknown` | `git config --global user.email / user.name` |
| `HTTP 408` al subir (internet lento) | `git config --global http.postBuffer 524288000` |
| `rejected (fetch first)` | Hacer `git pull --no-edit` antes de cada lote |
| No se veía el estado de cada caja | Colores y nombre por estado |
| Página amontonada | Rediseño minimalista con menú ⋯ |
| Zoom poco práctico | Gestos táctiles: pellizcar, doble toque, arrastrar, deslizar |
| No quedaba claro qué se había guardado | Avisos «✓ Guardado», contador «sin enviar» y botón Enviar |

## 7. Estado actual

- ✅ Lotes **1, 2 y 3** publicados (900 de 6198 fotos).
- ✅ Página con revisión, filtro, edición de cajas, zoom y envío.
- ⏳ Faltan los lotes **4 al 21**.
- ⏳ El lote 1 está en revisión (13 o más fotos marcadas en el celular).

## 8. Límites que conviene conocer

- **Lo revisado vive en el navegador del celular** hasta que pulsas **Enviar**. Si borras los datos del navegador, cambias de teléfono o usas otro navegador, no aparece. En iPhone, Safari puede borrar datos de páginas que no abres en unos 7 días. Envía cada cierto tiempo.
- Las fotos originales miden 1008 × 756 px, así que a partir de unos 3× de zoom se ven pixeladas.
- Si el repositorio es privado, GitHub Pages puede requerir un plan pago.
- `--aplicar` escribe todas las cajas de una foto como rectángulos.
- La edición en la página y `--aplicar` se probaron en un navegador simulando un celular y con archivos de prueba. Todavía no se han probado con dedos reales ni con las anotaciones reales. Conviene revisar 1 o 2 fotos después del primer `--aplicar`.

## 9. Archivos del repositorio

| Archivo | Qué es |
|---|---|
| `revisar_fresas.py` | Script: lista las carpetas, genera los lotes y las páginas, aplica las correcciones |
| `README.md` | Instrucciones de uso |
| `RESUMEN.md` | Este documento |
| `revision/index.html` | Índice de lotes |
| `revision/lote_NN/` | Página del lote (`index.html`) y fotos reducidas (`img/`) |
| `.nojekyll` | Hace que GitHub Pages publique los archivos tal cual |
