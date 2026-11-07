# Arranque para un agente nuevo

Lee esto antes de tocar nada. Resume dónde está cada cosa, cómo trabaja Jairo y
qué trampas han costado tiempo antes. Escrito el 7-11-2026.

## 1. Qué es el proyecto

Detección y clasificación de madurez de fresa en cinco estados, con vista a contar
las fresas comerciales de un surco para planificar la cosecha. Es el PFC de Jairo
Custodio, de UTEC. Hay tres frentes vivos a la vez: el dataset etiquetado, el
entrenamiento de modelos en Kaggle y la presentación de resultados.

## 2. Dónde está cada cosa

En este repo:

- `etiquetado/` — las anotaciones finales, 3 099 JSON con 16 089 cajas, las
  etiquetas YOLO y los 29 scripts del pipeline. Ver `etiquetado/LEEME.md`.
- `modelos/PROTOCOLO_VISION.md` — el protocolo de evaluación y la justificación
  estadística de las metas. Es la fuente de verdad cuando haya dudas sobre qué
  hay que alcanzar y por qué.
- `modelos/META_ACIERTO.md` — la derivación corta de las metas: el ± 10 % al 95 %,
  el k = R/P que da precisión ≥ 78 %, y la meta por fruto de 94,4 %.
- `modelos/kaggle_benchmark.py` — el benchmark. Su docstring explica cómo lanzarlo.
- `revision/lote_01` a `lote_11` — las fotos y las cajas con las que entrena Kaggle.
- `presentacion/` — la presentación previa, con el cálculo de las metas al final.
- Rama `yolo-benchmark` — los resultados de los entrenamientos.

En el disco de Jairo, fuera del repo:

- `D:\新草莓\DatasetId_360753_1652783343` — el dataset con las imágenes a
  resolución completa, 9,8 GB en PNG más 4,9 GB del export YOLO. No caben aquí.
- `D:\DatasetFresas` — este repo clonado.

## 3. Cómo trabajar con Jairo

- **Pregunta antes de empezar algo grande o caro.** Alcance, profundidad y costo
  primero. No lances flujos, agentes ni subagentes sin permiso explícito.
- **Todos los archivos van al disco D, nunca al C.** Carpeta de trabajo
  `D:\Claude_trabajo`, temporales en `D:\Claude_trabajo\temp`.
- **Correcciones quirúrgicas, no re-etiquetar en bloque.** Esta es la regla que
  más le importa. La medición automática sirve para *detectar sospechosos*, no
  para decidir: se revisan los recortes a ojo en hojas de contacto y se cambia
  solo lo que la vista confirma. Lo que no sea concluyente se deja como está y se
  anota en un CSV. Medido contra 120 frutos, el clasificador cromático acertaba
  lo mismo que las etiquetas que ya había, un 68 %, así que re-etiquetar en bloque
  movería miles de cajas sin ganancia.
- **Comprueba antes de entregar.** Si cambias la presentación, renderízala y
  mírala. Varias veces se entregaron diapositivas con texto cortado o superpuesto
  por no abrir el resultado.
- **Si un error tuyo le cuesta tiempo, discúlpate primero** y después explica.

## 4. Trampas que ya han costado tiempo

- **No ejecutes `etiquetado/_work/apply.py`.** Reconstruye desde
  `data_anylabeling_2` y borraría las 184 correcciones de la revisión en celular.
  Los ajustes se hacen in situ con `cierre.py`.
- **GrabCut usa el RNG global de OpenCV.** Sin `cv2.setRNGSeed` fijo, la medición
  cambiaba en torno al 6 % de las cajas entre ejecuciones de un mismo script.
- **Los lotes de `revision/` están congelados el 3-10-2026.** No llevan las 184
  correcciones del celular ni los nombres en español. Kaggle entrena con ellos,
  así que el benchmark no está medido sobre el dataset corregido. Regenerarlos es
  la mejora pendiente más valiosa, pero rompe la comparación con lo ya entrenado:
  no lo hagas sin que Jairo lo pida.
- **En Kaggle, un modelo por sesión.** El script reparte las horas entre los
  trabajos pendientes. Con tres modelos y cinco folds cada entrenamiento recibió
  1 h 24 cuando necesitaba unas 2 h, y se perdió el tiempo de GPU. Con cinco
  trabajos el tope sube a unas 4,3 h.
- **PowerPoint bloquea el .pptx.** Si está abierto, la copia falla con «está
  siendo utilizado en otro proceso». Pídele que lo cierre.
- **En este Windows, LibreOffice no renderiza.** Para exportar diapositivas a PNG,
  PowerPoint vía COM desde PowerShell.

## 5. Qué está pendiente

- La validación cruzada quedó parada: `yolov9t-1024` con los folds 1, 2 y 3
  completos, `yolo12n-1024` con 1 y 2. Faltan los demás.
- La variante `intensidad` de `--hiper`, para ver si el modelo aprovecha la
  intensidad del color, está lista pero sin correr.
- La auditoría de la frontera del 75 %, en `etiquetado/informe/`, tiene 96 de 438
  recortes juzgados. Jairo dijo que la seguiría él.
