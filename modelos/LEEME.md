# Modelo YOLO entrenado con la revisión humana del lote 1

- `yolo_fresas_lote01.pt`: YOLOv8n entrenado con las 256 fotos del lote 1 revisadas por la persona (las etiquetas finales tras
  sus correcciones), 60 épocas, imgsz 640, en CPU. Validado en 40 fotos del lote 1 que no vio: mAP50 ≈ 0,87.
  Tono de color sin aumentar (`hsv_h=0`), porque el color es justamente lo que define la clase.
- Como asistente de revisión, con confianza ≥ 0,6: el 50 % de sus sugerencias de estado coincide con la corrección humana, y detecta
  6 de cada 11 errores de la etiqueta automática. **No se aplica solo**: se muestra como sugerencia en la página.
- Lote 2: 110 sugerencias de estado + 22 cajas faltantes; tras revisión visual se quitaron 7 (fresas con rubor rosado
  marcadas como `unripe`) y 7 faltantes que eran flores u hojas. Quedan 103 + 15 en 94 fotos.
- Lote 3 (mismo modelo): 114 sugerencias de estado + 23 faltantes; tras revisión visual se quitaron 12 sugerencias
  (7 `early-pink` con rubor marcadas como `unripe`, 5 `commercial-high` anaranjadas o con zonas pálidas marcadas como `overripe`)
  y 10 faltantes (flores, hojas, centros de flor). Quedan 102 + 13.
- Lote 4 (mismo modelo): 111 + 29; tras revisión visual se quitaron 24 sugerencias (16 `early-pink` con rubor marcadas como
  `unripe`, 8 `commercial-high` anaranjadas marcadas como `overripe`) y 17 faltantes (flores, hojas, piedras). Quedan 87 + 12.
  Patrón: YOLO confunde `early-pink` con rubor leve y `unripe`; esas sugerencias casi siempre están mal.
- La página avisa además «¿duplicada?» cuando dos cajas se superponen ≥ 80 % (38 casos en los lotes 1–10).
- `yolo_fresas_lote01_02.pt`: reentrenado desde el anterior con los lotes 1 y 2 revisados (513 fotos de entrenamiento,
  30 épocas). Con la revisión del lote 2: se aceptaron 54 de 102 sugerencias y 10 de 15 faltantes. En validación (80 fotos) rinde
  parecido al anterior (45 % vs 42 % de sugerencias correctas), pero **aprendió el criterio de la persona**: en el lote 4 bajó
  de 21 a 8 las sugerencias `early-pink → unripe` (su error típico) y subió de 18 a 28 `commercial-basic → early-pink`
  (la corrección humana más común). Con él se regeneraron las sugerencias de los lotes 3 y 4, otra vez filtradas a mano.
- `yolo_fresas_lote01_03.pt`: reentrenado con los lotes 1–3 (773 fotos). En 120 fotos de validación: 56 % de sugerencias
  correctas (antes 49 %) y menos falsas alarmas (16 contra 23). Revisión del lote 3 (modelo anterior): 57 % de sugerencias
  aceptadas y 12 de 17 faltantes agregadas. Lote 5 generado con este modelo y filtrado a mano.
- `scripts/`: preparar el dataset desde la revisión (`preparar.py`), entrenar (`entrenar.py`), validar contra la revisión
  (`validar.py`) y escribir las sugerencias en un lote (`inyectar.py modelo lote umbral_estado umbral_faltantes`).
  Son scripts de trabajo con rutas del entorno donde se entrenó; hay que ajustar las rutas para usarlos en otra máquina.
