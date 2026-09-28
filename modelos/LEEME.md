# Modelo YOLO entrenado con la revisión humana del lote 1

- `yolo_fresas_lote01.pt`: YOLOv8n entrenado con las 256 fotos del lote 1 revisadas por la persona (las etiquetas finales tras
  sus correcciones), 60 épocas, imgsz 640, en CPU. Validado en 40 fotos del lote 1 que no vio: mAP50 ≈ 0,87.
  Tono de color sin aumentar (`hsv_h=0`), porque el color es justamente lo que define la clase.
- Como asistente de revisión, con confianza ≥ 0,6: el 50 % de sus sugerencias de estado coincide con la corrección humana, y detecta
  6 de cada 11 errores de la etiqueta automática. **No se aplica solo**: se muestra como sugerencia en la página.
- Lote 2: 110 sugerencias de estado + 22 cajas faltantes; tras revisión visual se quitaron 7 (fresas con rubor rosado
  marcadas como `unripe`) y 7 faltantes que eran flores u hojas. Quedan 103 + 15 en 94 fotos.
- `scripts/`: preparar el dataset desde la revisión (`preparar.py`), entrenar (`entrenar.py`), validar contra la revisión
  (`validar.py`) y escribir las sugerencias en un lote (`inyectar.py modelo lote umbral_estado umbral_faltantes`).
  Son scripts de trabajo con rutas del entorno donde se entrenó; hay que ajustar las rutas para usarlos en otra máquina.
