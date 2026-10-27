# Detector de dos etapas y mapas de calor con cobertura: decisiones de diseño

*27/10/2026. Resumen operativo. El informe completo, con fuentes y nivel de evidencia de cada afirmación, está en
`reports/Dos etapas y cobertura en fresa.md`; las notas por tema, en `research_notes/Dos etapas y cobertura en fresa/`.*

## Lo que dice la evidencia (en una línea cada punto)
- **Nadie ha comparado en fresa** YOLO de 5 clases contra YOLO + clasificador de recortes con los mismos datos. En
  tomate (3 clases) dio empate. Fuera de fruta, recortar ayuda cuando las clases son finas, ordenadas o hay cambio de
  dominio; con pocas clases separables por color, no. Nuestro caso (5 estados ordenados, dos comerciales adyacentes,
  despliegue en otro campo) está en el régimen donde sí se reportan ganancias, pero la magnitud es desconocida.
- **El estado intermedio es la clase más débil en todos los detectores de una etapa** (7–20 puntos de AP por debajo de
  «ripe»; 19–22 % de los intermedios se van a un vecino). Es la motivación citable.
- **Diseño del clasificador** (evidencia indirecta, de otros dominios): margen 10–20 % por lado con caja expandida a
  cuadrado; 224 px con `Resize`, no `CenterCrop`; aumentos que respeten el tono (`hsv_h` ≤ 0,015, mejor 0; nada de
  grayscale ni channel shuffle); entropía cruzada como base y etiquetas suaves ordinales como variante; frutos con lado
  < 40 px marcados «indeterminado».
- **Cobertura:** se mide sobre el clasificador (Grad-CAM = HiResCAM en la última convolución, target en el logit de la
  clase), no sobre el detector; cuatro números dentro de la máscara/elipse del fruto: cobertura@0,5, entropía normalizada,
  Gini y *relevance mass accuracy*; con controles (mapa uniforme y aleatorio) y chequeo de fidelidad (ROAD, aleatorización
  de pesos). No existe métrica canónica de «cobertura»: se construye con métricas publicadas (Quantus) y se dice así.
- **Raspberry Pi 5 (CPU, NCNN FP32, 640 px):** detector nano 67–94 ms; clasificador 7–12 ms por recorte (≈2 ms INT8).
  Sin torch en producción (1,2 GB contra ≈475 MB). A 1024 px no hay cifra publicada (≈170–240 ms extrapolado).
- **Estadística:** con 152 frutos comerciales, McNemar solo detecta diferencias de ≈9–10 puntos (potencia 80 %); una
  diferencia de 5 puntos tiene potencia ≈0,3. Métrica primaria preregistrada: error crítico; McNemar mid-p + IC bootstrap
  pareado (remuestreo por foto); media ± DE de los folds como estabilidad, sin p sobre 5 folds; tabla en tres niveles
  (cajas reales = cota superior, cajas detectadas = extremo a extremo, atribución del error).

## Lo que se implementa (`kaggle_dos_etapas.py` y `kaggle_cobertura.py`)
1. **Recortes** de las cajas reales con margen del 15 % (caja a cuadrado), 224 px, por fold de la CV (mismos folds y test
   fijo de `cv5_<origen>`), formato ImageFolder; sobremuestreo ×2 de las comerciales como en el detector.
2. **Clasificadores:** `yolo11n-cls`, `yolo11s-cls` (exportables a NCNN) y `mobilenet_v3_small` (torchvision) como
   contraste; entropía cruzada y variante con etiquetas suaves ordinales (σ = 0,75). Se eligen por acierto comercial en la
   validación del fold. Ablación de margen {0, 0,15, 0,30} en validación.
3. **Tubería y comparación:** detector de la CV (5 clases) → clasificador sobre cada caja detectada; métrica igual que
   ahora (predicción más segura por fresa real, IoU ≥ 0,5). Tres filas por modelo: YOLO solo · YOLO + clasificador ·
   clasificador sobre cajas reales (cota superior). McNemar mid-p e IC bootstrap entre YOLO solo y YOLO + clasificador.
   Detector de 1 clase («fresa»): segunda fase, si la primera muestra ganancia.
4. **Mapas de calor:** Grad-CAM del clasificador y EigenCAM P3 del detector sobre las mismas fresas que
   `atencion_comerciales.png`; tabla de cobertura en aciertos y en errores (Mann-Whitney, delta de Cliff), con la
   advertencia de que es exploratorio.
5. **Criterio de éxito fijado antes de correr:** la segunda etapa se adopta si reduce el error crítico con IC pareado
   que excluya el cero y sin bajar el acierto comercial, y si su costo medido en la Pi entra en el presupuesto.
