# Protocolo estadístico para comparar YOLO 5-clases vs. YOLO + clasificador de recortes, y para comparar "cobertura" de mapas de calor, con un test set fijo pequeño (~235 imágenes, ~152 frutos comerciales) y 5-fold CV

Nota metodológica sobre las fuentes: las páginas primarias (JMLR, BMC, arXiv, PMC) no eran accesibles por fetch directo desde esta sesión (DNS/proxy bloqueado), así que los hallazgos citados provienen de los resúmenes devueltos por el buscador (que incluyen citas textuales) y de documentación de software. Los números de potencia / tamaño muestral marcados como **[cálculo propio]** fueron computados en esta sesión en Python puro (fórmulas de Connor 1987, binomial exacta para McNemar condicional y mid‑p, aproximación de Noether 1987, Monte Carlo para la t pareada). Las fórmulas se dan explícitamente para que el redactor pueda verificarlas.

---

## KQ1. Tests pareados para resultados de clasificación por objeto sobre los mismos objetos de test (McNemar exacto, mid‑p, bootstrap del IC de la diferencia) y su potencia con n≈150

### Takeaway
Con ~150 frutos pareados, McNemar (en cualquiera de sus variantes) sólo detecta con 80 % de potencia diferencias de ~7–12 puntos de accuracy según la tasa de pares discordantes; una diferencia de 5 puntos (75 % vs 80 %) tiene potencia ≈ 0.20–0.49. La variante recomendada es la **mid‑p** (Fagerland et al. 2013); el exacto condicional es conservador y pierde potencia. Debe reportarse siempre un IC de la diferencia (Wald pareado o bootstrap pareado), no sólo el p‑valor.

### Cited Findings
- Fagerland, Lydersen & Laake (2013), *BMC Med Res Methodol* 13:91, compararon error tipo I y potencia del test mid‑p frente al McNemar asintótico (con y sin corrección de continuidad), el exacto condicional y el exacto incondicional. "The mid‑p test performs well compared with the asymptotic, asymptotic with continuity correction, and exact conditional tests, and almost as good as the vastly more complex exact unconditional test." Recomendación: el mid‑p y el exacto incondicional "can be recommended for use in any situation"; el asintótico "if small but frequent violations of the nominal level is acceptable"; "The exact conditional test and the asymptotic test with continuity correction did not perform well for any of the considered scenarios." El mid‑p "did not violate the nominal level in any of the 9595 scenarios". — [Fagerland et al. 2013, BMC MRM](https://www.doi.org/10.1186/1471-2288-13-91); [PDF](https://bmcmedresmethodol.biomedcentral.com/track/pdf/10.1186/1471-2288-13-91)
- Definición mid‑p: "The mid‑p value is the exact p‑value minus half the point probability of the observed statistic"; material suplementario con cálculo en Excel/SAS/SPSS/Stata. — [Supplementary file](https://www.springermedizin.de/resource/supplementary/12874_2013_967_MOESM1_ESM.pdf/9433268)
- Dietterich (1998), *Neural Computation*: revisa cinco tests aproximados; "two common tests, a two‑proportions test and a paired t‑test on random train‑test splits, are shown to have high probability of type I error in certain situations and should never be used"; McNemar mostró bajo error tipo I; "For algorithms that can be executed only once, McNemar's test is the only test with acceptable type I error." — [Dietterich 1998 PDF](https://web.engr.oregonstate.edu/~tgd/publications/dietterich-approximate-statistical-tests-nc1998.pdf); [abstract](https://mlanthology.org/neco/1998/dietterich1998neco-approximate/)
- Implementación Python: `statsmodels.stats.contingency_tables.mcnemar(table, exact=True, correction=True)`; con `exact=True` "the p‑value comes from the binomial distribution" y el estadístico reportado es "the smaller of the two discordant counts"; `exact=False` usa chi‑cuadrado con corrección de continuidad opcional. — [statsmodels docs](https://www.statsmodels.org/v0.10.2/generated/statsmodels.stats.contingency_tables.mcnemar.html)
- En R, `exact2x2::mcnemar.exact` devuelve además un intervalo de confianza para el test exacto pareado. — [exact2x2](https://rdrr.io/cran/exact2x2/man/exact2x2.html)
- Fórmula de tamaño muestral para McNemar (Connor 1987), número de pares: n = ( z₁₋α/₂·√p_d + z₁₋β·√(p_d − (p₂₁−p₁₂)²) )² / (p₂₁−p₁₂)², con p_d = p₁₂+p₂₁ (probabilidad de discordancia); "Its power comes entirely from the discordant pairs"; "more accurate than the simple McNemar formula for small sample sizes". — [MetricGate: sample size McNemar](https://metricgate.com/docs/sample-size-mcnemar-paired); SAS PROC POWER describe METHOD=CONNOR como "an approximation from Connor (1987)". — [SAS POWER](https://support.sas.com/documentation/cdl/en/statug/63347/HTML/default/statug_power_sect009.htm)
- Bootstrap pareado para la diferencia de accuracy: "Each bootstrap sample contains both the gold standard and each team's prediction for the same [item]; if the interval contains zero, it is impossible to rule out that the performance of both algorithms is the same." Percentil: remuestrear el test set con reemplazo al mismo tamaño, calcular la métrica (o diferencia) en cada réplica y tomar los percentiles 2.5 y 97.5. — [arXiv 2403.04693](https://arxiv.org/pdf/2403.04693); [arXiv 2305.10452](https://arxiv.org/pdf/2305.10452); [KDnuggets bootstrap CI](https://www.kdnuggets.com/2021/10/calculate-confidence-intervals-performance-metrics-machine-learning.html)
- Alpaydin (1999) propuso el test F combinado 5×2cv para eliminar la inestabilidad del 5×2cv t de Dietterich debida a la elección arbitraria de uno de los 10 estadísticos posibles. — [Raschka, arXiv 1811.12808](https://arxiv.org/pdf/1811.12808)

### Fórmulas (para el texto de la tesis)
Tabla 2×2 pareada sobre los N frutos: b = #(A correcto, B incorrecto), c = #(A incorrecto, B correcto). Accuracy_B − Accuracy_A = (c − b)/N.
- McNemar asintótico con corrección de continuidad (forma usada por Dietterich 1998): χ² = (|b − c| − 1)² / (b + c), 1 g.l.
- Exacto condicional: bajo H₀, b ~ Binomial(b+c, ½); p = 2·P(B ≤ min(b,c)) (truncado a 1).
- Mid‑p: p_mid = p_exacto − P(B = min(b,c)) [= exacto menos la probabilidad puntual, contando ambas colas].
- IC Wald de la diferencia pareada: (c−b)/N ± z·√(b + c − (c−b)²/N)/N. (Fórmula estándar de diferencia de proporciones pareadas; no verificada contra fuente primaria en esta sesión — ver Gaps.)

### Números concretos **[cálculo propio, α=0.05 bilateral]**
Escenario "A = 75 %, B = 80 %" (diferencia 5 puntos) con N = 150 frutos, variando la tasa de discordancia p_d = p₁₂+p₂₁ (p₂₁ − p₁₂ = 0.05):

| p_d | p₁₂ | p₂₁ | n pares para 80 % (Connor) | Potencia @150 (Connor) | Potencia @150 exacto cond. | Potencia @150 mid‑p |
|---|---|---|---|---|---|---|
| 0.10 | 0.025 | 0.075 | 312 | 0.49 | 0.41 | 0.49 |
| 0.15 | 0.050 | 0.100 | 469 | 0.35 | 0.28 | 0.34 |
| 0.20 | 0.075 | 0.125 | 626 | 0.28 | 0.21 | 0.26 |
| 0.25 | 0.100 | 0.150 | 783 | 0.23 | 0.18 | 0.23 |
| 0.30 | 0.125 | 0.175 | 940 | 0.20 | 0.16 | 0.20 |

Diferencia mínima detectable (MDD) con N = 150 y 80 % de potencia: p_d = 0.10 → ≈ 7 puntos; 0.15 → ≈ 8.7; 0.20 → ≈ 10.1; 0.25 → ≈ 11.2; 0.30 → ≈ 12.3 puntos (Connor y mid‑p coinciden a ±0.3 puntos). Con N = 235 (si se usara la imagen o todos los frutos como unidad): potencia para 5 puntos ≈ 0.51 (p_d=0.15) / 0.40 (p_d=0.20). Para dos modelos con 75–80 % de accuracy sobre las mismas imágenes, una discordancia de 0.15–0.25 es lo típico (los errores están parcialmente correlacionados; si fueran independientes p_d ≈ 0.75·0.20 + 0.25·0.80 = 0.35).

### Inferences
- Con 152 frutos, el estudio está **infra‑potenciado** para una diferencia de 5 puntos: debe declararse como limitación y reportar el IC de la diferencia (ancho esperado ≈ ±6–8 puntos) en lugar de interpretar "no significativo" como "equivalente".
- La variante mid‑p es la que conviene reportar como principal (mayor potencia que el exacto condicional sin inflar el error tipo I según Fagerland et al.); el exacto condicional puede añadirse como análisis de sensibilidad. Con b+c < 25 los asintóticos son dudosos; con b+c ≈ 20–40 (lo esperable aquí) la diferencia mid‑p vs exacto es ~0.03–0.07 de potencia.
- Para accuracy la unidad pareada debe ser el fruto (no la imagen), porque la clasificación se decide por fruto; para mAP/detección la unidad natural es la imagen. Hay que fijar esto a priori.
- El bootstrap pareado (≥ 2000 réplicas, remuestreo por **imagen** si varios frutos por imagen para respetar la dependencia intra‑imagen) es el complemento natural para métricas que no son proporciones (macro‑F1, MAE ordinal, QWK), para las que McNemar no aplica.

### Gaps
- No pude acceder al texto completo de Fagerland 2013 para extraer cifras exactas de la pérdida de potencia del exacto condicional vs mid‑p; mis cifras de la tabla son cálculo propio.
- La fórmula del IC Wald pareado y la fórmula de Connor no fueron verificadas contra el artículo original de Connor (1987, *Biometrics*); SAS la describe como aproximación.
- No encontré una fuente que publique potencia de McNemar tabulada específicamente para n≈150 en contexto ML.

---

## KQ2. Comparar modelos a través de los folds de CV (t pareada, Wilcoxon, t corregida de Nadeau–Bengio, Holm/Bonferroni) con un **test set fijo**

### Takeaway
Con 5 folds que sólo varían el conjunto de entrenamiento y se evalúan en el mismo test set, la t pareada sobre 5 números tiene dos problemas: (i) los 5 valores no son independientes (comparten test set y solapan entrenamiento → varianza subestimada, Dietterich 1998 / Nadeau–Bengio 2003) y (ii) con k=5 la potencia es bajísima (sólo detecta diferencias ≥ ~1.7 desviaciones estándar de las diferencias entre folds) y el Wilcoxon exacto con n=5 ni siquiera puede alcanzar p<0.05 bilateral. Lo defendible es: McNemar/bootstrap pareado sobre el test set por fruto como test principal (con el modelo de cada fold o con el ensamblado/mejor fold preespecificado), reportar media ± DE de las 5 réplicas como **variabilidad por semilla/entrenamiento**, y si se hace una t sobre folds, usar la corrección Nadeau–Bengio y declararla como heurística.

### Cited Findings
- Dietterich (1998): la t pareada sobre particiones aleatorias train/test tiene alto error tipo I y "should never be used"; el 5×2cv "directly measures variation due to the choice of training set" y es "slightly more powerful than McNemar's test". — [Dietterich 1998](https://web.engr.oregonstate.edu/~tgd/publications/dietterich-approximate-statistical-tests-nc1998.pdf)
- Motivación de la corrección: "The standard t‑test inflates Type I error when used in conjunction with random subsampling due to an underestimation of the variance, as found by Dietterich (1998)." Nadeau & Bengio (2003) "proved there is no unbiased estimator [of the variance], but it can be approximated with the factor 1/k". — [correctR vignette](https://archive.linux.duke.edu/cran/web/packages/correctR/vignettes/correctR.html)
- Fórmula del *corrected resampled t‑test* (Nadeau & Bengio 2003, *Machine Learning* 52:239–281, doi 10.1023/A:1024068626366): t = d̄ / √( (1/n + n₂/n₁)·σ̂² ), donde n = número de remuestreos, n₁ = tamaño del train, n₂ = tamaño del test, σ̂² = varianza muestral de las diferencias (la misma que usa la t pareada estándar, cuyo denominador es σ̂/√n). — [correctR vignette](https://cran.hafro.is/web/packages/correctR/vignettes/correctR.html); [correctR CRAN manual](https://search.r-project.org/CRAN/packages/correctR/correctR.pdf); artículo: [Nadeau & Bengio 2003](https://mlanthology.org/mlj/2003/nadeau2003mlj-inference); versión NIPS 1999: [papers.nips.cc](https://papers.nips.cc/paper/1661-inference-for-the-generalization-error)
- Razón de la correlación entre folds: "The t test assumes the measures of accuracy taken on the different folds to be independent. Such measures are instead correlated because of the overlap of the training sets built during cross‑validation. As a result the t test is not calibrated, namely its rate of Type I errors is much larger than the nominal size of the test." — [Corani et al., Machine Learning 2015 (doc.rero.ch)](https://doc.rero.ch/record/332061/files/10994_2015_Article_5486.pdf)
- Grados de libertad: una fuente usa df = r·k − 1 (para k‑fold repetido r veces; k−1 si r=1) y escribe el factor como 1/(rk) + n_test/n_train; otra describe "a t statistic on n − 1 degrees of freedom". Las fuentes **no coinciden** en la convención exacta y no pude verificar con el artículo original. — [arXiv 2606.04010](https://arxiv.org/pdf/2606.04010); [MetricGate NB test](https://metricgate.com/docs/nadeau-bengio-corrected-resampled-t-test/)
- Bouckaert & Frank (2004, PAKDD): "A test has low replicability if its outcome strongly depends on the particular random partitioning of the data used to perform it." Su extensión del factor de corrección para k‑fold repetido T veces: F = 1 + T·K/(K−1) (documentación de MachineShop); el resumen ICML 2004 de Bouckaert indica que "the combination of a sorted runs sampling scheme with a t‑test gives the most desirable performance judged on Type I and II error and replicability". — [Waikato research commons](https://researchcommons.waikato.ac.nz/handle/10289/1451); [MachineShop t.test](https://rdrr.io/cran/MachineShop/man/t.test.html); [ICML 2004 abstract](https://icml.cc/Conferences/2004/proceedings/abstracts/61.htm)
- Magnitud de la corrección: para r=20, k=10 el factor es 1/200 + 1/9 ≈ 0.116, "dominated by the 1/(k−1) term". — [arXiv 2606.04010](https://arxiv.org/pdf/2606.04010)
- Demšar (2006, JMLR 7:1–30) recomienda "the Wilcoxon signed ranks test for comparison of two classifiers and the Friedman test with the corresponding post‑hoc tests for comparison of more classifiers over multiple data sets" e introduce los diagramas CD. — [JMLR demsar06a](https://www.jmlr.org/papers/v7/demsar06a.html); García & Herrera (2008) extienden a comparaciones n×n. — [JMLR garcia08a](https://jmlr2020.csail.mit.edu/papers/volume9/garcia08a/garcia08a.pdf); Benavoli, Corani & Mangili (2016) advierten que los post‑hoc de rangos medios tipo Nemenyi dan conclusiones que "depend strongly on the number of methods included in the pool". — [citado en arXiv 2207.04258](https://arxiv.org/pdf/2207.04258)
- Holm (1979, *Scand. J. Statist.* 6:65–70): ordenar p‑valores p(1) ≤ … ≤ p(m), comparar p(i) con α/(m−i+1), rechazar secuencialmente y detenerse en el primer fallo; "controls the FWER for any joint distribution of the test statistics, and it is at least as powerful as single‑step Bonferroni". — [arXiv 1211.3313](https://arxiv.org/pdf/1211.3313); [R doc Holm](https://search.r-project.org/CRAN/refmans/HDLSSkST/html/Holm.html)
- Raschka (2018) resume el k‑fold CV paired t‑test, sus problemas de dependencia entre folds, el 5×2cv t y F combinado, y la nested CV. — [arXiv 1811.12808](https://arxiv.org/pdf/1811.12808)
- `scipy.stats.wilcoxon`: `zero_method` = "wilcox" (descarta ceros, default), "pratt" (más conservador), "zsplit"; `mannwhitneyu(method='exact')` recomendado "when there are no ties and when either sample size is less than 8". — [scipy wilcoxon](https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.stats.wilcoxon.html); [scipy mannwhitneyu](https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.stats.mannwhitneyu.html)

### Números concretos **[cálculo propio]**
- Potencia de la t pareada bilateral (α=0.05) sobre k folds, para d = media de las diferencias / DE de las diferencias entre folds: k=5 → d=0.5: 0.15; d=1.0: 0.40; d=1.5: 0.71; d=2.0: 0.91. k=10 → d=1.0: 0.80. k=25 → d=0.5: 0.67. Es decir, con 5 folds sólo se detecta (80 %) una diferencia ≈ 1.7 DE de la variabilidad entre folds.
- Wilcoxon de signos con n=5 pares: hay 2⁵=32 patrones de signo equiprobables bajo H₀; el p bilateral mínimo posible es 2/32 = 0.0625 > 0.05. **No puede dar significación** con 5 folds (puede con ≥6: 2/64 = 0.031).
- Corrección de Nadeau–Bengio con k=5, n₁ ≈ 4/5 del train, n₂ = test fijo: el factor (1/5 + n₂/n₁) se aplica al estimador naive 1/5; con n₂/n₁ ≈ 0.25 (test de 235 imágenes vs ~940 de entrenamiento) el error estándar se infla por √((0.2+0.25)/0.2) ≈ 1.5.

### Inferences
- **Validez de la t pareada sobre 5 folds con test set fijo**: no es válida como test de hipótesis en sentido estricto. La corrección Nadeau–Bengio corrige la dependencia por **solapamiento de entrenamiento**, pero fue derivada para remuestreo aleatorio train/test, no para un test set idéntico en todos los folds; la variabilidad que captura es la "variabilidad por conjunto de entrenamiento" (Dietterich), no la variabilidad muestral del test set. Con un test set fijo, la fuente de incertidumbre que importa para generalizar a nuevos frutos es la del test set → McNemar / bootstrap por fruto.
- Diseño recomendado (defendible en tesis): (1) Pre‑registrar un único modelo final por brazo (p. ej. el modelo del fold con mejor validación, o el promedio de predicciones de los 5 folds) y comparar los dos brazos en el test set fijo con McNemar mid‑p + IC bootstrap de la diferencia. (2) Reportar media ± DE (y rango) de la métrica en test de los 5 modelos de fold como medida de **estabilidad frente a la semilla/partición**, sin p‑valor, o con t corregida NB explícitamente etiquetada como aproximación. (3) Si se hacen k comparaciones (p. ej. accuracy, macro‑F1, MAE, QWK, error crítico), aplicar Holm sobre la familia de hipótesis confirmatorias y declarar el resto como exploratorias.
- Demšar/Friedman/Wilcoxon están diseñados para comparar sobre **múltiples datasets**, no sobre folds del mismo dataset; usarlos sobre 5 folds es un uso fuera de su supuesto (y sin potencia). Si la tesis compara ≥3 configuraciones (p. ej. YOLO‑5c, YOLO+CNN, YOLO+ViT), Friedman sobre folds es cuestionable; es mejor Cochran's Q / McNemar por pares con Holm sobre el test set por fruto.

### Gaps
- No se pudo verificar en el artículo original de Nadeau & Bengio (2003) la convención de grados de libertad ni la variante "conservative Z"; las fuentes secundarias difieren (n−1 vs rk−1).
- No encontré una fuente que trate explícitamente el caso "k‑fold que varía sólo el entrenamiento con test set fijo"; la inferencia anterior se deriva de los supuestos descritos en Dietterich 1998 y Corani et al. 2015.
- Bouckaert & Frank (2004) texto completo no accesible; la recomendación concreta (10×10 CV con t corregida) se cita de documentación de software.

---

## KQ3. Cómo reportar de forma justa un pipeline detectar‑luego‑clasificar: cajas GT (cota superior) vs cajas detectadas, atribución de errores, métricas (per‑class, macro‑F1, MAE ordinal, QWK, error crítico)

### Takeaway
La práctica estándar en visión es reportar (a) el clasificador sobre recortes de cajas **ground‑truth** como cota superior ("oracle boxes"), (b) el pipeline extremo a extremo sobre cajas **detectadas** emparejadas por IoU≥0.5, y (c) una descomposición de errores tipo Hoiem 2012 / TIDE 2020 (clasificación, localización, duplicado, fondo, perdido). Para 5 estadios ordenados hay que añadir métricas ordinales (MAE sobre el índice de estadio, QWK) y una métrica de coste de negocio ("error crítico" = cruce de frontera comercial), porque accuracy y macro‑F1 tratan igual confundir estadios adyacentes y extremos.

### Cited Findings
- Hoiem, Chodpathumwan & Dai (2012, ECCV) "Diagnosing Error in Object Detectors": analizan el efecto de oclusión, tamaño, aspecto, visibilidad de partes y punto de vista, y clasifican los falsos positivos en error de localización, confusión con objetos semánticamente similares, otros objetos etiquetados y fondo; "Sensitivity to size, localization error, and confusion with similar objects are the most impactful error sources." Publican software y anotaciones. — [Project page](https://dhoiem.cs.illinois.edu/projects/detectionAnalysis/); [ECCV 2012](https://mlanthology.org/eccv/2012/hoiem2012eccv-diagnosing)
- Bolya et al. (2020, ECCV) TIDE: "segments errors into six types and, crucially, is the first to introduce a technique for measuring the contribution of each error in a way that isolates its effect on overall performance"; funciona sobre ficheros de predicción, `pip install tidecv`; soporta COCO, LVIS, Pascal, Cityscapes; "the foundational work [Hoiem 2012] ... covers only false positives". — [TIDE arXiv 2008.08115](https://arxiv.org/pdf/2008.08115); [ECCV 2020](https://ecva.net/papers/eccv_2020/papers_ECCV/html/849_ECCV_2020_paper.php); [github dbolya/tide](https://www.github.com/dbolya/tide). (Los seis tipos — Cls, Loc, Cls+Loc, Dupe, Bkg, Missed — no se verificaron en el PDF en esta sesión.)
- Borji & Iranmanesh (2019) "Empirical Upper Bound in Object Detection": al clasificar con un clasificador fuerte las cajas GT, "the classification accuracy on VOC is very high (94.7%), which gives a very high upper‑bound AP (91.6 with the COCO API)" frente a 47.9 AP del mejor detector real; también "the surrounding boxes may contain additional visual content which may introduce noise in the labels". — [arXiv 1911.12451](https://arxiv.org/pdf/1911.12451)
- "Oracle MCG" (Pont‑Tuset & Van Gool 2015): elegir la mejor propuesta sobre cada objeto anotado "could be understood as the upper‑bound result one could get by restricting themselves to selecting object proposals without any further refinement". — [arXiv 1509.03660](https://arxiv.org/pdf/1509.03660)
- Un trabajo de segmentación evalúa tanto "with the ground truth bounding boxes around an object" como "using an object detector to automatically determine the object bounding boxes", y reporta una cota superior. — [Mottaghi et al. PAMI supp.](https://www.cs.utoronto.ca/~fidler/papers/mottaghiPAMI15supmat.pdf)
- Criterio de emparejamiento detección‑GT: "A predicted box counts as a true positive when its IoU with a ground‑truth fruit box meets a threshold, typically 0.50"; no emparejadas = FP; GT sin emparejar = FN. — [RF‑DETR vs YOLOv12 greenfruit, arXiv 2504.13099](https://arxiv.org/pdf/2504.13099)
- Ejemplo agrícola de dos etapas sin descomposición: pipeline de dátiles con YOLOv11 detector + YOLOv11 clasificador sobre recortes: "detection precision exceeded 90%, the classification accuracy approached 92%", **sin** separar accuracy sobre cajas GT vs detectadas. — [Research Square rs‑10395374](https://www.researchsquare.com/article/rs-10395374)
- Ejemplo de dos etapas con ordinal: Wei, Zheng et al. (2026, *J. Fungi* 12:737) sobre *Pleurotus citrinopileatus*: "a two‑stage RGB‑based framework for instance‑level maturity recognition"; tras localizar, "each detected instance was represented by paired local and context views and classified using a DINOv3 ViT‑S/16‑based ordinal model". — [citedrive / DOI 10.3390/jof12100737](https://www.citedrive.com/en/discovery/instance-level-maturity-recognition-of-pleurotus-citrinopileatus-via-ordinal-learning-and-morphology-guided-feature-fusion/)
- FruitProM‑V2 (Cheppally et al., Kansas State, 2026): trata la madurez como variable latente continua, convierte la distribución en probabilidades de clase vía la CDF; su estudio de anotación "found disagreement concentrated near adjacent maturity stages", motivación para tratamiento ordinal. — [alphaxiv 2604.26084](https://www.alphaxiv.org/abs/2604.26084.md)
- Dataset ordinal de palma aceitera (Kamal et al. 2024, Mendeley): 4 728 imágenes en cinco niveles ordenados "from immature to decayed", "designed for ordinal regression", sugiere "weighted loss functions to address imbalances". — [Mendeley 424y96m6sw](https://data.mendeley.com/datasets/424y96m6sw/1)
- Strawberry ripeness YOLO benchmark (2026) reporta TP/FP por clase para YOLOv8s/9c/11s y encuentra la clase "semi‑ripe" la más difícil. — [arXiv 2602.15656](https://arxiv.org/pdf/2602.15656)
- Survey Rizzo et al. (2023): la madurez la juzgan expertos y "the distance between the predicted class and the ground truth is based on some notion of error". — [arXiv 2212.14441](https://arxiv.org/pdf/2212.14441)
- QWK: pesos cuadráticos w_ij = 1 − ((i−j)/(k−1))² (o, equivalentemente, penalización ((i−j)/(k−1))²); "weights errors farther from the correct class more heavily, which suits ordered labels"; QWK y MAE son "similar in spirit, since both penalise errors on farther classes more". — [MetricGate weighted kappa](https://metricgate.com/docs/kappa-weighted-ordinal); [arXiv 1905.13392](https://arxiv.org/pdf/1905.13392); [arXiv 2305.09288](https://arxiv.org/pdf/2305.09288); QWK definido como "chance‑corrected agreement measure between a ground truth and a classifier". — [arXiv 1612.00775](https://arxiv.org/pdf/1612.00775)
- Benchmarks de Landis & Koch: κ > 0.80 "almost perfect". — [MetricGate weighted kappa](https://metricgate.com/docs/weighted-kappa)
- Principio de atribución: "if the detector returns a correct box and the final label is wrong, the classifier stage is at fault" (recurso educativo, no primario). — [1cademy](https://1cademy.com/node/attributing-a-correct-detection-with-a-wrong-subclass-label/7MCD5HAIyxwc7wwysSEx)

### Fórmulas
- κ_w = 1 − Σ_ij w_ij·O_ij / Σ_ij w_ij·E_ij, con w_ij = (i−j)²/(k−1)² (penalización), O = matriz de confusión observada, E = esperada por marginales (E_ij = row_i·col_j / N). Para k=5 estadios la penalización por confundir adyacentes es 1/16, por saltar dos estadios 4/16, extremos 16/16.
- MAE ordinal = (1/N) Σ |ŝ_i − s_i| con s = índice de estadio 0..4; también "accuracy ±1" (fracción con |ŝ−s| ≤ 1).
- Error crítico (coste de negocio): tasa de frutos cuya predicción cruza la frontera comercial (p. ej. predecir "comercial" cuando es "no comercial" o viceversa); es una proporción por fruto → McNemar/IC pareado aplican directamente.

### Protocolo propuesto de reporte (inferencia a partir de las fuentes)
Tabla A — **Clasificación con cajas GT (cota superior, "oracle boxes")**: sobre los 152 frutos comerciales (y todos los frutos), accuracy, per‑class recall, macro‑F1, MAE, QWK, error crítico, para (a) YOLO‑5c *restringido a sus cajas emparejadas* (tomando la clase de la detección con IoU máximo ≥0.5 con el GT) y (b) clasificador de 2ª etapa sobre recortes GT. Esto aísla la capacidad de clasificación.
Tabla B — **Extremo a extremo con cajas detectadas**: por fruto GT: {perdido (sin detección IoU≥0.5)} ∪ {detectado y clase correcta} ∪ {detectado y clase incorrecta}; además FP de fondo y duplicados. Reportar accuracy "condicional a detectado" y accuracy "global" (perdidos cuentan como error), más mAP@0.5 por clase.
Tabla C — **Atribución**: fracción de error total debida a (i) detección (perdidos + localización IoU<0.5), (ii) clasificación sobre caja bien localizada, (iii) ambos; con TIDE si se usa formato COCO. Diferencia Tabla A − Tabla B = coste de la detección imperfecta para cada brazo.
Comparación entre brazos: la misma detección en ambos brazos (mismo YOLO, uno con 5 clases y otro con 1–2 clases + clasificador) permite la comparación pareada por fruto; si el detector de 1 clase detecta más frutos que el de 5 clases, hay que reportar ambos conjuntos (intersección de frutos detectados por ambos para la comparación de clasificación pura; conjunto completo para la comparación global).

### Inferences
- Comparar accuracy de clasificación sólo sobre frutos detectados por *ambos* brazos (intersección) es lo que hace el test pareado válido; sobre el conjunto completo la comparación mezcla detección y clasificación y debe reportarse por separado.
- macro‑F1 es necesario por el desbalance entre estadios; MAE/QWK y el error crítico capturan la ordinalidad y el coste; conviene declarar **una** métrica primaria confirmatoria (p. ej. error crítico o accuracy en las dos clases comerciales) y el resto como secundarias.
- Para QWK y MAE el IC se obtiene por bootstrap pareado (remuestreando frutos/imágenes), no existe un McNemar.

### Gaps
- No encontré ningún artículo de madurez de fruta que reporte explícitamente accuracy sobre cajas GT vs cajas detectadas en un pipeline de dos etapas; la práctica está documentada en visión general (Borji 2019; Hoiem 2012; TIDE) y debe presentarse así en la tesis.
- No encontré artículos de madurez de fresa/tomate/banano/mango que usen QWK o MAE como métrica (sólo el trabajo de *Pleurotus* 2026 y FruitProM‑V2 usan formulación ordinal; sus métricas exactas no se pudieron extraer). La tesis puede justificar QWK/MAE citando la literatura de clasificación ordinal general.
- Las seis categorías de TIDE y las cifras numéricas del artículo no fueron verificadas en el PDF.

---

## KQ4. Selección de hiperparámetros sólo en folds de validación y el argumento de la nested CV (Varma & Simon 2006; Cawley & Talbot 2010)

### Takeaway
Reutilizar la misma CV para elegir hiperparámetros y estimar el error produce un sesgo optimista sustancial (Varma & Simon: −8 a −12 puntos en datos nulos); la nested CV lo elimina casi por completo. Cawley & Talbot formalizan que la selección de modelo es parte del entrenamiento y debe quedar dentro del bucle de evaluación. Con un test set fijo nunca tocado, el diseño "5 folds de train/val para seleccionar + test set único para reportar" es equivalente a una capa externa de un solo split; lo esencial es que ninguna decisión (umbrales, época, arquitectura del clasificador, umbral de confianza/IoU, umbral de cobertura) se tome mirando el test set.

### Cited Findings
- Varma & Simon (2006), *BMC Bioinformatics* 7:91: "Using CV to compute an error estimate for a classifier that has itself been tuned using CV gives a significantly biased estimate of the true error." En datos nulos (error verdadero 50 %), el estimador CV del clasificador optimizado dio en promedio 37.8 % (Shrunken Centroids) y 41.7 % (SVM), sesgos de −12.2 y −8.3 puntos; "for SVM with optimal parameters the estimated error rate was less than 30% on 38% of 'null' data‑sets". "Proper use of CV for estimating true error of a classifier developed using a well defined algorithm requires that all steps of the algorithm, including classifier parameter tuning, be repeated in each CV loop"; la nested CV "provides an almost unbiased estimate of the true error". — [PMC1397873](https://pmc.ncbi.nlm.nih.gov/articles/PMC1397873); [BMC Bioinformatics PDF](https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/1471-2105-7-91); [Table 1](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-91/tables/1)
- Cawley & Talbot (2010), *JMLR* 11:2079–2107, "On Over‑fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation": el criterio de selección de modelo evaluado sobre datos finitos tiene varianza y puede sobreajustarse; la selección de modelo debe tratarse como parte integral del ajuste y quedar dentro del procedimiento de evaluación (nested CV) para evitar el sesgo de selección. — [JMLR cawley10a](https://www.jmlr.org/papers/v11/cawley10a.html); [PDF](https://jmlr2020.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf); [UEA eprints](https://ueaeprints.uea.ac.uk/id/eprint/3640/)
- Raschka (2018) describe la nested CV para selección de modelo y la advertencia de que el k‑fold paired t‑test compara *algoritmos* con datos solapados, mientras McNemar compara dos *modelos* ya entrenados. — [arXiv 1811.12808](https://arxiv.org/pdf/1811.12808)
- Dietterich (1998): McNemar sólo mide la variación del test set para dos modelos concretos; "does not measure variation due to the choice of training set" (razón por la que recomienda 5×2cv cuando se puede entrenar 10 veces). — [Dietterich 1998](https://web.engr.oregonstate.edu/~tgd/publications/dietterich-approximate-statistical-tests-nc1998.pdf)

### Inferences
- Diseño coherente con estas fuentes: (1) partición única estratificada train+val (5 folds) / test; (2) **toda** selección (épocas, lr, aumentos, arquitectura del clasificador de recortes, umbral de confianza de YOLO, umbral de IoU, umbral de "cobertura" del mapa de calor, política de desempate entre detecciones solapadas) con la métrica de validación promediada sobre los 5 folds; (3) congelar el protocolo (idealmente por escrito/pre‑registro en el repositorio) antes de evaluar en test; (4) una sola pasada por test por brazo. Si se compara también el umbral de cobertura seleccionado sobre test, debe reportarse como análisis exploratorio.
- La selección del "mejor fold" para el modelo final debe hacerse por métrica de **validación**, no de test; de lo contrario se reintroduce el sesgo de Varma & Simon (equivalente a optimizar sobre test con 5 candidatos: sesgo pequeño pero real, ≈ el máximo de 5 variables correlacionadas).
- La magnitud del sesgo de Varma & Simon (8–12 puntos) es en escenarios de alta dimensión con muchas configuraciones; con pocas configuraciones el sesgo será menor, pero en un test set de 152 frutos cualquier sesgo de 2–3 puntos es del orden de la diferencia que se quiere detectar.

### Gaps
- No pude leer el texto completo de Cawley & Talbot para extraer cifras experimentales concretas; sólo el resumen/claims generales.
- No encontré literatura específica sobre selección de umbral de confianza de YOLO en validación vs test con cuantificación del sesgo.

---

## KQ5. Comparar dos grupos de valores de "cobertura" de saliencia (frutos correctos vs. frutos con error crítico): Mann–Whitney U, Cliff's delta y tamaño mínimo de muestra

### Takeaway
Mann–Whitney U (exacto o permutación si el grupo pequeño tiene < ~20) con Cliff's delta (δ = 2U/(n₁n₂) − 1, equivalente a P(X>Y) − P(X<Y)) y su IC bootstrap es la elección correcta para cobertura (variable acotada en [0,1], no normal). El problema es el **desbalance**: con ~130 frutos correctos y sólo 10–20 con error crítico, el efecto mínimo detectable (80 %) es δ ≈ 0.4–0.53 (efecto "medio–grande" según Romano et al.); efectos pequeños no son detectables.

### Cited Findings
- Cliff's delta: "estimates the probability that a randomly drawn value from one group exceeds a randomly drawn value from the other, minus the reverse probability"; forma estandarizada del estadístico de Mann–Whitney; δ = 2·U/(n₁n₂) − 1. — [rstatix cliff_delta](https://rdrr.io/cran/rstatix/man/cliff_delta.html); [MetricGate Cliff's delta](https://metricgate.com/docs/cliffs-delta)
- Umbrales de Romano et al. (2006) "Appropriate statistics for ordinal level data" (Florida Association of Institutional Research): |δ| < 0.147 insignificante, < 0.33 pequeño, < 0.474 medio, ≥ 0.474 grande. — [rstatix cliff_delta source](https://rdrr.io/cran/rstatix/src/R/cliff_delta.R). (Un pie de tabla en PLoS ONE da cortes distintos, probablemente erróneo. — [PMC6927660 t007](https://pmc.ncbi.nlm.nih.gov/articles/PMC6927660/table/pone.0226224.t007))
- Tamaño muestral de Noether (1987) para Mann–Whitney: método "for determining a sample size which guarantees that the Mann‑Whitney test at a given significance level α has a predetermined power 1−β"; efecto p = P(X<Y) (0.5 = nulo). "Approximations by Noether and O'Brien‑Castelloe are shown to be inaccurate for small sample sizes"; alternativa Monte Carlo en el paquete R `wmwpow`; en su tabla (α=0.05 bilateral), Noether con 15 por grupo y p=0.8 da ≈ 81 % de potencia, con 6 por grupo ≈ 44 %. — [Mollan et al. 2020, PMC7316590](https://pmc.ncbi.nlm.nih.gov/articles/PMC7316590); [tabla](https://pmc.ncbi.nlm.nih.gov/articles/PMC7316590/table/T1)
- Potencia a posteriori de Mann–Whitney: "the rank‑sum statistic has no closed‑form power formula under realistic alternatives", se estima por simulación igualando P(X₁ > X₂). — [MetricGate post‑hoc power MW](https://metricgate.com/docs/posthoc-power-mann-whitney/)
- `scipy.stats.mannwhitneyu`: 'exact' recomendado sin empates y con n < 8 en algún grupo; "the exact method is not corrected for ties"; con empates y muestras pequeñas usar `PermutationMethod`. — [scipy mannwhitneyu](https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.stats.mannwhitneyu.html)

### Fórmulas
- Noether (grupos iguales): n por grupo ≈ (z₁₋α/₂ + z₁₋β)² / (6·(p − ½)²), p = P(X>Y). Forma general con N total y fracción c = n₁/N: N ≈ (z₁₋α/₂ + z₁₋β)² / (12·c·(1−c)·(p − ½)²). Relación con Cliff: δ = 2p − 1 (sin empates).
- IC de δ: bootstrap percentil remuestreando dentro de cada grupo (≥2000 réplicas), o IC asintótico de Cliff (1993) como implementa `rstatix::cliff_delta(ci=TRUE)`.

### Números concretos **[cálculo propio, Noether, α=0.05 bilateral, 80 %]**
- Grupos iguales: δ=0.20 (p=0.60) → 131 por grupo; δ=0.30 → 58; δ=0.40 → 33; δ=0.50 → 21; δ=0.60 → 15; δ=0.70 → 11.
- Desbalanceado (n₁ correctos, n₂ error crítico): 130/10 → δ detectable ≈ 0.53 (p≈0.77); 125/15 → 0.44; 120/20 → 0.39; 110/30 → 0.33; 100/40 → 0.30.
- Nota: Mollan et al. advierten que Noether es inexacto para n pequeño (≈ 6–15 en el grupo pequeño), así que para el diseño real conviene confirmar por simulación (p. ej. `wmwpow` o Monte Carlo propio con distribuciones Beta ajustadas a la cobertura observada).

### Inferences
- Con 10–20 frutos con error crítico, sólo un efecto **grande** (δ ≥ 0.4–0.5, es decir, P(cobertura_correcto > cobertura_error) ≥ 0.70–0.77) será significativo; debe plantearse como análisis exploratorio con IC de δ y no como prueba confirmatoria, o ampliar el grupo "error" usando todos los frutos mal clasificados (no sólo errores críticos) y/o agregando los 5 modelos de fold (con la salvedad de que las predicciones de los 5 modelos sobre el mismo fruto no son independientes → remuestreo por fruto).
- Reportar también medianas y rangos intercuartílicos de cobertura por grupo y la probabilidad de superioridad p = (δ+1)/2, interpretable directamente ("en el 72 % de los pares, el fruto correcto tiene más cobertura").
- Si la cobertura se mide con varios métodos de saliencia (Grad‑CAM, Eigen‑CAM…) o umbrales, aplicar Holm a la familia de tests.

### Gaps
- No se accedió al texto original de Noether (1987) ni al de Romano et al. (2006); fórmulas y umbrales provienen de fuentes secundarias (Mollan et al. 2020; documentación de rstatix) y de conocimiento previo verificado contra la tabla de Mollan (15/grupo, p=0.8 → ≈81 %).
- No encontré literatura que estandarice métricas de "cobertura" de mapas de calor para frutos ni tamaños muestrales típicos en estudios de XAI agrícola.

---

## KQ6. Tamaño de muestra: cuántos frutos comerciales se necesitan en el test set para detectar una diferencia de 5 puntos de accuracy con 80 % de potencia

### Takeaway
Pareado (McNemar, mismo test set): entre ~310 y ~940 frutos según la tasa de discordancia (470–630 para discordancia 15–20 %, lo típico entre dos modelos de 75–80 % de accuracy). No pareado (dos proporciones independientes 75 % vs 80 %): ~1 094 frutos **por grupo**. Con 152 frutos el estudio detecta ~9–10 puntos; con 235 unidades, ~7–8 puntos. Esto debe declararse en la sección de limitaciones y, si es posible, ampliarse el test set comercial o reportar la comparación como estimación con IC.

### Cited Findings
- Fórmula de Connor (1987) para pares en McNemar y su dependencia exclusiva de los pares discordantes: ver KQ1. — [MetricGate](https://metricgate.com/docs/sample-size-mcnemar-paired); [SAS POWER (METHOD=CONNOR, aproximación)](https://support.sas.com/documentation/cdl/en/statug/67523/HTML/default/statug_power_syntax49.htm)
- Dietterich 1998: el test de dos proporciones independientes aplicado a dos clasificadores evaluados en el mismo test set "should never be used" (ignora el pareo). — [Dietterich 1998](https://web.engr.oregonstate.edu/~tgd/publications/dietterich-approximate-statistical-tests-nc1998.pdf)
- Mid‑p como test con mayor potencia manteniendo el nivel (Fagerland et al. 2013). — [BMC MRM 13:91](https://www.doi.org/10.1186/1471-2288-13-91)

### Fórmulas
- Pareado (Connor): n = ( z₁₋α/₂·√p_d + z₁₋β·√(p_d − Δ²) )² / Δ², Δ = p₂₁ − p₁₂ = diferencia de accuracy; p_d = tasa de discordancia. z₀.₉₇₅ = 1.960, z₀.₈₀ = 0.842.
- No pareado (dos proporciones): n por grupo = ( z₁₋α/₂·√(2p̄(1−p̄)) + z₁₋β·√(p₁(1−p₁)+p₂(1−p₂)) )² / (p₂−p₁)².
- Semi‑ancho del IC de Wald al 95 % para una accuracy p: 1.96·√(p(1−p)/n).

### Números concretos **[cálculo propio]**
| Objetivo | Supuesto | n requerido |
|---|---|---|
| Δ=5 pts pareado, p_d=0.10 | errores muy correlacionados | 312 frutos |
| Δ=5 pts pareado, p_d=0.15 | típico | 469 frutos |
| Δ=5 pts pareado, p_d=0.20 | típico | 626 frutos |
| Δ=5 pts pareado, p_d=0.25 | | 783 frutos |
| Δ=5 pts pareado, p_d=0.30 | errores casi independientes | 940 frutos |
| Δ=5 pts no pareado 75 % vs 80 % | grupos independientes | 1 094 por grupo |
| Potencia con n=152, Δ=5, p_d=0.15 / 0.20 | | 0.35 / 0.28 |
| Potencia con n=235, Δ=5, p_d=0.15 / 0.20 | | 0.51 / 0.40 |
| Potencia con n=400, Δ=5, p_d=0.15 / 0.20 | | 0.74 / 0.61 |
| MDD (80 %) con n=152, p_d=0.15 / 0.20 | | ≈ 8.7 / 10.1 pts |
| Semi‑ancho IC95 de una accuracy ≈0.78 | n=150 / 235 / 500 | ±6.6 / ±5.3 / ±3.6 pts |

### Inferences
- La discordancia p_d se puede **estimar en validación** (contando frutos en que los dos brazos discrepan en los folds de validación) y usarla para un cálculo de potencia a priori que se incluya en el capítulo de metodología; eso convierte la limitación en un análisis formal.
- Si se quiere potencia razonable para 5 puntos, el test set comercial debería tener ≥ ~450–650 frutos; una alternativa realista en tesis es (a) ampliar el test con frutos adicionales anotados sólo para evaluación, o (b) reformular el objetivo confirmatorio en términos de la métrica de coste (error crítico), cuya diferencia esperada entre brazos puede ser mayor que 5 puntos y, por tanto, detectable con 152 frutos.
- Reportar siempre el IC de la diferencia pareada; con 152 frutos un IC de ±7–8 puntos es informativo (descarta diferencias grandes) aunque no demuestre superioridad.

### Gaps
- Las cifras dependen de p_d, que no está en la literatura para este problema; debe estimarse empíricamente en validación.
- No encontré guías publicadas de tamaño de test set específicas para visión agrícola; las fuentes son bioestadística general y Dietterich 1998.
