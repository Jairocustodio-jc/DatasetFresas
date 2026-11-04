# Two-stage (detect → crop → classify) vs single-stage multi-class detectors for fruit ripeness, with emphasis on strawberry and intermediate stages

**Access note (important for the writer):** in this session the egress proxy blocked direct full-text access to mdpi.com, arxiv.org, pmc.ncbi.nlm.nih.gov, sciencedirect.com, researchgate.net, wiley.com, frontiersin.org, europepmc.org and semanticscholar.org (HTTP 403 CONNECT). Only huggingface.co was reachable. All findings below therefore come from (a) search-engine excerpts of the primary papers' abstracts/tables/figure captions and (b) secondary citations in later papers. Every number is flagged as **[primary excerpt]** (text that the search engine returned verbatim from the primary paper) or **[secondary]** (reported by a citing paper/review). Numbers that could not be verified from the primary paper are listed under *Gaps*. Please re-check every number against the full text before it goes into the thesis.

---

## KQ1. Which papers (2019–2026) use YOLO + crop classifier for strawberry or other fruit ripeness? Second-stage accuracy, number of classes, and whether they compared with a single-stage detector on the same data

### Takeaway
Only a handful of fruit-ripeness papers implement a true detector → crop → classifier pipeline, and almost none run a controlled comparison against a single multi-class detector on the same data. The two strawberry examples (He et al. 2025, CEA; Agriculture 2026 greenhouse grading) use the second stage for a *different* label set than the detector (pickability, or a 5-grade ordinal scale) and compare the classifier only against *other classifiers* (YOLOv8s-cls, ResNet-18, EfficientNet, MobileNetV3, ResNet50), not against the detector's own class head. The one same-data comparison found (tomato, Mutha et al. 2023, Plants) shows YOLOv5m alone ≈ YOLOv5m+CNN (97% vs 97–98%), i.e., a near-null result on 3 coarse classes.

### Cited Findings

**Strawberry**

- **He, Karkee & Zhang (2025), "Enhanced machine vision system for field-based detection of pickable strawberries: Integrating an advanced two-step deep learning model merging improved YOLOv8 and YOLOv5-cls", Computers and Electronics in Agriculture 234:110173.** Stage 1: modified YOLOv8s (C3x modules, extra detection head, α-IoU / IoU-based loss) detecting strawberries in 3 maturity stages (Stage I immature, Stage II nearly mature, Stage III mature). Stage 2: YOLOv5s-cls classifying *mature* berries into pickable vs unpickable (binary), selected over YOLOv8s-cls, ResNet-18, EfficientNet-b0 and EfficientNet-b1 for "comparable accuracy" at 2.8 ms/image. Field validation on a separate field: AP 89.0% / 82.0% / 90.0% for Stages I/II/III; classifier accuracy 100.0% (unpickable) and 95.0% (pickable). Combined inference 10.6 ms (8.3 ms detection + 2.3 ms classification). The modified detector ("Model D") gained +2.5% mAP over original YOLOv8s **[primary excerpt via abstract/search]** — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0168169925002790); [ResearchGate](https://www.researchgate.net/publication/389778343_Enhanced_machine_vision_system_for_field-based_detection_of_pickable_strawberries_Integrating_an_advanced_two-step_deep_learning_model_merging_improved_YOLOv8_and_YOLOv5-cls)
- Secondary report of the same paper gives per-stage AP 84.2% (I), 77.8% (II), 87.8% (III), mAP 83.2% (+2.5% vs original YOLOv8s) **[secondary]** — [B2G-YOLO11-S, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S277237552500838X); the discrepancy with the 89/82/90 figures is most plausibly test-set (lab/validation) vs the separate field-validation set, but this could not be confirmed. Note the "nearly mature" Stage II is the worst class in both reports (77.8% / 82.0%).
- **Key point for the thesis:** in He et al. the *maturity* labels are produced by the single-stage detector; the second stage does **not** re-classify maturity, it classifies pickability. The 2025 *Plants* review (DHN-YOLO) paraphrases it as "the second stage employs YOLOv5s-CLS to perform maturity classification", which is a loose description **[secondary]** — [DHN-YOLO, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12655673/)
- **Agriculture 2026, 16(14):1550, "Vision-Based Strawberry Ripeness Grading for Harvest Decision Support in Smart Greenhouses".** Two-stage: YOLO11m detects strawberry ROIs (detection mAP50 = 0.910 on test), each ROI classified into **5 ordinal ripeness grades** by a custom classifier (CP2-A2C2F) with a 5-channel input (RGB + Excess Green + Hue). Classification accuracy 0.877, weighted F1 0.877, MAE 0.162 on the ordinal scale (with auxiliary regression head); baselines on the same crops: MobileNetV3 0.861, ResNet50 0.851. Latency ≈ 16 ms detection + ≈ 13 ms classification. Ablation: adding ExG+Hue channels reduced ordinal error vs RGB-only; P2+P4 feature fusion beat including P3. Authors state that **missed detections in stage 1 cannot be recovered by stage 2** **[primary excerpt]** — [doi:10.3390/agriculture16141550](https://doi.org/10.3390/agriculture16141550). No comparison against a single 5-class YOLO on the same data was found in excerpts.
- **Wang et al. (2024), Agriculture 14(5):751, "YOLOv8+ Model and Image Processing Method"** — detector + colour-ratio rule rather than detector + CNN; see KQ3.
- **Ogundele et al. (2025), Agriculture 15(18):1906** — single-stage improved YOLOv8s (C3x, extra head, WIoU: same recipe as He et al.'s detector), 3 classes, overall mAP@0.5 92.5%, −27.9% parameters vs baseline; per-class AP@0.5: unripe 93.0 (P 84.2, R 90.7), semi-ripe 86.3 (P 79.1, R 82.6), ripe 98.0 (P 93.4, R 95.6) **[primary excerpt, Table 7]** — [doi:10.3390/agriculture15181906](https://doi.org/10.3390/agriculture15181906). No classifier stage; included as the same-architecture single-stage reference, where the intermediate class is ≈ 7–12 AP points below the extremes.
- **Zhou et al. (2021), Smart Agricultural Technology 1:100001, "Strawberry maturity classification from UAV and near-ground imaging"** — YOLOv3 as a single-stage multi-class detector with **3 (UAV) and 7 (near-ground) maturity stages**; best mAP 0.88 (UAV, 2 m) and 0.89 (near-ground); highest class AP 0.93/0.94 for fully mature **[primary excerpt]**. The ResearchGate listing says ResNet18/50/101, VGG16/19, DenseNet, Inception-v3, AlexNet were also run as maturity classifiers, but no accuracies surfaced **[secondary]** — [ResearchGate](https://www.researchgate.net/publication/355152962_Strawberry_Maturity_Classification_from_UAV_and_Near-Ground_Imaging_Using_Deep_Learning)

**Other fruits**

- **Mutha et al. (2023), Plants 12(4):790, "Classification of Tomato Fruit Using Yolov5 and Convolutional Neural Network Models".** YOLOv5m alone vs YOLOv5m + ResNet-50 / ResNet-101 / EfficientNet-B0; 3 classes (ripe, immature, damaged); 4,500 training images (1,500/class after augmentation), 200 epochs. Testing accuracies 98% (ResNet-50), 98% (EfficientNet-B0), 97% (YOLOv5m alone), 97% (ResNet-101). YOLOv5m+ResNet-50: 100% ripe, 100% immature, 94% damaged. Authors: "combining Yolo5m with ResNet-101 not only failed to improve the accuracy but also increased the training time"; damaged-class errors attributed to colour similarity with ripe/immature **[primary excerpt]** — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9959894/); [MDPI](https://www.mdpi.com/2223-7747/12/4/790). This is the closest thing found to a same-data two-stage vs single-stage ablation for fruit; the gain is ≤1 point on coarse classes.
- **Chen et al. (2022), Precision Agriculture, citrus maturity** — stage 1 YOLOv5 detects fruits; stage 2 improved visual-saliency map + RGB fed to a **4-channel ResNet34** for maturity; accuracy 95.07%, +3.14 points over best RGB-only CNN (VGG16) and +18.24 over best classical ML (KNN) **[primary excerpt]** — [Springer](https://link.springer.com/article/10.1007/s11119-022-09895-2). Comparison is classifier-vs-classifier, not vs a multi-class detector.
- **AgroDualNet (2026, PMC13216622)** — YOLOv8 crops → MobileNetV2, 4 ripeness classes (unripe, semi-ripe, ripe, over-ripe), apples (Kaggle + field), 98.88% classification accuracy; edge-oriented **[primary excerpt]** — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13216622/). No single-stage baseline reported in excerpts.
- **DHN-YOLO (2025), Plants 14(22):3439** (the review that cites He et al.) — single-stage YOLO11-pose-based joint maturity + keypoint detector on the MSRBerry dataset; precision 87.3%, recall 88%, mAP@50 91.8%, mAP@50:95 78.6%. Class definitions: "ripe" = 1/3–3/4 red surface; "fully ripe" = fully enlarged vivid red without white-green mottling; the authors claim it "significantly reduces confusion between ripe and fully ripe" **[primary excerpt]** — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12655673/); [doi:10.3390/plants14223439](https://doi.org/10.3390/plants14223439)
- Tomato three-stage YOLOv5s cascade (bunch → unoccluded fruit → maturity) exists (2024) but no accuracy surfaced **[secondary]** — [AFBF-YOLO, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12389433/)

### Inferences
- In the fruit literature, "two-step" systems are usually motivated by a *second task* (pickability, occlusion, grading granularity, ordinal regression, extra colour channels) rather than by a documented accuracy deficit of the detector's class head. The only same-data ablation (tomato) shows essentially no gain on 3 coarse classes, consistent with the general-CV evidence (KQ6) that the benefit of cropping concentrates in fine-grained/long-tailed/small-object settings.
- He et al.'s detector and Ogundele et al.'s detector are the same architecture recipe (C3x + extra head + IoU-loss) by overlapping teams, with the intermediate class ("nearly mature"/"semi-ripe") consistently the weakest (77.8–86.3 AP vs 87.8–98.0 for the ripe class).

### Gaps
- No strawberry study found that trains a crop classifier on the *same* maturity labels as the detector and reports both on the same test set.
- He et al. 2025: dataset image counts, train/test split, stage-2 training set size, classifier input resolution and the exact accuracies of YOLOv8s-cls/ResNet-18/EfficientNet-b0/b1 could not be retrieved (full text blocked). Discrepancy 89/82/90 vs 84.2/77.8/87.8 unresolved.
- Agriculture 2026 16(14):1550: dataset size, class definitions and per-class confusion not retrieved.
- Zhou 2021: per-class AP for the 7 near-ground stages and the ResNet classifier numbers not retrieved.

---

## KQ2. Primary source of "He et al." cited in Plants 14(22):3439

### Takeaway
The citation resolves to Zixuan He, Manoj Karkee & Qin Zhang, *Computers and Electronics in Agriculture* 234 (2025) 110173. Its actual protocol is: improved YOLOv8s (3 maturity stages) → YOLOv5s-cls binary pickable/unpickable on mature-fruit crops; the second stage is *not* a maturity classifier.

### Cited Findings
- Title: "Enhanced machine vision system for field-based detection of pickable strawberries: Integrating an advanced two-step deep learning model merging improved YOLOv8 and YOLOv5-cls"; authors He, Karkee, Zhang; CEA vol. 234, art. 110173, 2025 **[primary listing]** — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0168169925002790); author profile [Zixuan He, ResearchGate](https://www.researchgate.net/profile/Zixuan-He-15)
- Stage 1: "enhanced YOLOv8s model, incorporating C3x modules, an additional head network, and IOU-based techniques, for strawberry detection"; Stage 2: "YOLOv5s-CLS to perform maturity classification of strawberries, thereby determining their harvest readiness" (wording as quoted by DHN-YOLO) **[secondary]** — [PMC12655673](https://pmc.ncbi.nlm.nih.gov/articles/PMC12655673/)
- Abstract wording: "YOLOv5s-cls was identified as the preferred model for classifying mature strawberries into pickable and unpickable groups, offering a good inference speed of 2.8 ms per image" with accuracy comparable to YOLOv8s-cls, ResNet 18, EfficientNet-b0 and EfficientNet-b1; field validation AP 89.0/82.0/90.0% (Stages I/II/III), classification accuracy 100.0% unpickable, 95.0% pickable; total 10.6 ms (8.3 + 2.3 ms) **[primary excerpt]** — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0168169925002790)
- Alternative numbers reported by a citing paper: AP 84.2% / 77.8% / 87.8%, mAP 83.2%, +2.5% over original YOLOv8s **[secondary]** — [B2G-YOLO11-S](https://www.sciencedirect.com/science/article/pii/S277237552500838X)
- DHN-YOLO (Plants 2025) additionally attributes "94.4% detection accuracy, 83% picking-point accuracy and 87.3% maturity-classification accuracy" to prior work in the same paragraph; it is unclear from the excerpt which numbers belong to He et al. and which to other cited works **[secondary, ambiguous]** — [PMC12655673](https://pmc.ncbi.nlm.nih.gov/articles/PMC12655673/)

### Inferences
- The Plants review's framing ("two-stage maturity classification") over-states what the primary paper did: maturity comes from the detector; the classifier adds a pickability decision. For the thesis, cite He et al. as evidence that a lightweight cls head (YOLOv5s-cls, ~2.3–2.8 ms) is cheap enough for real time, not as evidence that a second stage improves maturity accuracy.

### Gaps
- Full protocol (image counts, split, crop resolution, training details, confusion matrices) not retrievable; the 89/82/90 vs 84.2/77.8/87.8 discrepancy needs the full text.

---

## KQ3. "Strawberry Detection and Ripeness Classification Using YOLOv8+ Model and Image Processing Method" (Agriculture 2024, 14(5):751)

### Takeaway
Wang et al. use an improved YOLOv8 ("YOLOv8+") as a **binary** ripe/unripe detector, then a rule-based colour analysis (red-pixel ratio along the fruit centreline, 50% threshold) rather than a learned classifier for the "ripeness classification"; the colour stage reaches 91.91% accuracy, below the detector's 97.81%. No timing figure surfaced.

### Cited Findings
- Authors Wang, Wang, Han, Zhang, Kong & Zou; Agriculture 2024, 14(5), 751; doi 10.3390/agriculture14050751; data on GitHub **[primary listing]** — [MDPI](https://www.mdpi.com/2077-0472/14/5/751)
- Dataset: 1,187 JPEG strawberry images split 949 / 119 / 119 (train/val/test), of which 742 ripe images were used for the image-processing stage; the abstract elsewhere mentions 1,021 annotated images from Changchun Xiaohongmao Plantation (conflict unresolved) **[primary excerpt]** — [ResearchGate](https://www.researchgate.net/publication/380579454_Strawberry_Detection_and_Ripeness_Classification_Using_YOLOv8_Model_and_Image_Processing_Method)
- Detector classes: ripe vs unripe (2 classes). YOLOv8+ accuracy (precision) 97.81%, recall 96.36%, F1 97.07 **[primary excerpt]** — [ResearchGate](https://www.researchgate.net/publication/380579454_Strawberry_Detection_and_Ripeness_Classification_Using_YOLOv8_Model_and_Image_Processing_Method)
- Ripeness rule: red ratio = red pixels on the strawberry's centreline / all centreline pixels; < 50% "not fully ripe", ≥ 50% "fully ripe". Image-processing accuracy 91.91%, FPR 5.03%, FNR 14.28% **[primary excerpt]** — [ResearchGate figure "Ripeness classification of strawberries"](https://www.researchgate.net/figure/Ripeness-classification-of-strawberries_fig2_380579454)
- Workflow figure "Overall workflow of the proposed method" (detection → centreline colour analysis) — [ResearchGate fig. 7](https://www.researchgate.net/figure/Overall-workflow-of-the-proposed-method_fig7_380579454)

### Inferences
- This paper is often cited as a "two-stage" strawberry ripeness system, but the second stage is hand-crafted colour thresholding on a binary decision, and it is *less* accurate than the detector. It is weak support for a learned crop classifier; it is better used as evidence that colour fraction is the de-facto ripeness criterion (consistent with the 10%–90% / >90% red thresholds used by Yurdakul et al. and the 1/3–3/4 red rule in DHN-YOLO).

### Gaps
- Inference time / FPS, hardware, mAP50 value, and the 1,187 vs 1,021 image-count conflict could not be verified (full text blocked).

---

## KQ4. "A Novel Public Dataset for Strawberry (Fragaria × ananassa) Ripeness Detection and Comparative Evaluation of YOLO-Based Models" (arXiv 2602.15656; Food Science & Nutrition 2026, 14(7):e72112; PMC13365362)

### Takeaway
Yurdakul & Taşdemir benchmark 12 YOLOv8/9/11 variants on a small 3-class greenhouse dataset (566 images; 1,169–1,734 annotations depending on the version). Best mAP@50 is 86.09% (YOLOv8s); the intermediate "semi-ripe" class is the hardest, with 75% of semi-ripe instances correct, 19% predicted unripe and 6% fully-ripe, which the authors attribute to intrinsic visual overlap between adjacent stages and for which they suggest finer-grained or ordinal labels.

### Cited Findings
- Dataset: images from two greenhouses in Turkey under varying illumination; classes unripe / semi-ripe / fully-ripe, with semi-ripe = 10%–90% red surface and fully-ripe > 90% red; public on Kaggle and IEEE DataPort **[primary excerpt]** — [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/fsn3.72112); [IEEE DataPort](https://ieee-dataport.org/documents/multi-class-strawberry-ripeness-detection-dataset)
- Size: 566 images; annotation count reported as 1,201 (arXiv preprint), 1,169 (published version excerpt) and 1,734 (another arXiv excerpt) — **conflict unresolved** **[primary excerpts]** — [arXiv PDF](https://arxiv.org/pdf/2602.15656); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13365362/)
- Headline: YOLOv8s best mAP@50 86.09%, mAP@50:95 75.52%; YOLOv9c best mAP@50:95 76.53%, best F1 83.79% and highest precision 90.94%; YOLO11s highest recall 83.74% and higher recall on semi-ripe **[primary excerpt]** — [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/fsn3.72112); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13365362/)
- YOLOv8s per class (Table 5, test set): unripe (70 inst.) P 83.4, R 86.0, mAP50 85.2, F1 84.7; semi-ripe (16 inst.) P 97.2, R 75.0, mAP50 93.3, F1 84.7; fully-ripe (37 inst.) P 85.3, R 73.0, mAP50 79.8 **[primary excerpt]** — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13365362/). Note the test set has only 16 semi-ripe instances, so per-class figures are noisy.
- Confusion analysis (YOLOv8s): unripe and fully-ripe extremes "almost never confused with each other"; errors concentrate on semi-ripe: 75% correct, 19% → unripe, 6% → fully-ripe; partially reddened fruit sometimes predicted unripe and less often fully-ripe. Authors attribute this to "inherent visual overlap of adjacent ripeness stages rather than a weakness of the architecture" and suggest finer-grained or ordinal labels **[primary excerpt]** — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13365362/)
- "The semi-ripe class stands out as the most challenging class due to its morphological transition characteristics"; models diverged most on this class **[primary excerpt]** — [arXiv PDF](https://arxiv.org/pdf/2602.15656)
- Authors note YOLO single-stage detectors "work faster and can operate with relatively lower hardware requirements" as their reason for a single-stage benchmark **[primary excerpt]** — [arXiv PDF](https://arxiv.org/pdf/2602.15656)

### Inferences
- The semi-ripe class has high precision (97.2) but low recall (75.0): the detector under-predicts the middle class and pushes ambiguous fruit to the extremes, mostly to unripe. For a thesis argument, this is the clearest published confusion pattern that a dedicated crop classifier (possibly ordinal) could target.
- With 16 semi-ripe test instances, the 19%/6% split corresponds to ~3 and ~1 fruit; treat as indicative only.

### Gaps
- Per-class tables for the other 11 models, training hyper-parameters, and the exact annotation count could not be verified.
- The paper does not test a two-stage pipeline; its remedy (finer/ordinal labels) is a suggestion, not an experiment.

---

## KQ5. Ablations showing per-class confusion between adjacent ripeness stages, and whether a crop classifier reduces it

### Takeaway
Adjacent-stage confusion is reported consistently across single-stage strawberry detectors (semi-ripe/half-ripe/turning classes 7–20 AP points below the ripe class; 19–22% of mid-stage fruit misassigned to a neighbour), but **no study was found that measures the same confusion before and after adding a crop classifier** on the same data. The only quantified fine-grained gains from a second stage come from outside fruit (plant disease, camera traps, CUB).

### Cited Findings
- Yurdakul 2026: semi-ripe 75% correct / 19% → unripe / 6% → fully-ripe (YOLOv8s) **[primary excerpt]** — [PMC13365362](https://pmc.ncbi.nlm.nih.gov/articles/PMC13365362/)
- LBS-YOLO (Fu et al., Frontiers in Plant Science, Dec 2025, PMC12711796): mAP@0.5 88.6%, mAP@0.5:0.95 75.8% (+2.2 / +1.3 over YOLO11n), recall 86.4%, F1 82.9%, 1.6 M params, 3.4 MB; "commercially-ripe" has the lowest AP, 0.748, and is "seriously mutually misjudged with the half-ripe category"; one excerpt reports 22% of half-ripe samples misjudged as commercially-ripe and only 53% of commercially-ripe correctly identified **[primary excerpt]** — [Frontiers](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1715263/full); [PubMed](https://pubmed.ncbi.nlm.nih.gov/41424561/)
- Ogundele 2025: semi-ripe AP 86.3 vs ripe 98.0 and unripe 93.0 **[primary excerpt]** — [doi:10.3390/agriculture15181906](https://doi.org/10.3390/agriculture15181906)
- He et al. 2025: Stage II "nearly mature" AP 77.8–82.0 vs 84.2–90.0 for the extremes **[primary excerpt / secondary]** — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0168169925002790)
- A YOLOv11 strawberry study: "Red (fully ripe)" class mAP@0.5 0.889 highest; transitional classes lower "due to class imbalance and visual similarity" **[primary excerpt]** — [JAIC, Polibatam](https://jurnal.polibatam.ac.id/index.php/JAIC/article/view/12283?articlesBySimilarityPage=22)
- A 2026 strawberry growth-stage study: misclassifications "predominantly concentrated between phenologically adjacent stages", e.g. 8% of green-fruit → early-fruit **[primary excerpt]** — [Frontiers 2026](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1888504/xml)
- DHN-YOLO claims reduced ripe vs fully-ripe confusion from its architecture (single-stage), without a published matrix in the excerpts **[primary excerpt]** — [PMC12655673](https://pmc.ncbi.nlm.nih.gov/articles/PMC12655673/)
- Agriculture 2026 greenhouse grading (two-stage, 5 ordinal grades): crop classifier 0.877 acc vs MobileNetV3 0.861 / ResNet50 0.851 on the same crops; adding ExG + Hue channels reduced *ordinal* error (MAE 0.162) **[primary excerpt]** — [doi:10.3390/agriculture16141550](https://doi.org/10.3390/agriculture16141550). This is the only strawberry paper that treats adjacent-stage error as an ordinal problem in a crop classifier, but it does not compare against a 5-class detector.
- Tomato (Mutha 2023): the crop classifier did not fix the confusable class either — damaged-class recall stayed 0.92–0.94 across YOLOv5m alone and all YOLO+CNN combos **[primary excerpt]** — [PMC9959894](https://pmc.ncbi.nlm.nih.gov/articles/PMC9959894/)

### Inferences
- Across datasets with different class schemes (3 to 7 classes), the intermediate stage is systematically the weakest for single-stage YOLO (AP deficit ≈ 7–20 points vs the ripe class; 19–22% leakage to a neighbour). This supports the *motivation* for a second stage, but the *efficacy* claim for strawberry remains untested in the literature — a gap the thesis could fill.
- Where the confusable class is confusable because of colour (tomato "damaged"), a crop classifier on RGB did not help; where additional channels (ExG/Hue) or ordinal losses were added, modest gains (1.6–2.6 acc points, lower MAE) were reported.

### Gaps
- No before/after confusion matrix for strawberry detector-only vs detector+classifier on identical test data was found.
- Full confusion matrices for LBS-YOLO, DHN-YOLO, Ogundele and the Agriculture 2026 paper could not be retrieved.

---

## KQ6. General CV evidence: does detect-then-classify improve fine-grained accuracy, and when does it not?

### Takeaway
Outside fruit, controlled comparisons exist and the direction is consistent: cropping to a detected object before classification helps when classes are fine-grained, long-tailed, small in the frame or when background causes overfitting (camera traps +≈25% macro-F1; CUB 63.0 → 67.8 → 70.5%; plant disease +0.6–7.5 macro-F1 and up to 6× on unseen fields). It does not help, and can hurt, when context matters (herds, background cues), when the first stage misses objects (no recovery), when classes are coarse (tomato 3-class), or when crop quality is poor.

### Cited Findings

**Positive results**

- **Gadot et al. (2024), IET Computer Vision 18(8), "To crop or not to crop"**: adding a species-agnostic detection stage before an EfficientNetV2-M classifier "yields a macro-average F1 improvement of around 25% on a very large, long-tailed dataset", replicated on two public datasets; same architecture/hyper-parameters for whole-image and crop classifiers; the gain is attributed to both the detector and to "cropping images to detected animals prior to classification … above and beyond the coarse-categorisation benefit" **[primary excerpt]** — [Wiley](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.12318); [Google Research](https://research.google/pubs/to-crop-or-not-to-crop-comparing-whole-image-and-cropped-classification-on-a-large-dataset-of-camera-trap-images/). Basis of Google SpeciesNet — [GitHub cameratrapai](https://github.com/google/cameratrapai)
- Camera-trap active learning (arXiv 1910.09716, Snapshot Serengeti): "using crops produces significantly better results on our data than using full images … cropped images eliminate background pixels" **[primary excerpt]** — [arXiv](https://arxiv.org/pdf/1910.09716)
- Zero-shot camera-trap paper: full-image classifiers "tend to overfit to backgrounds", hurting generalisation to new locations; some background remains even in crops **[primary excerpt]** — [arXiv 2410.12769](https://arxiv.org/html/2410.12769v1)
- A Spanish mammal study: two-stage pipeline outperforms a single 24-class classifier via "divide-and-conquer" **[primary excerpt]** — [Scientific Reports 2025](https://www.nature.com/articles/s41598-025-90249-z)
- **CUB-200 (Zhou et al. 2016, CAM)**: GoogLeNet-GAP 63.0% on full images → 67.8% when trained/tested on crops from its own localisation → 70.5% with human bounding boxes **[primary excerpt]** — [arXiv 1512.04150](https://arxiv.org/pdf/1512.04150). Fine-grained surveys note CUB "cropped version is also typically higher in accuracy than the original form, because it avoids interference from the background", and a two-pass "infer twice" scheme lifted CUB from 85.6% to 88.1% **[primary excerpt]** — [Springer](https://link.springer.com/article/10.1007/s11042-022-13619-z); WS-BAN also crops-and-resizes the localised object for a second inference and averages probabilities **[primary excerpt]** — [arXiv 1808.02152](https://arxiv.org/pdf/1808.02152)
- **Plant disease, HODRF (Iwano et al., arXiv 2407.17906 / IEEE 2024)**: YOLOv7 detects ROIs without disease label, EfficientNetV2 classifies; on 4 crops, 20 classes, 281K images, HODRF beat YOLOv7-alone by 5.8–21.5 points on healthy data and 0.6–7.5 macro-F1 points; rationale: "OD over-detects healthy cases, and HODRF significantly reduces these errors by using CL in the second stage" **[primary excerpt]** — [arXiv](https://arxiv.org/html/2407.17906v1); [IEEE](https://ieeexplore.ieee.org/document/10948896/)
- **Suwa et al. (arXiv 1910.11506, IEEE BigData Workshops 2019, cucumber wide-angle)**: end-to-end detectors (SSD, Faster R-CNN) reach F1 81.5–84.1% on same-population test data, but two-stage systems reach F1 ≈ 33.4–38.9% on an *unseen* farm, "more than six times higher than end-to-end systems"; a later paper paraphrases that an SSD system "directly caused overfitting" and was "extremely problematic … for diagnosing unknown data" **[primary excerpt / secondary]** — [arXiv abs](https://arxiv.org/abs/1910.11506); [HODRF](https://arxiv.org/html/2407.17906v1). Related same-farm vs other-farm drop 86.0% → 20.7% F1 cited for the wide-angle setting **[secondary]** — [arXiv 2407.18000](https://arxiv.org/html/2407.18000v1)
- **Fuentes et al. (2018), Frontiers in Plant Science, "Refinement Filter Bank"**: Faster R-CNN-style primary detector + per-class secondary CNN verifiers that remove false positives; recognition ≈96% (reviews cite mAP 96.25% vs 85.98% for baseline detectors), "+13% compared to our previous work" **[primary excerpt; mAP figures secondary]** — [Frontiers](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2018.01162/full); [PMC6124392](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6124392/)
- Agricultural robot crop-type/growth-stage paper: one-phase detector mAP ≈ 67.50% vs two-phase (crop type first) ≈ 74.2% **[primary excerpt]** — [Agronomy 13(12):2873](https://www.mdpi.com/2073-4395/13/12/2873)
- Vehicle fine-grained: RT-DETR → ViT-B/16 on crops, 6 body types, 0.94 accuracy, per-class F1 0.91–0.97, abstention when softmax < 0.60; rationale "coarse detection stage reduces the search space and filters background", classifier invoked selectively **[primary excerpt]** — [arXiv 2606.05149](https://arxiv.org/abs/2606.05149). YOLO11 two-stage make/model (625 classes, 84,732 images): classifier on crops "gives cleaner inputs in multi-vehicle scenes"; 80% on real traffic (98.29% in another listing, setup unclear) **[primary excerpt]** — [Urban Science 10(2):74](https://www.mdpi.com/2413-8851/10/2/74)
- Edge deployment: a YOLO detector + crop classifier on a Jetson Orin NX ran at 12.5 vs 13.3 FPS detector-only at 1080p by running the classifier on the DLA one frame behind; crop preprocessing costs 6–9% of pipeline time **[primary excerpt]** — [arXiv 2608.11770](https://arxiv.org/html/2608.11770v1)

**Negative / null results and caveats**

- Tomato 3-class: YOLOv5m alone 97% vs +ResNet-101 97% ("failed to improve the accuracy but also increased the training time"); +ResNet-50/EfficientNet-B0 98% **[primary excerpt]** — [PMC9959894](https://pmc.ncbi.nlm.nih.gov/articles/PMC9959894/)
- iWildCam 2021 winner: the bbox-crop model gives better predictions for single highlighted animals, "while the full image model has been shown to be more effective at classifying images with animal herds" **[primary excerpt]** — [arXiv 2609.03233](https://arxiv.org/pdf/2609.03233)
- Crop-only classification "ignores context"; background animals detected by MegaDetector yield mislabelled crops (Dussert et al., Methods Ecol Evol 2026) **[primary excerpt]** — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.07.15.664849v2.full)
- First-stage misses are unrecoverable (strawberry greenhouse grading) **[primary excerpt]** — [doi:10.3390/agriculture16141550](https://doi.org/10.3390/agriculture16141550)
- Crop quality matters: "rigid rectangular proposals often introduce background noise … leading to overly large or imprecise bounding boxes" (CUB/iNat method paper) **[primary excerpt]** — [Biomimetics 10(12):834](https://doi.org/10.3390/biomimetics10120834)
- DeepForest (tree crowns): the shipped ResNet-18 crop model with 148 species reaches only 39.3% accuracy / macro-F1 0.23 on 2,453 test crops — a two-stage design does not by itself solve a long-tailed fine-grained problem **[primary, model card]** — [Hugging Face](https://huggingface.co/weecology/cropmodel-tree-species)
- Reviews repeat the generic trade-off "two-stage methods perform better in precision, single-stage in speed" but this refers to RPN-based detectors, not to detect-then-classify pipelines **[secondary]** — [PMC11659763](https://pmc.ncbi.nlm.nih.gov/articles/PMC11659763/); [arXiv 2406.10139](https://arxiv.org/pdf/2406.10139)

### Inferences
- The size of the gain scales with (i) class granularity/long tail (camera traps +25% macro-F1; CUB +5–8 points) and (ii) domain shift (cucumber 6× on an unseen farm), and shrinks toward zero for 3 coarse, colour-separable classes (tomato). Strawberry ripeness with 3 classes sits near the "small gain" end; with 5–7 ordinal grades or cross-farm deployment it moves toward the regime where the literature reports gains.
- The main documented *mechanisms* are background removal (less overfitting to scene), higher effective resolution of the object, the ability to use a stronger/different backbone or extra channels on the crop, and selective invocation only on confusable detections.

### Gaps
- No controlled detect-then-classify vs single-stage detector ablation on iNaturalist was found.
- Exact macro-F1 values and dataset sizes for Gadot et al. and HODRF's unseen-field results were not retrievable.

---

## KQ7. Classifier backbones used on crops and input sizes

### Takeaway
Crop classifiers in this literature are small CNNs — YOLOv5s-cls/YOLOv8s-cls, ResNet-18/34/50, EfficientNet-B0/B1/V2, MobileNetV2/V3 — with ViT-B/16 appearing only in non-agricultural fine-grained work. Where stated, input is 224×224 (ImageNet-style; YOLO-cls pretrained weights are 224); no fruit paper found reports 128 px crops.

### Cited Findings
- He et al. 2025: YOLOv5s-cls chosen over YOLOv8s-cls, ResNet-18, EfficientNet-b0, EfficientNet-b1; 2.8 ms/image (2.3 ms within the 10.6 ms system) **[primary excerpt]** — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0168169925002790)
- Ultralytics YOLOv8-cls pretrained models (n–x) all use 224 input, ImageNet-pretrained; docs' toy example trains at imgsz 64, so 224 is the pretrained resolution rather than a hard default **[primary docs]** — [Ultralytics classify docs mirror](https://huggingface.co/spaces/chelseat2023/yolov8/blob/main/docs/en/tasks/classify.md?code=true). YOLOv5-cls default not verified (see Gaps).
- Agriculture 2026 greenhouse grading: custom CP2-A2C2F with 5-channel input vs MobileNetV3 and ResNet50 **[primary excerpt]** — [doi:10.3390/agriculture16141550](https://doi.org/10.3390/agriculture16141550)
- Mutha 2023 tomato: ResNet-50, ResNet-101, EfficientNet-B0 on YOLOv5m detections **[primary excerpt]** — [PMC9959894](https://pmc.ncbi.nlm.nih.gov/articles/PMC9959894/)
- Chen 2022 citrus: 4-channel ResNet34 (RGB + saliency) **[primary excerpt]** — [Springer](https://link.springer.com/article/10.1007/s11119-022-09895-2)
- AgroDualNet: MobileNetV2 on YOLOv8 crops **[primary excerpt]** — [PMC13216622](https://pmc.ncbi.nlm.nih.gov/articles/PMC13216622/)
- DeepForest crop model: ResNet-18, 224×224 RGB, ImageNet normalisation, nearest-neighbour resize, crops at 0.1 m resolution, sqrt inverse-frequency class weights **[primary, model card]** — [Hugging Face](https://huggingface.co/weecology/cropmodel-tree-species)
- Gadot 2024 / SpeciesNet: EfficientNetV2-M for both whole-image and crop classifiers **[primary excerpt]** — [Wiley](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.12318)
- HODRF: EfficientNetV2 on YOLOv7 ROIs **[primary excerpt]** — [arXiv 2407.17906](https://arxiv.org/html/2407.17906v1)
- Vehicle pipelines: ViT-Base/16 on RT-DETR crops **[primary excerpt]** — [arXiv 2606.05149](https://arxiv.org/abs/2606.05149); a person-attribute crop classifier paper stresses "high input resolution is necessary to preserve sufficient spatial detail" for fine-grained attributes in small crops **[primary excerpt]** — [arXiv 2608.11770](https://arxiv.org/html/2608.11770v1)
- Fuentes 2018: independent per-class CNN verifiers (filter bank) **[primary excerpt]** — [Frontiers](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2018.01162/full)

### Inferences
- For a strawberry thesis pipeline, the field norm is a 224×224 ImageNet-pretrained small CNN (YOLOv8s-cls / MobileNetV3 / EfficientNet-B0 / ResNet-18) with 2–13 ms per crop on GPU; the only agricultural evidence of added value from the classifier comes with extra inputs (saliency, ExG/Hue) or ordinal heads, not from the backbone alone.

### Gaps
- Crop input resolution was not stated in any fruit paper excerpt (He et al., Mutha et al., Chen et al., AgroDualNet, Agriculture 2026); the YOLOv5 `classify/train.py` default imgsz was not verified.
- No fruit paper found that ablates crop size (e.g. 128 vs 224) or ViT vs CNN on fruit crops.
