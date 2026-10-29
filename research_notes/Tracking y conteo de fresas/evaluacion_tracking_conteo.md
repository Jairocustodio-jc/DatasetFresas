# Evaluating detection + tracking + per-track classification for fruit counting (thesis protocol notes)

Scope note on sourcing: in this session every direct page fetch (arxiv.org, sciencedirect.com, openaccess.thecvf.com, zenodo.org, academic.oup.com, link.springer.com, pmc.ncbi.nlm.nih.gov) was blocked by the sandbox network policy, so all "Cited Findings" below come from search-engine result summaries of those pages, not from reading the full papers. Numbers were cross-checked across several hits where possible, but exact formulas/tables inside paywalled or PDF sources should be verified by the thesis author before quoting. Everything labelled **[inference]** is the researcher's own derivation or recommendation, not a sourced claim. All sample-size arithmetic in this file was recomputed with Python (stdlib) from the stated formulas and assumptions.

---

## KQ1. Public fruit video datasets with track IDs (any crop) and their baselines

### Takeaway
There is no widely adopted standard benchmark for fruit "counting from tracks"; the closest public MOT-format releases are NeSTCount (4 crops, moving camera, 2026), the Growing Strawberries Dataset (GSD, WACV 2024, time-lapse strawberries with identities), StraMOT (StraTracker, 53 videos) and MOrangeT (citrus). Baselines are reported as HOTA/MOTA/IDF1 and, for counting, as MAE / MAPE / R² against manual counts; none of the strawberry sets is a walking-speed robot pass over beds, so your own annotated subset will be needed for in-domain evaluation.

### Cited Findings
- **NeSTCount** ("A multi-species moving-camera field benchmark for sequential fruit counting": grapefruit, sugar belle, tomato, Valencia) provides eight sequences of 800 contiguous frames from continuous moving-camera passes over field rows, with per-frame boxes and persistent fruit IDs in MOT format, CC-BY-4.0 — [Zenodo record 21636487](https://zenodo.org/records/21636487). (Zenodo blocked in session; baseline numbers not retrievable.)
- **Growing Strawberries Dataset (GSD)**: time-lapse captured hourly by six cameras over 16 months (2021–2022), 24 plants in two greenhouses, human-annotated boxes for every strawberry at all growth stages plus identities for trajectories; evaluated with HOTA, MOTA, IDF1; ByteTrack best HOTA, OC-SORT fewest ID switches, StrongSORT improved when weighting motion over appearance — [Wen et al., WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Wen_The_Growing_Strawberries_Dataset_Tracking_Multiple_Objects_With_Biological_Development_WACV_2024_paper.pdf); [TU Delft record](https://repository.tudelft.nl/record/uuid:0020c8ea-c47f-4f4d-af63-c1562186eb01); [4TU extension dataset](https://data.4tu.nl/datasets/73424098-2e29-4cc0-9fb2-86e379c8ced0). Supplementary: a GNN-based matcher (GMT) reached HOTA ≈ 38 on the first 750 frames and took nearly a week to run — [WACV 2024 supplementary](https://openaccess.thecvf.com/content/WACV2024/supplemental/Wen_The_Growing_Strawberries_WACV_2024_supplemental.pdf).
- **StraMOT / StraTracker**: 56,584 frames from 53 videos, dual-region counting strategy for growing strawberries — [StraTracker, Computers and Electronics in Agriculture](https://www.sciencedirect.com/science/article/abs/pii/S0168169924009554). Public availability not confirmed.
- **YOLOv7-pruning strawberry paper**: 55 test videos × 100 frames each, tracking labels made manually with DarkLabel — [arXiv 2407.12614](https://arxiv.org/pdf/2407.12614).
- **YOLOv8s + class-aware ByteTrack (strawberry ripeness)**: 2,190 images and 37 greenhouse videos, labelled with Label Studio; counting R² = 0.914 and MAPE = 9.52 % vs ground truth, ripe class best (R² = 0.950); notes that counting every detection from a moving camera over-counts, motivating a counting line — [Agriculture 15(18):1906, 2025](https://doi.org/10.3390/agriculture15181906).
- **MinneApple** (Häni, Roy, Isler; IEEE RA-L 2020): >41,000 instances in 1,000 images for detection/segmentation plus a patch-based counting set (64,597 train / 5,764 test patches, 6 scenes); baselines for detection, segmentation and counting; CodaLab challenge; data at UMN conservancy — [arXiv 1909.06441](https://arxiv.org/pdf/1909.06441); [UMN repository](https://conservancy.umn.edu/handle/11299/206575). It is image/patch based, **not** a video-tracking set.
- **MOrangeT** (citrus) with 3-D fruit relocalization for occlusion/re-entry — [Pipeline for multiple orange detection and tracking (preprint)](https://www.researchgate.net/publication/376894201_A_PIPELINE_FOR_MULTIPLE_ORANGE_DETECTION_AND_TRACKING_WITH_3-D_FRUIT_RELOCALIZATION_AND_NEURAL-NET_BASED_YIELD_REGRESSION_IN_COMMERCIAL_CITRUS_ORCHARDS_A_PREPRINT).
- Apple-orchard tracking under intermittent occlusion (CVPRW 2024) built hand-labelled tracking GT where a unique fruit ID is manually assigned to each box — [Matos et al., CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/papers/Matos_Tracking_and_Counting_Apples_in_Orchards_Under_Intermittent_Occlusions_and_CVPRW_2024_paper.pdf).
- **NTrack** (cotton bolls) is another in-field MOT + counting dataset with tracker baselines — [arXiv 2312.10922](https://arxiv.org/pdf/2312.10922).
- Passion-fruit tracker reports HOTA 79.6 %, MOTA 92.58 %, IDF1 95.02 % together with IDSW — [Precision Agriculture 2024](https://link.springer.com/article/10.1007/s11119-024-10132-1).
- Strawberry re-identification counting (Kirk, Mangan, Cielniak, ICVS 2021, LNCS 12899 pp. 211–222): counting evaluated against manually labelled track counts; MAPE 3 % (L1 = 7) vs 21 % (L1 = 41) for a baseline MOT tracker — [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-030-87156-7_17).

### Inferences
- **[inference]** For a Huaral strawberry campaign, GSD is useful only for pre-training/benchmarking the association step (static camera, hourly time-lapse); NeSTCount is the closest public analogue to the robot-pass setting (moving camera, field rows) and should be used as an external sanity check of your tracker's HOTA/IDSW, while the in-domain evaluation must be your own annotated clips.
- **[inference]** Report your baselines in the same format as GSD/NeSTCount (MOT-format GT, HOTA/MOTA/IDF1/IDSW) so results are comparable.

### Gaps
- NeSTCount baseline numbers, GSD track counts and annotation hours could not be read (fetch blocked).
- Public availability of StraMOT and the 2407.12614 videos is unconfirmed.

---

## KQ2. Which tracking metric best predicts counting accuracy (HOTA vs MOTA debate; IDSW/Frag → double counts)

### Takeaway
No paper found establishes a quantitative mapping from any MOT metric to count error; fruit papers treat ID switches and fragmentation as *diagnostics* for double counting, and MOTA is known to be detection-dominated while IDF1 is association-dominated (Luiten et al. 2021). For counting-by-tracking, association quality (AssA, IDF1, IDSW, Frag) is the relevant family, and the count error itself must be measured separately.

### Cited Findings
- HOTA (Luiten et al., IJCV 129:548–578, 2021; online Sept 2020) combines detection, association and localization; final HOTA is the geometric mean of DetA and AssA at each localization threshold, averaged over thresholds; MOTA matches and scores at the detection level (detection bias) while IDF1 works at trajectory level (association bias) — [cvlibs PDF](https://www.cvlibs.net/publications/Luiten2020IJCV.pdf); [arXiv 2009.07736](https://arxiv.org/html/2009.07736.pdf); [IJCV](https://link.springer.com/content/pdf/10.1007/s11263-020-01375-2.pdf).
- "MOTA is mostly a proxy for detection accuracy and is therefore highly correlated with DetA" — [HOTA paper via search summary](https://www.cvlibs.net/publications/Luiten2020IJCV.pdf). Worked example: on MOT17 a tracker ranked 1st by MOTA was 3rd by HOTA; another ranked 31st by MOTA was 5th by HOTA because it had the best AssA of 37 trackers — [Computer Vision Online](https://computervisiononline.com/explains/hota/).
- HOTA user study: head-to-head video pairs where metrics disagreed (67 pairs HOTA vs MOTA, 51 HOTA vs IDF1, 69 MOTA vs IDF1); a secondary summary reports 230 participants (62 MOT researchers), HOTA preferred 61.6 % over MOTA and 72.0 % over IDF1 overall (79.3 % / 85.9 % among experts) — [arXiv 2009.07736](https://arxiv.org/pdf/2009.07736); numbers from [alphaxiv summary](https://alphaxiv.org/overview/2009.07736v2) (verify in paper).
- CLEAR MOT (Bernardin & Stiefelhagen, EURASIP JIVP 2008): MOTA = 1 − (FN + FP + IDSW)/GT, range (−∞, 100 %]; MOTP measures localization only and "does not take into account any tracking performance such as keeping consistent trajectories" — [EURASIP JIVP 2008](https://jivp-eurasipjournals.springeropen.com/articles/10.1155/2008/246309); survey formulation in [Ciaparrone et al. survey](https://arxiv.org/pdf/1907.12740). Fragmentation ("Frag") is listed only as "total number of fragmentations" (a GT trajectory interrupted and resumed) — [survey](https://arxiv.org/pdf/1907.12740).
- MOTChallenge evaluation "does not explicitly handle target re-identification": a target that leaves the field of view and reappears is treated as a new ID in GT — [MOT16 benchmark paper](https://arxiv.org/pdf/1603.00831).
- Fruit papers: a 2026 edge-hardware paper treats identity switches and fragmentation as diagnostics because they "can directly affect fruit-count reliability" and evaluates counting with event-level counting F1 and relative count error — [arXiv 2609.13551](https://arxiv.org/pdf/2609.13551). A dynamic-Kalman orchard tracker reports IDSW rising under heavy occlusion/overlap, lowering MOTA and IDF1 — [Sensors 25(13):4138, 2025](https://www.mdpi.com/1424-8220/25/13/4138).
- Structure-from-motion counting paper: a fruit lost to occlusion and later re-tracked "gets counted as a different fruit", appearing twice in the raw count — [Liu et al., arXiv 1804.00307](https://ar5iv.labs.arxiv.org/html/1804.00307). OrangeSort was designed specifically to alleviate double counting of occluded fruit — [Zhang et al., Hort. Res. 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9113225/).
- A fixed camera gives ~one track per fruit but IDSW still occur; with a moving camera the same fruit can receive several IDs so counts are "best effort" — [FruitCountingAndHarvestEstimator README](https://github.com/sanjarbek1030/FruitCountingAndHarvestEstimator).
- Tooling: TrackEval is the official MOTChallenge kit (HOTA, CLEAR, Identity; pure numpy/scipy) — [TrackEval GitHub](https://github.com/JonathonLuiten/TrackEval); [HOTA-metrics](https://github.com/JonathonLuiten/HOTA-metrics); Roboflow `trackers` also computes CLEAR/HOTA/Identity from MOT files — [Roboflow Trackers evals](https://trackers.roboflow.com/latest/api/evals/).

### Inferences
- **[inference]** Mechanism: for "count = number of unique track IDs", each fragmentation of a GT fruit into k predicted tracks adds (k−1) duplicates; each ID *swap* between two fruits that does not create a new ID does **not** change the count. So Frag (and the number of predicted tracks per GT identity) is the more direct predictor of over-count than IDSW; IDSW over-penalises swaps that are count-neutral. Missed fruits (GT identities with zero matched tracks) drive under-count and relate to DetA/recall, not to association.
- **[inference]** Recommended reporting set: HOTA (headline), DetA, AssA, IDF1, IDSW, Frag, MT/ML from TrackEval; plus the count-specific decomposition below (KQ4/KQ6). Report MOTA for comparability but do not use it to pick a tracker for counting.
- **[inference]** Because MOTChallenge GT assigns a *new* ID when a target re-enters after leaving the frame, HOTA/IDF1 computed with that convention will *reward* a tracker that re-identifies a fruit after it leaves/re-enters less than counting does; for counting GT you need a per-fruit identity that persists across the whole pass (see KQ4), i.e. two GT files or one GT with a "fruit_id" column distinct from "track_id".

### Gaps
- No source quantifies correlation between IDSW/Frag/AssA and count error on fruit data; this would be a novel contribution of the thesis (compute Spearman ρ between per-clip Frag and per-clip |relative count error| across clips/trackers).
- The exact Frag definition in Bernardin & Stiefelhagen 2008 could not be read.

---

## KQ3. Standard "counting from tracks" evaluation in fruit papers: ground truth, unit, number of units, metrics

### Takeaway
The dominant practice is manual counting of the *same video* (per tree/plant/row or per clip) as GT, with MAE, RMSE, MAPE/relative error and R² (regression of predicted vs manual) as metrics; units are usually trees or short video clips, with n in the range ~6–60 units. Only a minority validate against in-field (post-harvest) counts, which several authors note is the only GT that reflects the real number of fruits.

### Cited Findings
- Zhang et al. (Hort. Res. 9, uhac003, 2022): six video sequences, two fields, 22 trees; tracking counts vs manual counts on the same videos; MAE 0.081 (SD 0.08) for OrangeSort vs 0.45 / 1.212 (SD 0.4741 / 1.3975) for SORT / DeepSORT; detector mAP 0.957 — [Hort. Res.](https://academic.oup.com/hr/article/doi/10.1093/hr/uhac003/6526907); [PMC9113225](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9113225/). (The MAE of 0.081 appears to be a *relative* error per sequence; verify units in the paper.)
- Kirk et al. 2021 (strawberry re-ID): GT = manually labelled tracks (counts); MAPE 3 %, L1 = 7 vs baseline MAPE 21 %, L1 = 41 — [Springer](https://link.springer.com/chapter/10.1007/978-3-030-87156-7_17).
- YOLOv8s + ByteTrack strawberry (2025): 37 videos; R² = 0.914, MAPE 9.52 % vs GT counts — [Agriculture 2025](https://doi.org/10.3390/agriculture15181906).
- Hydroponic strawberry yield mapping: central counting zone so each tracked berry is counted once; vs manual counts R² = 0.9997 and mean relative error 1.5511 % — [Smart Agricultural Technology 2025](https://www.sciencedirect.com/science/article/pii/S2772375525005246).
- Mobile robot indoor strawberry yield monitoring: counting algorithm checked against a manual count taken as GT so counting could be judged separately from detection — [PMC10167025](https://pmc.ncbi.nlm.nih.gov/articles/PMC10167025/).
- Greenhouse eggplant YOLOv5s-DeepSORT: relative error = % deviation of predicted from true count per video; best 16.7 % on one video; R² = 0.9677 overall — [Agriculture 16(2):253, 2026](https://doi.org/10.3390/agriculture16020253).
- UAV strawberry ripe counting at three times of day: accuracy 89.7–95.7 % — [PMC12516899](https://pmc.ncbi.nlm.nih.gov/articles/PMC12516899/).
- Fruitlet counting (YOLOv8–v12 comparison): GT = count of all visible and occluded fruitlets performed manually in the field for each image; RMSE "emphasises larger errors", MAE "average error magnitude per image" — [arXiv 2407.12040](https://arxiv.org/pdf/2407.12040).
- UAV apple DeepSORT with global motion compensation: R² 0.919, RMSE 25.98, MAE 25.0 apples per video — [Agriculture 16(19):2109](https://www.mdpi.com/2077-0472/16/19/2109). Another orchard tracker: R² 0.85, RMSE 1.57 — [Sensors 24(21):7012](https://doi.org/10.3390/s24217012).
- Apple canopy MOT (ASABE): 97 % detection rate; total counts ~10 % average error attributed to false positives — [ASABE](https://elibrary.asabe.org/abstract.asp?aid=49615).
- 3-D orchard mapping dataset: GT counting from video is "challenging due to the possibility of double-counting", so fruits are counted from extracted frames with digital markers, and even then double counting can occur when the same tree appears in multiple frames; some count per batch of trees rather than per tree — [arXiv 2409.19786](https://arxiv.org/pdf/2409.19786).
- Critique: "in many previously reported fruit counting pipelines the ground truth corresponds to fruits manually annotated in the images or videos … it does not necessarily reflect the actual number of fruits in the orchard"; that study validated against in-field manual counting of all fruits after harvest — [Video-based fruit detection and tracking: effects of scanning conditions, Comput. Electron. Agric. 2026](https://www.sciencedirect.com/science/article/pii/S0168169926009282).
- Egyptian citrus (2026): per-video manual counts vs YOLO counts, each fruit gets a persistent ID when crossing a virtual counting line — [Smart Agric. Technol. 2026](https://www.sciencedirect.com/science/article/pii/S2772375526007549).
- CMU 2012 apple yield (Wang et al., ISER): per-row GT by harvest; row-level error mean −2.9 %, SD 7.1 % (thinned red block rows 1–10); green block raw counts mean −29.8 %, SD 8.1 % (occlusion); after calibration mean 1.8 %, SD 11.7 %; whole green block error 1.2 % — [CMU ISER 2012](https://www.ri.cmu.edu/casc/publications/2012%20qi_wang_et_at_iser.pdf).
- Roy & Isler 2016: counting algorithm accuracy 91 %, 81–85 % with segmentation+registration; Dong/Roy/Isler two-side merging 91.98–94.81 % — [Semantic Scholar](https://www.semanticscholar.org/paper/Vision-Based-Apple-Counting-and-Yield-Estimation-Roy-Isler/d4f66091bec0e7197cbe95fee28945936cd34b25); Häni et al. 2020 J. Field Robotics 37(2):263–282 compares detection/counting methods for yield mapping — [ResearchGate](https://www.researchgate.net/publication/335067067_A_comparative_study_of_fruit_detection_and_counting_methods_for_yield_mapping_in_apple_orchards).

### Inferences
- **[inference]** Two-level GT is the defensible design for a thesis: (a) **video-level GT** (annotated tracks on a subset of clips) isolates tracker/counter error from visibility; (b) **field-level GT** (manual count per bed segment, ideally by picking/tagging) measures the end-to-end error that growers care about and yields the visibility calibration factor. Most papers report only (a); reporting both and the ratio between them is a contribution.
- **[inference]** Metric formulas to report per unit i (bed segment / plant group): e_i = (Ĉ_i − C_i)/C_i (signed relative error); MAE = mean|Ĉ_i − C_i|; MAPE = mean|e_i|; RMSE = sqrt(mean(Ĉ_i − C_i)²); R² of Ĉ on C (with slope and intercept, since R² alone hides bias); plus **total relative error** Σ(Ĉ_i − C_i)/ΣC_i, which is what per-field yield uses and is smaller than MAPE because signed errors cancel.

### Gaps
- Precise unit (tree vs sequence) and n for several reported MAEs could not be verified from full texts.
- Per-unit variance of relative error is rarely tabulated; only CMU 2012 gives an SD explicitly (7.1 %, 8.1 %, 11.7 %).

---

## KQ4. Building ground truth for counting from video: count-once rules, occlusion/re-appearance, double counting measurement

### Takeaway
MOTChallenge gives the only well-documented ID rules: keep the ID through occlusion when the path is unambiguous, give a new ID after long ambiguous occlusion or after leaving the frame. Fruit papers do not publish a "count-once" protocol; the standard trick is physical/digital markers when counting manually. No standard "double-counting rate" metric exists, so the thesis should define one explicitly from matched GT identities.

### Cited Findings
- MOT16 annotation protocol: "Keep ID as long as the person is inside the field of view and its path can be determined unambiguously"; "always annotate during occlusions if the position can be determined unambiguously"; after very long occlusion where the path cannot be determined by simple reasoning (e.g., constant-velocity), a new ID is assigned on reappearance; a target fully occluded that never reappears is terminated ("outside of view"); leaving and re-entering the field of view always gets a new ID — [MOT16, arXiv 1603.00831](https://arxiv.org/pdf/1603.00831); [MOTChallenge IJCV 2021](https://link.springer.com/article/10.1007/s11263-020-01393-0).
- Evaluation caveat: MOTChallenge evaluation does not handle re-identification across field-of-view exits — [MOT16](https://arxiv.org/pdf/1603.00831).
- Appearance cues immediately after re-appearance are unreliable and raise IDSW — [QDTrack appendix](https://arxiv.org/pdf/2210.06984).
- Manual GT from video: digital markers on extracted frames to avoid double counting; still error-prone across frames; some count per tree batch — [arXiv 2409.19786](https://arxiv.org/pdf/2409.19786).
- Visibility rule example: an apple counted as "clearly visible" if >50 % of cross-sectional area and >50 % of perimeter are unoccluded — [Häni et al., arXiv 1808.04336](https://arxiv.org/pdf/1808.04336).
- Count-once mechanisms in fruit papers: central counting zone (each tracked berry counted once) — [SAT 2025](https://www.sciencedirect.com/science/article/pii/S2772375525005246); virtual counting line with persistent ID — [SAT 2026](https://www.sciencedirect.com/science/article/pii/S2772375526007549); dual-region strategy — [StraTracker](https://www.sciencedirect.com/science/article/abs/pii/S0168169924009554); line-crossing to avoid over-count — [Agriculture 2025](https://doi.org/10.3390/agriculture15181906).
- Event-level counting F1 and relative count error used as count metrics — [arXiv 2609.13551](https://arxiv.org/pdf/2609.13551).

### Inferences
- **[inference] Proposed GT schema**: MOT-format file with columns frame, track_id, x, y, w, h, conf, class, visibility, **plus** a `fruit_id` (persistent physical identity across the whole bed pass, kept even after the fruit leaves the frame and re-enters, and across the two sides of a bed if both sides are filmed) and a `count_eligible` flag (fruit meets the visibility rule in at least one frame, e.g. ≥50 % visible in ≥3 frames). Compute HOTA/IDF1 on `track_id` (MOTChallenge convention) and count metrics on `fruit_id`.
- **[inference] Count-once rules to write into the annotation guideline**: (1) a fruit is counted once per bed-side pass; (2) a fruit visible in both the outbound and return passes of the same bed is one fruit (handled at the unit level by counting per bed-side, not per video); (3) fruit occluded and re-appearing keeps `fruit_id`; (4) fruits fully occluded in every frame are not count_eligible (define "visible GT count" vs "field GT count"); (5) only fruits whose box centre lies within the plant-row ROI are counted (exclude neighbouring bed).
- **[inference] Double-counting metrics** (define, since none exists): for each GT fruit_id g, let k_g = number of distinct predicted track IDs matched to g (IoU ≥ 0.5 on ≥ m frames, m e.g. 3). Then **Duplicates D = Σ_g max(0, k_g − 1)**; **Misses M = #{g : k_g = 0}**; **Ghosts G = #{predicted tracks matched to no GT fruit}**; **Merges = #{predicted tracks matched to ≥2 GT fruits}** (under-count). Then Ĉ − C = D + G − M − Merges, giving an exact additive decomposition of count error; report **double-count rate = D / C** and **tracks-per-fruit = mean k_g over k_g ≥ 1**. Also report Frag from TrackEval for comparability.
- **[inference]** Annotate GT at reduced frame rate (e.g. 5–10 fps keyframes with interpolation) but keep `fruit_id` continuity; evaluate the tracker at the same frames.

### Gaps
- No fruit paper found publishing its video GT guideline (visibility threshold, re-appearance rule) in enough detail to copy; the rules above are adapted from MOT16.
- No standard "double-count rate" metric exists in the literature surveyed.

---

## KQ5. Inter-annotator agreement for manual counts (and for per-fruit class labels)

### Takeaway
No study was found that quantifies inter-annotator variability of manual fruit counts in orchards/beds. General guidance: use chance-corrected or model-based agreement (ICC for counts with Koo & Li 2016 thresholds; Cohen's κ for class labels), report CIs, and treat disagreement as a signal of ambiguous guidelines (visibility rule, ripeness class boundaries).

### Cited Findings
- IAA rationale: if qualified annotators routinely disagree, the "right answer" is not well defined; low agreement usually points to ambiguous guidelines or class boundaries — [Voxel51 glossary](https://voxel51.com/glossary/inter-annotator-agreement); Label Studio tutorial on measuring IAA and building consensus — [Label Studio](https://labelstud.io/tutorials/how_to_measure_inter_annotator_agreement_and_build_human_consensus).
- Pairwise F1 between annotators' markings used to measure annotator variability — [Lampert et al., arXiv 1307.0426](https://arxiv.org/pdf/1307.0426).
- ICC guideline (Koo & Li, J. Chiropractic Med. 15(2):155–163, 2016): <0.50 poor, 0.50–0.75 moderate, 0.75–0.90 good, >0.90 excellent; judge on the 95 % CI, and report which of the 10 ICC forms (one-way/two-way, agreement/consistency, single/average) was used — [PMC4913118](https://pmc.ncbi.nlm.nih.gov/articles/4913118); thresholds reproduced in [effectsize::interpret_icc](https://r-packages.io/packages/effectsize/interpret_icc).
- Lin's CCC (Lin 1989/2000) combines precision and accuracy, range −1..1; McBride (2005) scale: <0.90 poor, 0.90–0.95 moderate, 0.95–0.99 substantial, >0.99 almost perfect — [MedCalc CCC](https://www.medcalc.org/en/manual/concordance.php); R implementation DescTools::CCC — [CRAN](https://search.r-project.org/CRAN/refmans/DescTools/html/CCC.html).
- Manual field GT itself is error-prone (double counting across frames) — [arXiv 2409.19786](https://arxiv.org/pdf/2409.19786).

### Inferences
- **[inference] Protocol**: have 2 (ideally 3) annotators independently (i) count fruits in the field for a subset of ≥10 bed segments, (ii) annotate fruit_ids on ≥3 clips, (iii) assign one ripeness label per fruit on ≥200 fruits. Report ICC(2,1) absolute-agreement for counts (with 95 % CI), Lin's CCC and the between-annotator MAPE; Cohen's/Fleiss' κ (and weighted κ if ripeness is ordinal) for class labels; pairwise track-level F1 (as in Lampert et al.) for identity annotation. Adjudicate disagreements to form the final GT and record the adjudication rate.
- **[inference]** The between-annotator relative difference gives a floor for achievable system error: if two humans differ by ~5 % on a bed, claiming system error <5 % is not demonstrable with that GT. Report it as the "GT noise floor".

### Gaps
- No published value for human-vs-human fruit count variability in the field or on video; the thesis will have to measure it.

---

## KQ6. Statistical design: equivalence (±10 %), Bland–Altman, bootstrap, sample size, calibration factor stability, fair single- vs two-stage comparison

### Takeaway
No agricultural counting paper found uses TOST or Bland–Altman explicitly; the methodology is imported from measurement-agreement literature. With per-unit relative-error SD of ~7–12 % (the only published values, CMU 2012), a one-sample TOST with ±10 % bounds needs roughly 6–14 bed-segments at 80 % power if bias is ~0, and 11–27 if true bias is ~3 %; Bland–Altman limits need many more units (tens to ~80) to be estimated with useful precision. Per-track class comparison via McNemar needs ~200–500 fruits with a human label each.

### Cited Findings
- TOST: two one-sided tests against bounds −Δ_L and Δ_U, each at α; equivalence concluded when both reject, equivalently when the 90 % CI of the difference lies within the bounds (α = 0.05) — [Lakens 2017, SPPS](https://journals.sagepub.com/doi/10.1177/1948550617697177); [Lakens, Scheel & Isager 2018 tutorial](https://journals.sagepub.com/doi/10.1177/2515245918770963); CI form stated for device evaluations — [FDA equivalence testing](https://www.fda.gov/media/124669/download). Bounds must be pre-specified from the smallest effect of interest; narrower bounds need larger n — [Lakens 2017](https://journals.sagepub.com/doi/10.1177/1948550617697177); [Lakens online book ch. 9](https://lakens.github.io/statistical_inferences/09-equivalencetest.html). TOSTER R package implements power functions — [TOSTER GitHub](https://github.com/Lakens/TOSTER). Example: power 0.9, α 0.05, bounds ±0.5 d, true d 0.1 → n = 108.9 per group — [Lakens tutorial](https://journals.sagepub.com/doi/10.1177/2515245918770963).
- ±10 % region is "a common choice when deviations of that size are presumed to have little practical importance"; ratio form 0.9 < μ_A/μ_B < 1/0.9 (or 1/1.1 < ratio < 1.1) avoids reference-dependence — [Equivalence test (Wikipedia)](https://en.wikipedia.org/wiki/Equivalence_test); primer on TOST for measurement agreement — [PMC5856600](https://pmc.ncbi.nlm.nih.gov/articles/PMC5856600/).
- Count data are often Poisson; a square-root transform may be applied before the t-test, though TOST is fairly robust without it (rapid-microbiology context) — [rapidmicrobio blog](https://www.rapidmicrobio.com/learning-center/rapid-blog/bid/49445/Equivalency-Tests-Rapid-Method-Validation-or-Lies-damn-lies-and-statistics).
- TOST power uses z_{1−β/2} (not z_{1−β}) when the true difference is 0 (two calculators state this; another uses z_{1−β}; conventions conflict) — [NIVEDI calculator](https://nivedi.res.in/Nadres_v2/sample_size_calculator/equivalence.php); [MetricGate TOST docs](https://metricgate.com/docs/sample-size-tost-equivalence-proportions/); exact power for TOST — [Shieh, NYCU](https://ir.lib.nycu.edu.tw/handle/11536/132673).
- Bland–Altman: LoA = mean difference ± 1.96 SD of differences, intended to contain 95 % of future paired differences; CIs of LoA matter, small samples give wide intervals; exact interval procedures preferred (Shieh 2018) — [BMC Med Res Methodol 2018](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0505-y); [Bland 2006 agreement notes](https://eprints.whiterose.ac.uk/id/eprint/3411/1/Bland_2006_Agreement.pdf). MedCalc sample size follows Lu et al. (2016); methods are interchangeable only when both LoA *including their CIs* fall within ±Δ; worked example gives n = 83 — [MedCalc sample size B-A](https://www.medcalc.org/en/manual/sample-size-bland-altman.php); [MedCalc book](https://www.medcalc.org/en/book/bland-altman.php). For repeated measures per unit use the MOVER-based multiple-measurements B-A (Zou 2013) — [MedCalc multiple](https://www.medcalc.org/en/manual/blandaltmanmultiple.php). Halving the CI width of a limit roughly quadruples n — [MetricGate agreement calculator](https://metricgate.com/calculator/sample-size-agreement).
- Calibration factor practice (CMU 2012): zero-intercept regression of computer count on GT over 10 random sections, slope = calibration factor; repeated 100× → mean 1.4, SD 0.1 (green block); 1.7 for red-block rows 11–15; after calibration row-level error 1.8 % mean, 11.7 % SD — [CMU ISER 2012](https://www.ri.cmu.edu/casc/publications/2012%20qi_wang_et_at_iser.pdf). 2026 factor-graph SLAM orchard paper tabulates per-row relative errors 3.20–30.77 % — [Agriculture 16(7):764](https://doi.org/10.3390/agriculture16070764). Scanning conditions (speed, camera side, etc.) affect fruit load estimation and are tested explicitly in — [Comput. Electron. Agric. 2026](https://www.sciencedirect.com/science/article/pii/S0168169926009282) (details not retrievable).
- McNemar: compares two classifiers on the same test items using only the discordant cells B and C; continuity-corrected version (Edwards 1948) most common — [Raschka, arXiv 1811.12808](https://arxiv.org/pdf/1811.12808); [MachineLearningMastery](https://machinelearningmastery.com/mcnemars-test-for-machine-learning/); used as observation-level complement to bootstrap metric comparisons — [arXiv 2607.24160](https://arxiv.org/pdf/2607.24160).

### Inferences (all computations are the researcher's own, from the formulas above)

**Primary estimand [inference]**: per-unit signed relative error e_i = (Ĉ_i − C_i)/C_i for n independent bed segments (unit = one bed side × ~5–10 m, ~100–300 fruits, so that C_i is large enough for e_i to be roughly normal). H0: |mean(e)| ≥ 10 %; H1: −10 % < mean(e) < 10 %. Test: one-sample TOST = check that the 90 % t-CI of mean(e) lies within ±10 %. Pre-register Δ = 10 % (grower-relevant margin). Use the ratio/log form log(Ĉ_i/C_i) with bounds ±log(1.1) if errors are multiplicative.

**Sample size formula used [inference, from the TOST sources]**: n ≈ (z_{1−α} + z_{1−β/2})² σ² / (Δ − |θ|)² for θ = assumed true mean error, iterated with t-quantiles; α = 0.05 (one-sided), Δ = 10 %. Results (n = bed segments):

| σ of per-unit relative error | θ = 0 %, power 0.8 | θ = 3 %, power 0.8 | θ = 0 %, power 0.9 | θ = 3 %, power 0.9 |
|---|---|---|---|---|
| 5 % | 4 | 6 | 5 | 8 |
| 7 % (CMU thinned block) | 6 | 11 | 8 | 13 |
| 8 % (CMU green block raw) | 8 | 13 | 9 | 16 |
| 10 % | 11 | 20 | 13 | 24 |
| 12 % (CMU after calibration) | 14 | 27 | 18 | 34 |
| 15 % | 21 | 41 | 27 | 52 |

Note the formula with z_{1−β/2} applies when θ = 0; for θ ≠ 0 the exact power is asymmetric and slightly different (use TOSTER::power_t_TOST), so the θ = 3 % column is approximate. **Recommendation [inference]**: plan for **≥ 20 bed segments** (e.g., 4 beds × 2 sides × 2–3 segments, or 5–7 beds × 2 sides × 2 segments) across ≥ 2 days; this covers σ up to ~10 % with a 3 % bias at 80 % power and gives some protection against the strawberry-specific σ being larger than the apple values.

**Why "±10 % per bed" is a different claim [inference]**: with σ = 7 % and zero bias, only 84.7 % of individual segments fall within ±10 % (68.3 % if σ = 10 %, 59.5 % if σ = 12 %, 95.4 % if σ = 5 %). The TOST above certifies the *mean* error; to certify that *individual* bed counts are within ±10 % you need Bland–Altman LoA (= bias ± 1.96 σ) inside ±10 %, i.e. σ ≤ ~5 % with zero bias, and the CI of the LoA must also lie inside. Approximate half-width of the 95 % CI of each LoA ≈ 1.96·sqrt(3σ²/n) (Bland & Altman's approximation, as recalled; verify in Bland & Altman 1986): σ = 7 %: n = 10 → ±7.5 pts, n = 20 → ±5.3, n = 30 → ±4.3, n = 50 → ±3.4, n = 83 → ±2.6; σ = 10 %: n = 20 → ±7.6, n = 50 → ±4.8. So B-A with a meaningful CI needs 30–80 segments; present it as descriptive (plot with CIs), with TOST on the mean as the confirmatory test.

**Bootstrap [inference]**: resample bed segments (clusters), B = 2,000–10,000, percentile or BCa CIs for MAPE, total relative error, R², CCC and the double-count rate; for a paired comparison of two pipelines, bootstrap the per-segment difference. With n ≈ 20 clusters, percentile intervals are acceptable but report that coverage is approximate.

**Fruit-level sample sizes [inference]**: to estimate a per-track class accuracy p with 95 % CI half-width h: n = 1.96² p(1−p)/h²: p = 0.9, h = 0.05 → 139 fruits; h = 0.03 → 385; p = 0.8, h = 0.05 → 246; p = 0.5 → 385. McNemar power (formula n = [z_{α/2}√ψ + z_β √(ψ − (p12 − p21)²)]²/(p12 − p21)², ψ = p12 + p21, two-sided α = 0.05): single-frame wrong & track-fused right 10 % vs reverse 5 % → 469 fruits (80 %) / 627 (90 %); 12 % vs 4 % → 194 / 259; 15 % vs 5 % → 155 / 206; 8 % vs 4 % → 587 / 784. **Recommendation**: label ≥ 400 fruits with one human ripeness label each (balanced across classes and beds).

**Per-class counting [inference]**: report the same count metrics per class (ripe / unripe / …) and a class-confusion-corrected count: per-class relative error ê_c = (Ĉ_c − C_c)/C_c; plus "class-conditional count error given correct identity" (fruits matched, class differs) vs "identity-driven error" (duplicates/misses within class). For the thesis's grower use case the ripe-class count is the primary outcome; apply the TOST to the ripe-class error with the same ±10 % margin, and expect σ to be larger than for total count (fewer fruits per unit and class ambiguity at the boundary).

**Calibration factor stability [inference, built on CMU 2012]**: define f_i = C_field,i / Ĉ_i (field count ÷ system count) per segment; fit a mixed model log f_i = μ + bed_j + day_k + ε (or two-way ANOVA with bed and day as factors; with 4–6 beds × 2–3 days). Report: between-bed SD and between-day SD of log f; ICC of f across repeated passes; and an equivalence test of f between days/beds with bounds ±10 % (ratio 0.9–1.1). Use leave-one-bed-out: estimate f on all other beds, apply to the held-out bed, and compute the resulting relative error — this is the honest "deployment" error. CMU's factor varied 1.4 vs 1.7 between blocks, i.e. ~20 %, showing a single global factor can fail; expect ripeness-dependent visibility to make f class-specific in strawberries.

**Repeated passes [inference]**: film each segment ≥ 2–3 times at walking speed (same day) to estimate repeatability: CV of Ĉ_i across passes, ICC(1,1) across passes, and a B-A of pass 1 vs pass 2. Repeatability CV sets the floor for the achievable σ and tells whether averaging two passes is worth it (σ/√2).

**Fair comparison single-stage (detector predicts class) vs two-stage (detector + classifier) with tracking [inference]**: freeze the detection boxes and the track/fruit assignment (same detector weights, same tracker, same hyper-parameters, same input frames), so the only varying component is the per-detection class score source; then (i) per-frame: paired comparison on identical boxes (McNemar on correctness per matched detection, clustered by fruit — or use only one frame per fruit to avoid within-track correlation); (ii) per-track: fuse each pipeline's per-frame class scores with the *same* fusion rule (majority vote / mean softmax / confidence-weighted) and compare per-fruit correctness with McNemar on the same fruits; (iii) per-unit per-class counts with paired bootstrap of the difference in per-class relative error. Also compare "single-frame label at the best-visibility frame" vs "track-fused label" within each pipeline by McNemar (same fruits), which is the direct test of whether tracking improves ripeness labelling. Report both the 2×2 discordance table and the exact binomial McNemar p (preferred when B + C < 25).

### Gaps
- No agricultural counting paper found that applies TOST or Bland–Altman; all examples are clinical/device. The per-unit σ values for strawberries are unknown — the CMU apple values (7–12 %) are the only published per-row SDs found; the first field day should be used as a pilot to re-estimate σ and update n.
- The exact TOST sample-size formula for a continuous one-sample case could not be retrieved from a primary source; conventions (z_{β} vs z_{β/2}) conflict across calculators. Use TOSTER::power_t_TOST for the final numbers.
- The Bland–Altman LoA-CI approximation (sqrt(3σ²/n)) is recalled from Bland & Altman 1986 and was not verifiable here.

---

## KQ7. Annotation tools, effort and cost for video-tracking GT (CVAT, DarkLabel, Label Studio)

### Takeaway
Keyframe-plus-interpolation is what makes video-tracking GT affordable; published effort data are heterogeneous (≈1–2 h per dense single-object track of 1,800 frames; 2.5 h for one track in a 52-s clip with 36 % keyframes; 5–10 s per simple box), and vendor prices run US$0.03–0.25 per object-frame or keyframe. For a dense strawberry clip the researcher's estimate is roughly 2–5 annotator-hours per 30-s bed-side clip at 5–10 fps keyframing.

### Cited Findings
- CVAT per-object times (images): ~5–10 s for a simple box, 30–60 s for a car, 1–3 min for a detailed human; assumes ~60 s/object for occasional annotators; CVAT Online Solo plan US$33/month — [CVAT blog, solo annotation cost](https://www.cvat.ai/resources/blog/calculating-the-cost-of-solo-image-annotation-for-ai-projects); [CVAT pricing](https://www.cvat.ai/pricing/cvat-online); video annotation guide — [CVAT video guide](https://www.cvat.ai/resources/blog/video-annotation-guide).
- Vendor rates: US$0.05–0.25 per object per frame for video object tracking — [HabileData](https://www.habiledata.com/blog/data-annotation-cost/); US$0.03–0.15 per keyframe, with a 30-s/30-fps clip needing only 30–90 keyframes when interpolating — [WeLabelData](https://welabeldata.com/blog/data-labeling-pricing-guide/); frame-by-frame US$0.05–0.25/frame (India) vs US$0.25–1+/frame (USA); object tracking US$3–15 per video-hour (India) vs US$15–60 (USA) — [Precise BPO](https://www.precisebposolution.com/blog/data-labeling-pricing.html).
- EgoTracks: dense single-object annotation of ~1,800-frame videos (5 fps) took roughly 1–2 h per track — [arXiv 2301.03213](https://arxiv.org/pdf/2301.03213). Airshow jet tracking in CVAT: 52-s clip, keyframes on 36 % of frames, 2.5 h for a single track — [Santa-Carmen/video-annotation](https://github.com/Santa-Carmen/video-annotation). Interpolation vendor claim: 1-min video from 30 min to 5 min — [DataTurks (marketing)](https://medium.com/@dataturks/video-object-tracking-working-with-exported-json-4a36500b9431). Mean time per box 15.8 s (Labelme), 8.1 s (VoTT), 4.7 s (ViTBAT semi-automatic) — [WebUAV-3M, arXiv 2201.07425](https://arxiv.org/pdf/2201.07425).
- Label Studio 1.6 added video object tracking with keyframes and automatic box interpolation; export via `--interpolate-key-frames`; SAM-2 video segmentation/tracking template — [Label Studio release](https://labelstud.io/blog/label-studio-1-6-video-object-tracking-and-new-annotation-ui); [template](https://labelstud.io/templates/video_object_detector).
- DarkLabel was used to label strawberry tracking videos manually — [arXiv 2407.12614](https://arxiv.org/pdf/2407.12614); Supervisely used to assign apple IDs manually in a 1,723-frame set — [Comput. Electron. Agric. 2026](https://www.sciencedirect.com/science/article/pii/S0168169926009282). GSD used human annotation of every strawberry with identities — [WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Wen_The_Growing_Strawberries_Dataset_Tracking_Multiple_Objects_With_Biological_Development_WACV_2024_paper.pdf).

### Inferences
- **[inference] Effort model for Huaral clips**: robot at ~1 m/s, a 10-m bed segment ≈ 10 s of video ≈ 300 frames at 30 fps; subsample GT to 5 fps (50 frames); assume ~150 count-eligible fruits visible per segment, each visible ~1.5 s (≈ 8 GT frames at 5 fps), keyframe every other GT frame → ~4 keyframes/fruit → 600 keyframes × ~10 s each (CVAT simple-box rate) ≈ 1.7 h, plus ID bookkeeping/QA (+50 %) ≈ 2.5 h per 10-s segment, i.e. **≈ 15 annotator-hours per minute of dense strawberry video**. With 20 segments this is ~50 h of annotation; two annotators on a 3-segment IAA subset add ~8 h. Cheaper alternative: annotate `fruit_id` only on *sparse* frames (one box per fruit every ~1 s) and compute count metrics (duplicates/misses) by matching predicted tracks to these sparse GT boxes; this gives the count decomposition without dense HOTA GT and cuts effort ~3–4×.
- **[inference]** Tool choice: CVAT (track mode, interpolation, MOT export, free self-hosted) for dense track GT; Label Studio if the team already uses it for image labels (same ripeness taxonomy); DarkLabel is a lightweight Windows tool with no documented recent maintenance — acceptable but harder to audit. Record annotation time per clip in CVAT's analytics and report it in the thesis (annotation cost is itself a result).

### Gaps
- No published per-minute annotation time for fruit MOT datasets (GSD, NeSTCount, StraMOT) was found; the per-minute figure above is an estimate from generic per-box rates.
- DarkLabel documentation not retrieved.

---

## Recommended minimal evaluation protocol (researcher's synthesis) [inference]

1. **Units & campaign**: ≥ 4 beds × 2 sides × 2–3 segments of 5–10 m (≥ 20 segments), filmed 2–3 passes each at walking speed, on ≥ 2 days; field GT per segment (manual count by 2 annotators with tags; ripe/unripe per fruit for ≥ 400 fruits on a subset).
2. **Video GT**: ≥ 6 segments (≈ 1–2 min of video) with dense MOT-format tracks (`track_id` per MOT16 rules + persistent `fruit_id` + visibility + class), annotated in CVAT at 5 fps keyframes; 2–3 clips double-annotated for IAA (ICC/CCC for counts, κ for class, pairwise track F1).
3. **Tracking metrics** (TrackEval): HOTA, DetA, AssA, LocA, IDF1, MOTA, IDSW, Frag, MT/ML — on `track_id`.
4. **Count metrics**: per segment e_i, MAE, MAPE, RMSE, R² with slope/intercept, Lin's CCC, total relative error; count-error decomposition D (duplicates), M (misses), G (ghosts), Merges from `fruit_id` matching; double-count rate D/C; per-class versions.
5. **Confirmatory test**: one-sample TOST (90 % CI within ±10 %) on mean e_i over segments, pre-registered; Bland–Altman plot with CIs on LoA as descriptive evidence; cluster bootstrap CIs (B ≥ 2,000) for all aggregate metrics.
6. **Calibration factor**: f = field count / system count per segment; mixed model / ANOVA by bed and day; leave-one-bed-out error; equivalence of f across days with ratio bounds 0.9–1.1.
7. **Classification**: one human label per fruit; compare single-frame vs track-fused labels, and single-stage vs two-stage pipelines, with McNemar (exact) on the same fruits using frozen detections/tracks; report per-class F1 and per-class count error with paired bootstrap.
8. **Secondary analysis**: Spearman correlation between per-clip Frag / AssA / IDSW and |e_i| across clips and trackers, to answer KQ2 empirically for strawberries.
