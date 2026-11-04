# Implementación práctica en Raspberry Pi 5 + ROS 2: detección + tracking (+ clasificación) para robot de monitoreo de fresas con LiDAR 2D/SLAM

Research notes (2022–2026 sources). Hardware target: Raspberry Pi 5 (4 GB, CPU only; optional AI HAT+/Hailo-8L), 2D LiDAR + slam_toolbox/cartographer in ROS 2, robot speed 0.2–0.5 m/s, camera 30–60 cm from plants, nano YOLO at 640 px.

Access note for the report writer: in this session the hosts docs.ultralytics.com, arxiv.org, raspberrypi.com, discourse.openrobotics.org, fujipress.jp and forum.dexterindustries.com could not be fetched directly. Ultralytics docs were read from the GitHub source mirror (same content); the other sources are cited from search-result snippets and are flagged "snippet only".

---

## KQ1. Per-frame CPU cost of trackers (ByteTrack, BoT-SORT ± GMC ± ReID, OC-SORT, Norfair, SORT) with ~10–30 boxes

### Takeaway
No source publishes ms/frame for these trackers on a Raspberry Pi. The best measured data (BoT-SORT C++ port, on an ARM Jetson NX) shows that the Kalman+Hungarian association itself is cheap (≈2.5 ms for 62 objects on ARM, 0.2 ms on x86) while BoT-SORT's camera-motion compensation (GMC, sparse optical flow) dominates at ≈37 ms/frame on ARM; ReID adds a second network forward pass per frame. For 10–30 boxes, ByteTrack/OC-SORT/SORT/Norfair association cost is negligible relative to YOLO inference on the Pi 5 (tens of ms), so the choice should be driven by GMC/ReID, not by base association cost.

### Cited Findings
- BoT-SORT C++ port benchmark, MOT20-01 (62 objects/frame), "Algorithm Execution Time (ms)", Release build, best of 3. Host x86 (CPU unspecified): with camera-motion estimation (GMC) + motion compensation + Kalman, no ReID: 4.78 ms (209 FPS); without GMC: 0.21 ms (4819 FPS); Kalman only: 0.20 ms. MOT20-05 (226 objects/frame): 6.68 ms with GMC vs 2.05 ms without. — [viplix3/BoTSORT-cpp](https://github.com/viplix3/BoTSORT-cpp)
- Same port on NVIDIA Jetson NX (ARM): MOT20-01 (62 obj): 40.0 ms with GMC (25 FPS) vs 2.64 ms without GMC (378 FPS) vs 2.52 ms Kalman only; MOT20-05 (226 obj): 52.4 ms with GMC vs 15.6 ms without vs 14.8 ms Kalman only. ReID was disabled in all rows, so no ReID timing exists there. — [viplix3/BoTSORT-cpp](https://github.com/viplix3/BoTSORT-cpp)
- BoT-SORT authors caution that FPS comparisons between trackers are hard because speed depends on the device and detection time is generally excluded for tracking-by-detection methods. — [BoT-SORT paper, arXiv 2206.14651 (snippet only)](https://arxiv.org/pdf/2206.14651)
- GPU head-to-head (RTX 3090, YOLOv11 traffic paper, 2026): ByteTrack mean latency 16.79 ms (59 fps), BoT-SORT-ReID 31.11 ms (32 fps), TrackTrack-ReID 31.88 ms (31 fps) — i.e. ReID roughly doubles tracker-stage latency even on a 3090. Whether detector time is included is not clear from the snippet. — [arXiv 2604.04080 (snippet only)](https://arxiv.org/pdf/2604.04080)
- SORT (Bewley et al. 2016): "the tracker updates at a rate of 260 Hz which is over 20x faster than other state-of-the-art trackers" (tracker only, detector excluded; 2016 desktop CPU). — [SORT, arXiv 1602.00763](https://arxiv.org/pdf/1602.00763v1)
- ByteTrack paper: 30 FPS on MOT17 (full pipeline incl. YOLOX detector, single GPU) and uses "only … the simplest similarity computation method Kalman filter in the association step". — [ByteTrack, arXiv 2110.06864 (snippet only)](https://arxiv.org/pdf/2110.06864)
- A 2024 study found that removing the Kalman filter from ByteTrack has "only a marginal effect on the tracking performance", with the effect mostly at high frame rates. — [arXiv 2411.16466 (snippet only)](https://arxiv.org/pdf/2411.16466)
- Roboflow `trackers` release notes: caching Kalman matrices gave "~38% lower BoT-SORT/CBIoU predict cost" (relative, no absolute ms). — [roboflow/trackers releases](https://github.com/roboflow/trackers/releases)
- Ultralytics docs: ByteTrack has "minimal tracker overhead"; BoT-SORT `gmc_method: none` on a static camera saves "a few ms/frame"; ReID "adds inference cost"; `model: auto` ReID (native YOLO backbone features) adds "minimal overhead". No FPS figures given. — [Ultralytics modes/track.md (GitHub source)](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/modes/track.md)
- Ultralytics qualitative guidance: ByteTrack "fast, simple, great default"; BoT-SORT "slower, robust to occlusion (uses appearance features)". — [Ultralytics Academy (snippet)](https://academy.ultralytics.com/courses/yolo-in-production/tracking-with-bytetrack-and-botsort)
- Norfair's project page: "the only thing bounding inference speed will be the detection network feeding detections to Norfair"; it supports a skip-frame approach where the detector skips frames and Norfair extrapolates. No CPU timing published. — [Norfair on PyPI](https://pypi.org/project/norfair)
- BoxMOT separates "detection and embedding generation (slow, GPU-bound)" from "tracker evaluation (fast, CPU-bound)"; its README gives MOT17 HOTA (botsort 69.68, bytetrack 67.68, ocsort 66.44, deepocsort 67.95, strongsort 68.05, boosttrack 69.25, hybridsort 67.31) but no FPS or hardware. — [mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot); [DeepWiki boxmot (snippet, lists BoT-SORT 46 FPS without hardware)](https://deepwiki.com/mikel-brostrom/boxmot/8-evaluation-and-benchmarking)

### Inferences
- Scaling to the Pi 5: the Jetson NX CPU (6x Carmel ARMv8.2 @ up to 1.9 GHz) is roughly Pi 4–Pi 5 class per core; the Pi 5's Cortex-A76 @ 2.4 GHz is likely 1.5–3x faster per thread than the Carmel cores. Scaling the C++ numbers, expect on Pi 5: Kalman+Hungarian for 10–30 boxes well under 1–2 ms (C++) and plausibly 2–5 ms in Ultralytics' pure-Python ByteTrack; BoT-SORT with `sparseOptFlow` GMC plausibly 15–30 ms/frame (because the optical-flow step scales with image size, not box count). These are my estimates, not measurements.
- On a moving robot, a camera-static assumption does not hold, so BoT-SORT's GMC is actually the feature that matters, yet it is also the expensive one. The cheap alternative is to feed the robot's own odometry/SLAM pose as the motion compensation (see KQ6) and keep ByteTrack/OC-SORT.
- ReID on a CPU-only Pi 5 means a second network pass per frame (the Ultralytics ReID encoders use 448 px input); with YOLO already at 70–150 ms/frame this is likely to halve the frame rate. Strawberries are also visually near-identical, so appearance embeddings bring little; GMC/odometry-based motion is the better investment.

### Gaps
- No measured ms/frame for Ultralytics' Python ByteTrack/BoT-SORT, OC-SORT, Norfair or SORT on Pi 5 (or any ARM board) with 10–30 boxes was found. Needs an in-house micro-benchmark (`tracker.update()` on recorded detections).
- No source quantifies the cost of BoT-SORT's `orb` vs `sparseOptFlow` vs `ecc` GMC methods separately.
- The host CPU in the BoTSORT-cpp benchmark is not stated.

---

## KQ2. Ultralytics `track()` end-to-end FPS on Raspberry Pi 5; `persist=True` pattern

### Takeaway
Ultralytics' own Pi 5 benchmarks (FP32, 640 px, inference only, no pre/post-processing) put the fastest nano model at ≈67 ms/frame (YOLO26n NCNN) and YOLO11n at ≈147 ms (ONNX) / ≈85–94 ms (OpenVINO/NCNN, user report), so a realistic end-to-end `track()` loop on the Pi 5 CPU is roughly 6–12 FPS at 640 px; no official number for `track()` itself exists. `persist=True` in a frame loop is the documented pattern for streams of consecutive frames.

### Cited Findings
- Ultralytics Raspberry Pi guide, Pi 5, FP32, imgsz 640, inference time excluding pre/post-processing, Ultralytics 8.4.108: YOLO26n — PyTorch 299.09 ms, TorchScript 353.20, ONNX 125.99, OpenVINO 104.55, LiteRT 123.30, MNN 91.87, NCNN 67.03, ExecuTorch 144.83 ms. YOLO26s — NCNN 172.88 ms, OpenVINO 281.78, ONNX 355.69, PyTorch 848.65 ms. — [Ultralytics guides/raspberry-pi.md (GitHub source)](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/guides/raspberry-pi.md)
- Same page, YOLO26 vs YOLO11 ONNX on Pi 5 (Ultralytics 8.4.14): YOLO11n 147.20 ms (mAP 39.5), YOLO26n 128.42 ms (mAP 40.1); YOLO11s 366.83 ms; YOLO11m 997 ms. "NCNN delivers the best inference performance on Raspberry Pi devices". Tips: Pi OS Lite to save RAM, SSD over SD, optional overclock to 3.0 GHz with cooling. Camera example: picamera2 at 1280x720 RGB888 in a loop calling `model(frame)`. — [Ultralytics guides/raspberry-pi.md](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/guides/raspberry-pi.md)
- Ultralytics community user, Pi 5: ~85 ms YOLO11n with OpenVINO vs ~94 ms with NCNN (≈11.8 vs 10.6 FPS, inference only). Anecdotal. — [Ultralytics community (snippet)](https://community.ultralytics.com/t/raspberry-pi-5-crash-when-running-yolov11n-ncnn/1525)
- LearnOpenCV: "25+ FPS" on Pi 5 with NCNN + quantization but at 240x240 input (not comparable to 640). — [LearnOpenCV YOLO11 on Raspberry Pi](https://learnopencv.com/yolo11-on-raspberry-pi/)
- Community project table: "Decent performance (~10–15 FPS)" for NCNN models on Pi 4/5 incl. display, resolution unspecified. Anecdotal. — [automaticdai/rpi-object-detection](https://github.com/automaticdai/rpi-object-detection)
- Ultralytics tracking docs: six built-in trackers (BoT-SORT, ByteTrack, OC-SORT, Deep OC-SORT, FastTracker, TrackTrack; TrackTrack is now the default); `persist=True` frame loop is the recommended way for video: "Use persist=True only when passing consecutive frames from the same video stream"; multithreaded tracking = one model instance per source thread. — [Ultralytics modes/track.md](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/modes/track.md)
- Ultralytics tracking guide: "effective fps = min(decode fps, model inference fps, write fps)". — [Ultralytics Academy video inference (snippet)](https://academy.ultralytics.com/courses/train-your-first-yolo/video-inference)
- Hailo-8L model zoo (chip throughput, 640x640): yolov8n 202 FPS (batch 1) / 438 (batch 8), yolov11n 157 / 371, yolov8s 110 / 208, yolov5s 124 / 243; hardware mAP yolov8n 36.4, yolov11n 37.5. — [hailo_model_zoo HAILO8L object detection](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst)
- Hailo forum: Model Zoo figures are raw chip throughput, not application FPS; a Pi 5 + Hailo-8L user measured 121.4 FPS at batch 8 for yolov8n; application pipelines with a Pi camera typically run camera-limited at ~30 FPS. Anecdotal. — [Hailo community (snippet)](https://community.hailo.ai/t/yolov8n-real-performance-didnt-expected-from-hailo-model-zoo-documentation/18904); [raspberry.tips (snippet)](https://raspberry.tips/en/raspberrypi-tutorials/raspberry-pi-ai-hat-plus-ai-acceleration)

### Inferences
- End-to-end budget on Pi 5 CPU at 640 px with YOLO26n/YOLO11n NCNN: ≈67–95 ms inference + ≈10–20 ms capture/resize/letterbox/NMS + ≈2–5 ms ByteTrack ≈ 80–120 ms → 8–12 FPS best case with the CPU otherwise idle; with slam_toolbox + LiDAR driver + ROS 2 middleware sharing the four cores, plan on 5–8 FPS. Dropping to imgsz 480 or 416 roughly scales inference with pixel count (≈0.56x / 0.42x) and is the main lever if more FPS is needed.
- With the AI HAT+ (Hailo-8L), detection is no longer the bottleneck (~30 FPS camera-limited); the CPU then only runs tracking + ROS plumbing, which comfortably fits alongside SLAM. Note the Hailo path uses HailoRT/GStreamer pipelines, not `model.track()`; tracking would be done with a separate ByteTrack/Norfair call on the Hailo detections (Ultralytics also lists Hailo as an export target, but no `track()` numbers exist).
- Ultralytics' Python `track()` adds little over `predict()` for ByteTrack; the significant extra costs are BoT-SORT GMC and ReID (KQ1).

### Gaps
- No published end-to-end `model.track()` FPS on Pi 5 (any version) was found; only inference-only benchmarks.
- RAM usage of Ultralytics + PyTorch vs NCNN runtime on the Pi was not found in any source (relevant for the 4 GB budget).
- Hailo-8L + ByteTrack application-level FPS on Pi 5 not found.

---

## KQ3. Minimum frame rate vs robot speed for IoU-based association (640 px, fruit 50–150 px)

### Takeaway
No paper gives this rule for fruit robots; the requirement follows from geometry. For a camera at 30–60 cm, 640 px wide, ~66° HFOV, a strawberry moves ≈250–800 px/s across the image at 0.2–0.5 m/s; keeping the inter-frame displacement below about half the box size (IoU ≥ 0.3 without motion prediction) needs ≈3–8 FPS for 100–150 px fruit at 0.2–0.3 m/s and 60 cm, but 15–30+ FPS for 50 px fruit at 0.5 m/s and 30 cm. The Pi 5 CPU (5–12 FPS) is therefore marginal for small fruit at close range unless the tracker uses motion prediction (Kalman with established velocity, GMC, or odometry).

### Cited Findings
- Ultralytics tracker thresholds that govern association: `match_thresh` (IoU matching), `track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `track_buffer` (raise it "if IDs flicker"), `fuse_score`; ByteTrack has no appearance model or camera-motion compensation. — [Ultralytics modes/track.md](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/modes/track.md)
- Kalman motion prediction matters mainly at high frame rates; at low frame rates removing it changes ByteTrack performance only marginally (i.e., at low FPS the association degrades to near-raw IoU). — [arXiv 2411.16466 (snippet only)](https://arxiv.org/pdf/2411.16466)
- Camera Module 3 optics: 4608x2592 sensor; common video modes 1080p50, 720p120 (official page) vs 1080p50, 720p100, 480p120 (product brief). — [raspberrypi.com product page (snippet)](https://raspberrypi.com/products/camera-module-3); [Camera Module 3 Product Brief](https://cdn.sparkfun.com/assets/c/b/3/a/1/Camera_Module_3_Product_Brief.pdf)
- Global Shutter Camera full mode 1456x1088 @ 60.38 fps (libcamera listing). — [CNX Software review](https://www.cnx-software.com/2023/03/09/review-raspberry-pi-global-shutter-camera/)
- Strawberry robot tracker that uses robot motion instead of a Kalman filter: IBTA "predicted object locations based on the robot's velocity and direction", four RGB cameras on an autonomous robot, outperformed centroid tracking (YOLOv7-pruned detector). — [Liu, Zhou, Lee – YOLOv7 pruning + IBTA (review snippet)](https://themoonlight.io/review/strawberry-detection-and-counting-based-on-yolov7-pruning-and-information-based-tracking-algorithm)

### Inferences (derivation — my arithmetic, not from a source)
- Image scale: with HFOV ≈ 66° (Camera Module 3 standard lens; verify on the Pi datasheet) the field width is W = 2·d·tan(33°) ≈ 0.39 m at d = 0.30 m and ≈ 0.78 m at d = 0.60 m. At 640 px width that is ≈1640 px/m (30 cm) and ≈820 px/m (60 cm). With a GS camera (≈ same lens class on C/CS mount, depends on lens) the figure is similar per 640-px frame.
- Pixel speed = v · px/m: 0.2 m/s → 330 px/s (30 cm) / 165 px/s (60 cm); 0.3 m/s → 490 / 245 px/s; 0.5 m/s → 820 / 410 px/s.
- IoU of a square box of side s shifted by δ along one axis = (s−δ)/(s+δ). Thresholds: IoU ≥ 0.5 ⇒ δ ≤ 0.33·s; IoU ≥ 0.3 ⇒ δ ≤ 0.54·s; IoU ≥ 0.2 (Ultralytics `match_thresh: 0.8` on the 1−IoU cost) ⇒ δ ≤ 0.67·s. Rule of thumb: keep δ below ~0.3–0.5 of the box size so that detection jitter and partial occlusion still leave a margin.
- Minimum FPS = pixel speed / δ_max, with δ_max = 0.5·s (IoU ≈ 0.33):

  | fruit box s | δ_max | 0.2 m/s @60 cm (165 px/s) | 0.3 m/s @60 cm (245) | 0.3 m/s @30 cm (490) | 0.5 m/s @30 cm (820) |
  |---|---|---|---|---|---|
  | 50 px | 25 px | 6.6 FPS | 9.8 FPS | 19.6 FPS | 33 FPS |
  | 100 px | 50 px | 3.3 FPS | 4.9 FPS | 9.8 FPS | 16.4 FPS |
  | 150 px | 75 px | 2.2 FPS | 3.3 FPS | 6.5 FPS | 11 FPS |

- Reading: at 60 cm and ≤0.3 m/s, 5–8 FPS from the Pi 5 CPU is enough for fruit ≥ 100 px and borderline for 50 px fruit; at 30 cm and 0.5 m/s only the Hailo path (≈30 FPS) or odometry-aided prediction keeps 50-px fruit associated by IoU. Because the robot motion is nearly constant and known, feeding the expected pixel shift from odometry (Δx ≈ v·Δt·px/m, with px/m from the known camera-to-plant distance) into the predicted box positions relaxes the requirement by the factor of prediction accuracy (typically to δ_residual of a few px), which is the main argument for KQ6.
- Also a counting constraint: each fruit must appear in ≥ 2–3 frames to confirm a track (Ultralytics `new_track_thresh`/min hits). Field width 0.39–0.78 m at v = 0.5 m/s gives 0.8–1.6 s in view, so even 3 FPS yields ≥ 2 frames; this constraint is weaker than the IoU one.

### Gaps
- No peer-reviewed source states a displacement/box-size rule for ByteTrack-type association; the table above is derived from IoU geometry.
- Camera Module 3 horizontal FOV (66° standard / 102° wide) was not confirmed from a fetched source in this session (raspberrypi.com blocked); verify before using the px/m values.

---

## KQ4. Global vs rolling shutter on Pi (Camera Module 3, HQ, Global Shutter Camera) and motion blur at 0.2–0.5 m/s

### Takeaway
All Raspberry Pi cameras except the Global Shutter Camera (Sony IMX296, 1.6 MP, 1456x1088 @ 60 fps, minimum exposure ≈30 µs) are rolling shutter; the IMX708 (Camera Module 3) reads out at ≈13.4 µs/line (≈35 ms full frame), which at 0.3 m/s and 30 cm produces a few-to-15 px shear in a 640-px frame. Motion blur is independent of shutter type and is controlled by exposure: at 0.3 m/s, 30 cm, 640-px frame, 1 px blur needs ≤ 2 ms (1/500 s) exposure; 60 cm allows ≈4 ms. Both rolling-shutter skew and blur are tolerable for detection at these speeds if exposure is kept short (≤ 1–2 ms, needs good lighting), and a GS camera removes the skew entirely at the cost of 1.6 MP resolution.

### Cited Findings
- Raspberry Pi: every camera released "from 2014's Camera Module 1 to the High Quality Camera and beyond, has used a rolling shutter sensor"; the Global Shutter Camera (IMX296, HQ-camera C/CS mount, $50) "is able to capture rapid motion without introducing rolling shutter artifacts". — [Raspberry Pi news (snippet)](https://www.raspberrypi.com/news/new-raspberry-pi-global-shutter-camera/)
- Jeff Geerling GS camera test: "motion blur is still present (the shutter speed was too slow to prevent that) but the distortion due to rolling shutter is gone". — [Jeff Geerling](https://www.jeffgeerling.com/blog/2023/testing-raspberry-pis-new-global-shutter-camera/)
- GS camera minimum exposure: 29 µs (David Plowman, Raspberry Pi engineer, in the announcement Q&A) / 30 µs (documentation); libcamera patch limits IMX296 minimum exposure to one line ≈ 30 µs. — [Raspberry Pi news comments (snippet)](https://www.raspberrypi.com/news/new-raspberry-pi-global-shutter-camera/); [libcamera-devel patch](https://lists.libcamera.org/pipermail/libcamera-devel/2022-October/034541.html)
- GS camera mode: `1456x1088 [60.38 fps]`; some materials list 1440x1080 active area. — [CNX Software](https://www.cnx-software.com/2023/03/09/review-raspberry-pi-global-shutter-camera/)
- IMX708 (Camera Module 3): 11.9 MP, 4608x2592, 1/2.43" rolling-shutter CMOS; third-party driver changelog states a 13.361 µs line time in every mode (not a Sony datasheet figure). — [Commonlands IMX708 page](https://commonlands.com/pages/image-sensors/imx708); [Espressif esp_cam_sensor_imx changelog](https://components.espressif.com/components/mushbraindave/esp_cam_sensor_imx/versions/0.4.0/changelog)
- Camera Module 3 exposure up to 112 s (long end); video modes 1080p50, 720p120 (official) / 720p100, 480p120 (brief). — [raspberrypi.com (snippet)](https://www.raspberrypi.com/documentation/accessories/camera-module-3.html); [Product Brief](https://cdn.sparkfun.com/assets/c/b/3/a/1/Camera_Module_3_Product_Brief.pdf)
- Alternative GS sensors for Pi: OV9281 1 MP global shutter modules (mono) for Pi 4B/5; Arducam IMX296 colour GS module with M12 lens. — [Core Electronics OV9281](https://core-electronics.com.au/ov9281-global-shutter-camera-raspberry-pi.html); [Arducam IMX296](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/Global-Shutter/1.58MP-IMX296/)
- Element14 blog shows Camera Module 3 working with the Pi 5 AI Kit (Hailo) pipeline. — [element14 (snippet)](https://community.element14.com/products/raspberry-pi/b/blog/posts/using-rpi-camera-module-3-with-rpi5-ai-kit)

### Inferences (my arithmetic)
- Motion blur b = v · t_exp · (px/m). Using the KQ3 scales (1640 px/m at 30 cm, 820 px/m at 60 cm, for a 640-px-wide frame): at 0.3 m/s, 1 px blur ⇒ t_exp ≤ 2.0 ms (30 cm) or 4.1 ms (60 cm); at 0.5 m/s ⇒ 1.2 ms / 2.4 ms. A 1/1000 s exposure gives ≤ 0.8 px blur in all cases. If the native frame (2304 px wide binned mode) is used for classification crops, blur in native pixels is 3.6x larger (e.g., 3.5 px at 0.3 m/s, 30 cm, 2 ms) — still small relative to a strawberry.
- Rolling-shutter skew: with ≈13.4 µs/line, a 2x2-binned 1296-line IMX708 mode reads in ≈17 ms (full 2592 lines ≈35 ms, blanking ignored). Horizontal motion of 490 px/s (0.3 m/s, 30 cm, 640-px frame) during 17–35 ms shifts the bottom of the frame 8–17 px relative to the top; across one 100-px-tall fruit (≈200 sensor lines binned ≈2.7 ms) the shear is ≈1–2 px — negligible for detection and tracking. Skew matters more for geometric projection into the map (KQ6) when using the whole frame; timestamping the frame at mid-exposure and using a short readout mode (lower resolution / higher fps) reduces it.
- Practical recommendation: Camera Module 3 (autofocus, 12 MP, cheap) is adequate at 0.2–0.5 m/s if exposure is fixed at ≤ 1–2 ms with supplemental LED lighting (shaded field/greenhouse conditions) and gain is capped; choose the GS camera if exact per-pixel geometry (3D projection, stereo) or very short exposures with external triggering are needed. Short exposure + high gain raises noise, which hurts small-object detection, so lighting is the real constraint, not shutter type.

### Gaps
- No direct side-by-side blur/skew test of Camera Module 3 vs GS Camera at robot speeds was found.
- IMX708 line time comes from a third-party driver changelog, not Sony/Raspberry Pi documentation.
- Picamera2 sensor-mode table (1536x864@120, 2304x1296@56) could not be verified (raspberrypi.com blocked).

---

## KQ5. ROS 2 packages for YOLO + tracking (+3D) and Pi 5 feasibility

### Takeaway
Two maintained options: mgonzs13/yolo_ros (ROS 2 Foxy→Lyrical, now a single `yolo_node` with pluginlib stages detection → tracking (bytetrack/botsort) → detection3d (RGB-D depth) → debug, ONNX Runtime with a `cpu` provider on aarch64, `target_frame` default `base_link`) and Alpaca-zip/ultralytics_ros (Humble, Python Ultralytics `tracker_node` + `tracker_with_cloud_node` fusing `vision_msgs/Detection2DArray` with a LiDAR `PointCloud2` to emit `Detection3DArray`). Neither publishes Pi 5 performance; both run on CPU, so expect the KQ2 inference numbers to dominate.

### Cited Findings
- yolo_ros: supports Ultralytics-exported ONNX for YOLOv3u/v5u/v8/v9/v10/11/12/26 (detect, seg, pose, OBB, cls); plugins `yolo_ros/DetectionPlugin`, `TrackingPlugin`, `Detect3DPlugin` (named `detection3d`), `DebugPlugin`; `tracker_type` bytetrack (default) or botsort; `provider` auto/tensorrt/cuda/cpu; "runs on x86_64 and aarch64 with the CPU execution provider, no CUDA needed"; `n_threads` auto; `img_width/img_height` default 640/480; depth via `depth_topic` + `depth_info_topic`; `target_frame` default `base_link`; topics `/detections`, `/tracking`, `/detections_3d`, `/debug_image`, `/debug_bb_markers`; distros Foxy, Galactic, Humble, Iron, Jazzy, Kilted, Lyrical. No benchmark numbers; Raspberry Pi not mentioned. — [mgonzs13/yolo_ros](https://github.com/mgonzs13/yolo_ros)
- yolo_ros architecture (older description): TrackingNode publishes `/yolo/tracking` with consistent IDs (`use_tracking` default True); Detect3DNode builds 3D boxes by "filtering depth data from an RGB-D camera using the 2D bounding boxes", `use_3d:=True` (default False), `target_frame` default base_link; last push Feb 26 2026. — [DeepWiki yolo_ros (snippet)](https://deepwiki.com/mgonzs13/yolo_ros/2-system-architecture); [gittrend (snippet)](https://gittrend.io/repo/mgonzs13/yolo_ros)
- ultralytics_ros: `tracker_node` (Ultralytics YOLO detection/segmentation, params `yolo_model, conf_thres, iou_thres, max_det, tracker, device` default `cpu`, `classes`), `tracker_with_cloud_node` ("3D object detection by integrating 2D detections, mask image, LiDAR data, and camera information"; params `cluster_tolerance, voxel_leaf_size, min/max_cluster_size`), messages `ultralytics_ros/YoloResult` (header + `vision_msgs/Detection2DArray` + masks), `vision_msgs/Detection3DArray`, `/detection_cloud`, `/detection_marker`; branches melodic/noetic/humble-devel; no benchmarks; Pi not mentioned. — [Alpaca-zip/ultralytics_ros](https://github.com/Alpaca-zip/ultralytics_ros)
- Camera–LiDAR extrinsics in ROS 2: `ros2_camera_lidar_fusion` v1.0 "calculates the intrinsic and extrinsic matrices between a camera and a LiDAR" from manual pixel↔point matches ("with just 4 points") and includes a node projecting LiDAR points onto the image; whether it accepts `sensor_msgs/LaserScan` was not confirmed. — [ROS Discourse announcement (snippet)](https://discourse.openrobotics.org/t/ros-2-camera-lidar-fusion-package-released/41550)
- A 2D-LiDAR-specific calibration difficulty: "due to the horizontal nature of the LiDAR's scan lines one can obtain the vertical edges, but not necessarily the horizontal ones" (2D-3D correspondence method). — [arXiv 1705.09785 (snippet)](https://arxiv.org/pdf/1705.09785.pdf)
- Real-time Raspberry Pi ROS 2 image updated for Jazzy / Ubuntu 24.04 exists (relevant to Pi 5 deployment). — [ROS Discourse (snippet)](https://discourse.openrobotics.org/t/real-time-raspberry-pi-ros-2-image-updated-for-jazzy-and-24-04/45047)

### Inferences
- yolo_ros is the better fit for the Pi 5: ONNX Runtime C++/pluginlib pipeline avoids a Python PyTorch runtime (RAM and CPU), supports aarch64 CPU, and already does bytetrack + depth-based 3D + `target_frame` transform. Caveat: ONNX on Pi 5 was ≈126–147 ms for a nano model in Ultralytics' table vs 67 ms for NCNN, so ONNX Runtime via yolo_ros will likely be ~2x slower than an NCNN loop unless the ORT build is tuned (thread count, arena, possibly int8).
- ultralytics_ros's 3D node assumes a 3D point cloud (Euclidean clustering with voxel filter); with a 2D LiDAR the laser plane rarely intersects the fruit, so this node is not useful for fruit 3D; use it only as a pattern (vision_msgs + PointCloud2 + camera_info).
- Minimal custom pattern (standard ROS 2 practice, no single citable source): detector node publishes `vision_msgs/Detection2DArray` with the image header stamp; a second node uses `tf2_ros.Buffer.lookup_transform('map', camera_frame, stamp)` (SLAM pose at the image timestamp) plus camera intrinsics and a depth estimate to publish `vision_msgs/Detection3DArray` in `map`, then a fruit-map node merges by 3D distance (KQ6). Timestamp alignment (image stamp vs. `/tf` from slam_toolbox at ~lidar rate) is the practical pain point; use `message_filters` and a tf buffer timeout of ~0.1–0.2 s.

### Gaps
- No published Pi 5 FPS/CPU/RAM figures for yolo_ros or ultralytics_ros.
- Not verified whether `ros2_camera_lidar_fusion` supports 2D `LaserScan`; for 2D LiDAR + camera a manual extrinsic (CAD + checkerboard-on-wall measurement) is usually sufficient and no dedicated ROS 2 tool was found.
- Isaac ROS is GPU/Jetson-only and not applicable to the Pi (not researched further).

---

## KQ6. Fruit mapping with SLAM/odometry pose to count once per fruit (instead of / in addition to 2D tracking)

### Takeaway
The dominant pattern in the literature is: 2D detector → lift detections to 3D using camera pose (SLAM/SfM/LiDAR-inertial) plus depth (stereo, RGB-D, LiDAR, or SfM) → merge detections that fall within a 3D distance threshold in a global "instance manager" to avoid double counting within and across passes; reported count accuracies are 96–97% (pear, apple) with 2D tracking alone over-counting. For a 2D-LiDAR robot, the SLAM pose supplies the camera motion but depth must come from a stereo/RGB-D camera or a size/plane prior.

### Cited Findings
- Lee et al., J. Robotics & Mechatronics 38(2), 2026, horizontal-trellis pear: LiDAR–camera system, YOLO 2D detection + SLAM point clouds, "cumulative moving average tracking filter" for occlusion and a "global 3D instance manager to avoid double-counting on repeated passes"; counting accuracy 96.2%, density MAE 0.64 fruits/m² over a 6 m x 70 m plot. — [Fujipress JRM (snippet only)](https://www.fujipress.jp/jrm/rb/robot003800020449/?full=1)
- Liu et al. 2018, "Robust Fruit Counting: Combining Deep Learning, Tracking, and Structure from Motion": detection+tracking still over-counts (double counts and fruit on trees in the background row); incremental SfM places fruit in 3D and detections "close together in 3D" are flagged as duplicates, size outliers removed; the 3D correction "significantly improves … reducing both the error mean and standard deviation". — [arXiv 1804.00307 (snippet only)](https://arxiv.org/pdf/1804.00307)
- Apple tracking & counting with LiDAR-RGB (arXiv 2310.15138, 2023): deep segmentation + Hungarian tracking, 3.1% total count error over 1790 fruits on 60 trees. — [arXiv 2310.15138 (snippet only)](https://arxiv.org/html/2310.15138v2)
- Gené-Mola et al. 2020 (Comput. Electron. Agric.): Mask R-CNN detections projected onto SfM point clouds, SVM false-positive removal; 2D F1 0.82 → 3D F1 0.88 on 11 Fuji trees / 1455 apples; SfM "processing time is long, so the method is not yet practical for real-time". Code/data MIT. — [recercat (snippet)](https://recercat.cat/handle/10459.1/67802?show=full); [PAgFRUIT news](https://www.pagfruit.udl.cat/en/news/New-scientific-article-published-Fruit-detection-and-3D-location-using-instance-segmentation-neural-networks-and-structure-from-motion-photogrammetry)
- Gené-Mola group, 2025 code release: "fruit detection and tracking using YOLOv5 and ByteTrack, built to be tested jointly with AMIGA robot for in-field fruit counting", ZED stereo camera, adaptable to OAK-D. — [UdL research portal (snippet)](https://portalrecerca.udl.cat/documentos/685699306364e456d3a670ed?lang=fr)
- FruitNeRF (IROS 2024): posed monocular images + per-image fruit segmentation → semantic NeRF → clustered fruit point cloud; "prevents double counting fruit and avoids counting irrelevant fruit" (offline). CountingFruit/FruitLangGS (2025) notes SfM camera poses let 2D detections be fused in 3D to prevent double counting; proposes Gaussian splatting for speed. — [FruitNeRF arXiv 2408.06190 (snippet)](https://arxiv.org/html/2408.06190v1); [arXiv 2506.01109 (snippet)](https://arxiv.org/html/2506.01109v1)
- Tang et al., IEEE Access 2024: camera+IMU+LiDAR SLAM in pear/persimmon orchards, odometry RMSE 6.04 cm / 14.59 cm, tree positioning error 0.57 / 3.23 cm (trees, not fruit). — [DOAJ (snippet)](https://doaj.org/article/0ea5bbfcb5074525946054cf6f4a775c)
- Strawberry video counting, YOLOv8s + ByteTrack (Ogundele et al., Agriculture 2025): mAP@0.5 92.5%, video counts R² 0.914, MAPE 9.52% — i.e., ~10% counting error with 2D tracking only. — [MDPI Agriculture 15(18):1906](https://www.mdpi.com/2077-0472/15/18/1906/htm)
- Strawberry robot tracker using robot velocity/direction for prediction (IBTA) instead of Kalman, four RGB cameras. — [Liu, Zhou, Lee (review snippet)](https://themoonlight.io/review/strawberry-detection-and-counting-based-on-yolov7-pruning-and-information-based-tracking-algorithm)
- yolo_ros already lifts 2D boxes to 3D from RGB-D depth and expresses them in `target_frame`. — [mgonzs13/yolo_ros](https://github.com/mgonzs13/yolo_ros)

### Inferences
- Design for this robot (2D LiDAR + slam_toolbox gives a planar pose (x, y, yaw) in `map`; the camera is rigidly mounted at a known height/pitch):
  1. Depth source options ranked for a 4 GB Pi 5: (a) stereo/RGB-D camera that computes depth on-board (OAK-D/ZED-class; the ZED needs a GPU host, OAK-D does depth on-device) — best; (b) monocular + plane prior: strawberry beds are roughly planar at a known lateral distance from the row path (30–60 cm); intersecting the pixel ray with the bed plane gives XYZ with error ≈ (depth error) ≈ bed-surface variation (several cm), usually enough to merge detections of the same fruit, which are 2–5 cm across and typically >3 cm apart; (c) monocular size prior (fruit diameter ≈ 2–4 cm → depth from box width) is noisier (±30% from size variation) and should only break ties.
  2. Association: transform each detection to `map` at the image timestamp; keep a fruit list; merge if 3D distance < ~3–5 cm (strawberry scale; the pear paper's thresholds were not visible in the snippet) and class/ripeness compatible; count a fruit after ≥ N observations. This replaces 2D ID tracking for counting and is immune to the FPS constraints in KQ3 because association is done in metric space using the known pose.
  3. Keep ByteTrack in 2D as a cheap short-term smoother (ID per frame, confidence fusion of ripeness class) but do the counting in the map-frame fruit list — the literature consistently shows 2D tracking alone over-counts (≈10% MAPE in strawberries, systematic over-count in apples).
  4. Pose-aided 2D prediction: the predicted pixel shift from odometry (KQ3) can be injected as BoT-SORT-style motion compensation at ~zero cost, replacing the sparse-optical-flow GMC that costs tens of ms on ARM.
- Limits: slam_toolbox 2D pose has cm-level jitter and yaw error that, at 0.3–0.6 m range, moves projected fruit by ≈ range x yaw error (1° → 5–10 mm) — acceptable. Across separate passes (hours/days apart) fruit grow and ripen and the map may be re-localized; revisit merging needs a looser threshold or per-plant (not per-fruit) aggregation.

### Gaps
- Full text of the 2026 pear paper (thresholds for the 3D instance manager, robot speed, frame rate, compute platform) could not be fetched.
- No source found that does fruit mapping with a 2D LiDAR SLAM pose on a Raspberry Pi; all cited systems use 3D LiDAR, SfM, or stereo on x86/Jetson hardware.
- No source quantifies depth-from-size-prior accuracy for strawberries.

---

## KQ7. slam_toolbox / cartographer CPU and RAM on Pi 5 with a 2D LiDAR; sharing the CPU with detection

### Takeaway
Only anecdotal numbers exist: one Pi 5 user reports slam_toolbox async mapping at "1% of 1 cpu, 500 MB, +0.4 W" (consuming lidar topics of another robot), a TurtleBot4 on Pi 5 running odometry + LiDAR navigation shows 38.5% total CPU (load 4.37 on four cores), a Pi 4 guide puts slam_toolbox RAM at ~600 MB, and the RPLIDAR driver takes 5–10% of one core. SLAM therefore leaves roughly 2–3 cores and ≈2.5–3 GB for perception on a 4 GB Pi 5; the practical conflict is RAM (PyTorch-based Ultralytics) and thermal throttling, not SLAM CPU.

### Cited Findings
- Pi 5, slam_toolbox async mapping (Docker, ROS 2 Humble, lidar topics from another robot): "uses 1% of 1 cpu, 500MB, +0.4W (base is 2.6W)". Anecdotal, snippet only. — [Dexter Industries forum](https://forum.dexterindustries.com/t/raspberry-pi-5-continued-ros-2-humble-in-docker-on-pios-bookworm/9561)
- TurtleBot4 on Raspberry Pi 5 navigation tuning: "1m load of 4.37, 109.1% demand on RPi 5's four cores, with total CPU usage at 38.5%", with odometry and LiDAR; ROS distro not stated in snippet. Anecdotal. — [ROS Discourse TurtleBot4 tuning (snippet)](https://discourse.openrobotics.org/t/turtlebot4-navigation-tuning-result/53609)
- Pi 4 + RPLIDAR A1 guide: "SLAM Toolbox alone uses ~600 MB RAM", recommends 8 GB where possible. Vendor guide, anecdotal. — [Zbotic](https://zbotic.in/slam-navigation-lidar-mapping-autonomous-robot-india/)
- RPLIDAR `sllidar` driver in Docker: "about 5-10% of one CPU core". Anecdotal. — [Husarion community](https://community.husarion.com/t/closed-rplidar-via-gpio-for-better-performance-why-howto-ros2-docker/1734)
- Pi 4 running Gazebo + RViz + slam_toolbox: all four cores ~95–98%, load ~9 (simulation included, not representative). — [Hello Robot forum](https://forum.hello-robot.com/t/odom-drifting-far-away-when-using-slam-toolbox/1440)
- Unverified vendor table: "~78% average CPU for a Pi 5 running SLAM" vs ~85% on Jetson Nano, no lidar/config stated — low reliability. — [AliExpress wiki](https://www.aliexpress.com/s/wiki-ssr/article/python-r2)
- Ultralytics Pi guide: use Raspberry Pi OS Lite to save RAM, SSD for 24x7, optional overclock to 3.0 GHz CPU with active cooling to avoid throttling. — [Ultralytics guides/raspberry-pi.md](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/guides/raspberry-pi.md)

### Inferences
- Budget on a 4 GB Pi 5: OS + ROS 2 daemon + LiDAR driver ≈ 0.5–0.8 GB; slam_toolbox ≈ 0.5–0.6 GB (grows with map size; a 50 x 10 m greenhouse at 5 cm resolution is small); leaves ≈ 2.5 GB. A PyTorch + Ultralytics process typically needs well over 1 GB (not sourced); an NCNN or ONNX Runtime C++ node (yolo_ros) needs far less, which argues for NCNN/ORT and `.onnx`/`.ncnn` exports rather than `.pt` on the robot.
- CPU: slam_toolbox at mapping rates typical of a 5–10 Hz 2D LiDAR plus scan matching is a fraction of one core on Pi 5; nav2 (if used) adds another ~1 core. Pin YOLO to 2–3 threads (`n_threads` in yolo_ros, or NCNN `num_threads`) and leave one core for SLAM/driver/middleware; expect ~20–30% slower inference than the idle-system benchmarks in KQ2 (≈ 5–8 FPS for nano at 640). Zenoh or CycloneDDS with shared memory reduces image-transport CPU on a single board (general ROS 2 practice; not sourced here).
- Thermal: sustained 4-core inference at 2.4 GHz throttles a Pi 5 without an active cooler; the Ultralytics overclock advice implies cooling is mandatory for stable FPS.
- Cartographer: no Pi 5 numbers were found; it is generally heavier than slam_toolbox at mapping (global optimization) and is less actively maintained in ROS 2 — slam_toolbox is the default recommendation (also by its being the Nav2 default); flag as general knowledge.

### Gaps
- No controlled measurement of slam_toolbox or cartographer CPU/RAM on Pi 5 with a specific LiDAR (RPLIDAR A1/A2/C1, LDS-01) was found; the two Pi 5 data points are forum anecdotes read from snippets.
- No measurement of combined YOLO + slam_toolbox on one Pi 5.
- No RAM figure for Ultralytics/PyTorch vs NCNN on Pi 5 found.

---

## KQ8. Ordinary rolling-shutter RGB camera (Camera Module 3 / HQ / USB webcam), no depth, no global shutter: blur, skew, exposure limits, and "stop-and-shoot"

### Takeaway
With an ordinary rolling-shutter camera at 30–60 cm and 0.2–0.5 m/s, blur in a 640-px frame stays under ~2 px if exposure is ≤ 1–2 ms (1/500–1/1000 s), which requires supplemental lighting; rolling-shutter shear within a single fruit is ≈1–3 px and irrelevant for detection, but whole-frame skew (8–17 px top-to-bottom at 0.3 m/s, 30 cm) matters for geometric projection. Stop-and-shoot removes both constraints at once (exposure can be 10–30 ms at low gain, skew is zero, several frames can be fused per plant) and is the recommended mode whenever the plant pitch and mission time allow; it also simplifies counting to "one plant = one scene" instead of continuous tracking.

### Cited Findings
- All Raspberry Pi cameras except the Global Shutter Camera are rolling shutter (Camera Modules 1–3, HQ). — [Raspberry Pi news (snippet)](https://www.raspberrypi.com/news/new-raspberry-pi-global-shutter-camera/)
- Motion blur is governed by exposure, not shutter type: in a GS-camera test "motion blur is still present (the shutter speed was too slow to prevent that) but the distortion due to rolling shutter is gone". — [Jeff Geerling](https://www.jeffgeerling.com/blog/2023/testing-raspberry-pis-new-global-shutter-camera/)
- IMX708 (Camera Module 3) line time 13.361 µs in every mode (third-party driver changelog, not Sony datasheet); sensor modes listed by a tutorial: 1536x864 @ ~120 fps, 2304x1296 @ ~56 fps, 4608x2592 @ ~14 fps. — [Espressif esp_cam_sensor_imx changelog](https://components.espressif.com/components/mushbraindave/esp_cam_sensor_imx/versions/0.4.0/changelog); [raspberry.tips (snippet)](https://raspberry.tips/?p=2430)
- Agricultural robotics practice (Univ. Bonn, ECMR 2021 Agri workshop slides): moving plants in wind "call for fast shutter speeds to avoid blur", keep the shutter under about 5 ms, which "yields darker images that they offset with onboard panel lights". — [Behnke ECMR 2021 slides (snippet)](https://www.ais.uni-bonn.de/behnke/talks/ECMR_2021_Agri_WS_Sven_Behnke_2021_08_30.pdf)
- Field imagery from a continuously moving robot (TerraSentia patent) "shows high clutter, varying sunlight, motion blur, and occlusion". — [US patent 11789453 (snippet)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11789453)
- Cheap USB webcams: "most cameras will likely support 640x480 at 30FPS (often at 60FPS)"; many cheap UVC cameras have only full-auto exposure, and the OpenCV exposure API is poorly documented and OS-dependent (Linux: auto=0/manual=1 mapping differs from Windows). — [ArPiRobot camera streaming README](https://github.com/ArPiRobot/ArPiRobot-CameraStreaming/blob/main/README.md); [Arducam forum (snippet)](https://forum.arducam.com/t/select-timeout-reading-from-imx283-arducam-usb3-20mp/7924)
- No quantitative study comparing stop-and-go vs continuous imaging for phenotyping/counting was found; stop-and-go systems exist (Pheno-Robot navigates around plants to collect data; Iowa State robot arm probing at 0.4 s planning time) but do not report the throughput cost. — [Pheno-Robot arXiv 2402.09685 (snippet)](https://arxiv.org/pdf/2402.09685); [Iowa State dissertation (snippet)](https://dr.lib.iastate.edu/entities/publication/2dd2d9ea-09ad-4ba9-afc5-e245cf5de738)
- Strawberry plasticulture spacing: 12 in (30 cm) in-row, two rows per bed (SARE project); WSU trial used staggered double rows on 8-in raised beds with 12-in standard spacing; home-garden advice 8–10 in. — [SARE strawberry project (snippet)](https://projects.sare.org/media/pdf/S/t/r/Strawberry-Power-Point-about-SARE-project.pdf); [WSU extension](https://extension.wsu.edu/wam/effect-of-planting-density-on-growth-and-yield-of-sweet-sunrise-strawberry-2022/)

### Inferences (my arithmetic unless noted)
- Scale recap (640-px frame, ~66° HFOV; verify lens): 1640 px/m at 30 cm, 820 px/m at 60 cm. Pixel speed: 0.2 m/s → 330/165 px/s; 0.3 m/s → 490/245; 0.5 m/s → 820/410 (30 cm / 60 cm).
- Exposure limits for blur ≤ b pixels: t_exp ≤ b / (pixel speed).

  | speed, range | 1 px blur | 2 px blur | 3 px blur |
  |---|---|---|---|
  | 0.2 m/s, 60 cm | 6.1 ms | 12 ms | 18 ms |
  | 0.3 m/s, 60 cm | 4.1 ms | 8.2 ms | 12 ms |
  | 0.3 m/s, 30 cm | 2.0 ms | 4.1 ms | 6.1 ms |
  | 0.5 m/s, 30 cm | 1.2 ms | 2.4 ms | 3.7 ms |

  A nano YOLO at 640 px tolerates 2–3 px blur on 50–150 px fruit without measurable mAP loss in my judgement (no source found), so ≤ 4 ms exposure at 30 cm / ≤ 8 ms at 60 cm for 0.3 m/s is the practical bound. Under polytunnel daylight (10–50 klux) 1–4 ms at ISO 100–400 is achievable; in shade/dusk or under dense canopy an LED bar is needed. If the 2304-px native frame is cropped for ripeness classification, blur in native pixels is 3.6x larger (e.g. 7 px at 2 ms, 0.3 m/s, 30 cm) — still small relative to a 150–500 px fruit crop.
- Rolling-shutter skew (IMX708): readout ≈ 13.4 µs x lines of the active mode: 1296-line (2x2 binned) mode ≈ 17 ms; 864-line mode ≈ 11.6 ms; full 2592 lines ≈ 35 ms (blanking ignored). Horizontal motion during readout shears the frame by (pixel speed) x (readout): at 0.3 m/s, 30 cm: 490 px/s x 17 ms ≈ 8 px (binned 1296) or 5.7 px (864-line) top-to-bottom at 640-px scale; at 0.5 m/s, 30 cm: ≈14 px / 9.5 px. Within one 100-px fruit (≈15% of frame height) the shear is ≈1–2 px — negligible for detection and IoU tracking. For projection into the map (KQ10) use the fastest-readout mode (1536x864@120 → 11.6 ms), stamp each frame at mid-exposure + half readout, and optionally correct u by (row/height) x skew. USB webcams (typically 1/30-s readout, ≈33 ms) skew ≈ 2x worse than Camera Module 3 in its binned modes and cannot be fixed in software without per-row timing; prefer the CSI camera.
- Stop-and-shoot assessment: with 30 cm in-row spacing (≈3.3 plants/m per row, ≈6.7/m for a double-row bed), a 1–2 s pause per plant station (0.3 m pitch) gives an effective advance of ≈0.1–0.2 m/s including the 0.3 m/s travel — i.e., stop-and-go roughly halves throughput compared with continuous 0.3 m/s but removes the exposure limit (10–30 ms at low gain), removes rolling-shutter skew, allows burst-averaging/HDR and multiple viewpoints per plant (small lateral jog for pseudo-stereo, KQ10), and makes counting per station trivial (count in the best frame, dedupe across neighbouring stations by map-frame position). Note the Bonn advice: even when stopped, wind-moved plants need ≤ 5 ms exposure outdoors; inside a polytunnel/greenhouse this is less of an issue. Recommended: stop-and-shoot (or "creep-and-shoot" at ≤ 0.1 m/s) for the counting/ripeness pass, continuous 0.3 m/s only for mapping/row traversal.

### Gaps
- No empirical study quantifying detection mAP vs blur pixels for fruit detectors found; the 2–3 px tolerance is judgement.
- IMX708 blanking/readout per mode not verified from Raspberry Pi/Sony documentation.
- No published throughput comparison stop-and-go vs continuous in agricultural robots.

---

## KQ9. Achievable camera frame rate on Pi 5 (libcamera/picamera2 or V4L2) and whether it suffices for IoU tracking

### Takeaway
Capture is not the bottleneck: the Camera Module 3 streams 56–120 fps in its binned modes through libcamera/picamera2 (ISP-scaled to 640 px), the GS camera 60 fps, and UVC webcams 30 (sometimes 60) fps at 640x480; the detector at 5–12 FPS on the Pi 5 CPU (or ~30 FPS with Hailo) sets the effective rate. Because frames can be acquired at 30+ fps, the recommended pattern is to timestamp every frame and run detection on the latest frame (drop, don't queue), using the extra frames only for pose interpolation.

### Cited Findings
- Camera Module 3 sensor modes 1536x864 @ ~120 fps, 2304x1296 @ ~56 fps, 4608x2592 @ ~14 fps; official "common video modes" 1080p50, 720p120 (product page) vs 720p100, 480p120 (product brief). — [raspberry.tips (snippet)](https://raspberry.tips/?p=2430); [raspberrypi.com (snippet)](https://raspberrypi.com/products/camera-module-3); [Product Brief](https://cdn.sparkfun.com/assets/c/b/3/a/1/Camera_Module_3_Product_Brief.pdf)
- Ultralytics' Pi recipe: picamera2 configured at 1280x720 RGB888, `model(frame)` in a loop; alternative `rpicam-vid … --listen -o tcp://127.0.0.1:8888` as a stream source. — [Ultralytics guides/raspberry-pi.md](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/guides/raspberry-pi.md)
- Pi 5 supports two CSI cameras at once; Camera Module 3 connects without an adapter cable on Pi 5 (22-pin). — [raspberry.tips (snippet)](https://raspberry.tips/en/?p=8792)
- UVC webcams: 640x480 @ 30 (often 60) fps; V4L2 `dmabuf` io mode suggested for latency in a Pi streaming tool. — [ArPiRobot README](https://github.com/ArPiRobot/ArPiRobot-CameraStreaming/blob/main/README.md)
- GS camera 1456x1088 @ 60.38 fps. — [CNX Software](https://www.cnx-software.com/2023/03/09/review-raspberry-pi-global-shutter-camera/)
- Effective pipeline fps = min(decode fps, inference fps, write fps). — [Ultralytics Academy (snippet)](https://academy.ultralytics.com/courses/train-your-first-yolo/video-inference)

### Inferences
- With picamera2, request the 1536x864 mode (fast readout, 120 fps) and a 640x360 or 640x480 "main" stream scaled by the Pi 5 ISP; CPU cost of capture is then a few percent of one core (ISP does the scaling; RGB888 copies to Python are cheap at 640 px). Use `sensor_modes`/`FrameDurationLimits` to pin the mode and `ExposureTime`/`AnalogueGain` to fix exposure (KQ8). No measured Pi 5 capture-latency figure was found; typical libcamera pipeline latency is 2–3 frames (judgement).
- Sufficiency for IoU tracking (KQ3 table): the effective tracking rate equals the detector rate, so 5–12 FPS (CPU) is enough at 60 cm/≤0.3 m/s for fruit ≥ 100 px and insufficient for 50-px fruit at 30 cm/0.5 m/s; with Hailo (~30 FPS) all cases in the table are covered. Capturing at 30–120 fps does not help IoU association unless detection also runs at that rate; it helps timestamping and pose interpolation.
- Prefer the CSI camera over USB: manual exposure is reliable through libcamera controls, readout is faster (less skew), and the frame timestamp (`SensorTimestamp`) is exposed for tf2 lookup; cheap UVC cameras often lack manual exposure and precise timestamps.

### Gaps
- No measured picamera2 fps/latency on Pi 5 at 640 px found (raspberrypi.com and forums blocked); the mode table comes from a tutorial snippet.
- No measured CPU cost of picamera2 capture on Pi 5.

---

## KQ10. Monocular projection of detections into the SLAM map frame without depth: options and expected error

### Takeaway
Without a depth sensor, three options exist: (a) fixed camera-to-bed distance / bed-plane intersection (error ≈ bed-geometry variation, ≈5–20 cm in depth, but lateral/along-row error only X·ΔD/D ≈ 0–3 cm, which is what matters for dedup along the row); (b) fruit-size prior (±30% depth error, 10–20 cm — tie-break only); (c) multi-view triangulation of a tracked fruit using the SLAM pose as baseline ("fruit-as-feature SfM", used by Liu et al. 2018 and the monocular mango pipeline), which can reach ≈1–3 cm at 0.3–0.6 m with 10-cm baselines if the pose is good. Along-row position from (a) plus optional (c) is adequate to merge repeated observations of the same strawberry (fruit ≥ 3 cm apart) and to localize per plant.

### Cited Findings
- Monocular fruit counting & mapping (Liu et al., 2018, RA-L/arXiv 1811.01417): single monocular camera; CNN detection of fruits and trunks, tracking with Kalman filter fusing CNN detections and optical flow, counts from "fruit-as-feature structure-from-motion"; on mango the monocular pipeline "achieves counting performance comparable to a state-of-the-art system that uses LiDAR and GPS/INS"; described as "cheap, lightweight, and fast". No fps number in accessible text. — [arXiv 1811.01417 (snippet)](https://arxiv.org/pdf/1811.01417)
- Robust Fruit Counting (2018): detections tracked in 2D, then located in 3D via incremental SfM; detections close in 3D flagged as duplicates; size outliers removed; this "significantly improves the algorithm performance by reducing both the error mean and standard deviation". — [arXiv 1804.00307 (snippet)](https://arxiv.org/pdf/1804.00307)
- Ground-plane monocular geometry sensitivity: 2 m camera height with 2 cm error → ≈1% relative distance error (Ground-VIO); homography ground-plane distance error "grows approximately quadratically with the true distance from the camera"; limits of ground-plane methods: projection-point localization, plane tilt, fixed preset plane fails when the surface is not flat (GPro3D). — [Ground-VIO arXiv 2306.08341 (snippet)](https://arxiv.org/pdf/2306.08341); [arXiv 2303.05192 (snippet)](https://arxiv.org/pdf/2303.05192v1); [GPro3D (snippet)](https://arxiv.org/pdf/2406.03835)
- Camera-pitch/roll estimation on ground robots to ≈1° (Hirano et al.). — [arXiv 2303.05192 (snippet)](https://arxiv.org/pdf/2303.05192v1)
- Strawberry yield-monitoring robot with monocular downward camera (Ren et al., Front. Plant Sci. 2023): AprilTag-based localization 13.0 mm positioning accuracy "at various traveling speeds", keyframe interval chosen from robot speed, frame rate and fruit distance; counting error 2–3% (from the project's slide deck). — [Frontiers 10.3389/fpls.2023.1162435 (snippet)](https://public-pages-files-2025.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1162435/text); [NSF PAR slides (snippet)](https://par.nsf.gov/servlets/purl/10451362)
- yolo_ros Detect3D lifts boxes using RGB-D depth only (no monocular mode). — [mgonzs13/yolo_ros](https://github.com/mgonzs13/yolo_ros)
- slam_toolbox 2D pose quality context: orchard LiDAR-inertial SLAM odometry RMSE 6–15 cm over long runs (3D LiDAR, Tang et al. 2024) — not directly comparable but shows the order of drift. — [DOAJ (snippet)](https://doaj.org/article/0ea5bbfcb5074525946054cf6f4a775c)

### Inferences (my derivations)
- Pinhole scale: f ≈ 320/tan(33°) ≈ 493 px for a 640-px frame at 66° HFOV. Lateral position X = (u − c_x)·D/f.
  - Option (a) fixed distance D (camera looking sideways/down at the bed shoulder, D = 0.3–0.6 m): depth error ΔD = canopy/bed variability ≈ 5–20 cm (fruit hang from bed surface to ~25 cm canopy height); lateral/along-row error ΔX = X·ΔD/D — zero at image centre, ≤ 0.2 m x (0.1/0.3) ≈ 6.7 cm at the frame edge for ΔD = 10 cm at D = 0.3 m; ≈ 1.7 cm at D = 0.6 m with ΔD = 5 cm. Mitigation: use only detections in the central ±30% of the image for mapping (ΔX ≤ 2 cm) — the robot's motion brings every fruit through the centre anyway.
  - Option (b) size prior: depth Z = f·w_real/w_px; strawberry diameter ≈ 2–4 cm (varies with ripeness, cultivar; ±30–40%) → depth error ±30–40% ≈ 10–20 cm at 0.3–0.6 m; useful only to reject background-row fruit or break ties.
  - Option (c) two-view triangulation using SLAM pose: δZ ≈ Z²·δd/(f·B) for pixel disparity error δd and baseline B. Z = 0.4 m, B = 0.10 m (robot moves 10 cm between keyframes), δd = 2 px → δZ ≈ 6.5 mm; pose yaw error δθ = 1° adds δZ ≈ Z²·δθ/B ≈ 28 mm; a 1-cm along-track pose error adds ≈10% of Z (4 cm) if uncorrected. So triangulation reaches ≈2–5 cm with slam_toolbox-grade pose and ≥ 10 cm baselines, degrading sharply for B < 5 cm. This is exactly the "fruit-as-feature SfM" idea in the two 2018 papers, applied with the SLAM pose instead of visual SfM; it needs the detection to be tracked across ≥ 2 keyframes ≥ 10 cm apart (at 0.3 m/s that is ≥ 0.33 s → any detector ≥ 3 FPS suffices). In stop-and-shoot mode, a deliberate 5–10 cm lateral jog at each station provides the baseline.
- Dedup threshold: merge observations within ≈ 3–5 cm in map frame (strawberry size), after projecting with (a) for the first observation and refining with (c). Fruit on the far side of the bed (double-row beds, ~30–40 cm further) separate cleanly by depth with either (a) or (c); background rows (≥ 1 m) are rejected by size prior (b) or by a depth gate.
- Rolling-shutter skew (KQ8) of 5–14 px at frame scale translates to 3–8 mm lateral error at 0.3 m — below the dedup threshold; apply the per-row timestamp correction only if triangulation is used.
- Timestamping: the SLAM pose is published at LiDAR rate (5–15 Hz); interpolate `/tf` (tf2 does this) at the image `SensorTimestamp`; at 0.3 m/s a 20 ms timestamp error = 6 mm.

### Gaps
- No paper found that projects strawberry detections into a 2D-LiDAR SLAM map with monocular geometry; error figures above are derived, not measured.
- Full text of Ren et al. 2023 (camera specs, speeds, keyframe formula) and Liu et al. 2018 (fps, camera) could not be fetched (hosts blocked).
- No published strawberry-diameter distribution used for size priors was retrieved.

---

## KQ11. Fruit-counting work that used ordinary rolling-shutter cameras (and their speed/fps)

### Takeaway
Most published fruit-counting-by-video systems used ordinary rolling-shutter RGB cameras (consumer/industrial RGB, phone/GoPro-class, ZED/OAK RGB streams) on slowly moving vehicles, and reached 2–10% counting error; explicit robot speeds and exposure settings are rarely reported, and none used a global-shutter requirement. This supports the feasibility of a Camera Module 3-class sensor at ≤ 0.3–0.5 m/s.

### Cited Findings
- Strawberry video counting, YOLOv8s + ByteTrack (Ogundele et al., Agriculture 2025): mAP@0.5 92.5%, video count R² 0.914, MAPE 9.52%; camera/speed not stated in accessible text. — [MDPI Agriculture 15(18):1906](https://www.mdpi.com/2077-0472/15/18/1906/htm)
- Strawberry robot, four RGB cameras, constant-speed robot, IBTA tracker using robot velocity/direction, detection 163.9 FPS on the paper's GPU, 89.1% accuracy (Liu, Zhou, Lee 2024). — [arXiv 2407.12614 (snippet)](https://arxiv.org/pdf/2407.12614)
- Strawberry yield monitoring AMR with downward monocular camera, AprilTag localization (13 mm), counting error 2–3%, keyframe interval tied to speed/fps/distance (Ren et al. 2023). — [Frontiers (snippet)](https://public-pages-files-2025.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1162435/text); [NSF PAR (snippet)](https://par.nsf.gov/servlets/purl/10451362)
- Monocular mango counting (Liu et al. 2018) matching a LiDAR+GPS/INS system — single ordinary camera. — [arXiv 1811.01417 (snippet)](https://arxiv.org/pdf/1811.01417)
- Apple/orange counting with detection + tracking + SfM, ordinary RGB video (Liu et al. 2018). — [arXiv 1804.00307 (snippet)](https://arxiv.org/pdf/1804.00307)
- Gené-Mola group AMIGA robot: YOLOv5 + ByteTrack with ZED stereo (rolling-shutter RGB sensors), adaptable to OAK-D. — [UdL portal (snippet)](https://portalrecerca.udl.cat/documentos/685699306364e456d3a670ed?lang=fr)
- Commercial: Saga Robotics Thorvald uses "2 cameras scan every row, counting flowers and fruits while collecting data about ripeness", farm-wide counts every 4 days; no accuracy or camera spec published. — [Saga Robotics (snippet)](https://www.sagarobotics.com/strawberries); [FreshPlaza (snippet)](https://www.freshplaza.com/africa/article/9748129/uv-c-robot-expanded-to-count-flowers-and-strawberries/)

### Inferences
- None of these systems report needing a global shutter; the two strawberry robots moved continuously at "constant speed" with multi-camera rigs, and the AMR work explicitly derived a keyframe interval from speed x fps x distance — the same geometry as KQ3. The 2–3% error of the pose-aided (AprilTag-localized) system versus ≈10% MAPE for pure ByteTrack video counting supports using the SLAM pose for association (KQ6/KQ10).

### Gaps
- Robot speeds, exposure times and camera models for the strawberry papers were not retrievable (arXiv, Frontiers, NSF PAR blocked).
- No paper found that reports rolling-shutter artefacts as a failure cause in fruit counting.

---

## KQ12. Triggering stops: can a 2D LiDAR detect individual strawberry plants? Alternatives and what field robots do

### Takeaway
A horizontally scanning RPLIDAR-class 2D LiDAR reliably sees the bed/row edge and row ends (used for strawberry-greenhouse row following), but individual-plant segmentation with 2D LiDAR has only been demonstrated for well-separated upright plants (maize) by tilting the scanner and assembling scans into 3D with precise positioning (70.7% detection, 3.6 cm RMSE); for a dense strawberry canopy on a mulched bed it is not a dependable per-plant trigger. Use fixed-pitch stops from the SLAM/odometry pose (plants at ≈30 cm spacing) or camera-based triggers instead.

### Cited Findings
- AgriEco strawberry greenhouse robot: 2D LiDAR point cloud processed "to estimate the robot's heading and lateral offset relative to crop rows" and detect row ends to switch rows; used for guidance while spraying. Individual plants are not detected. — [AIMS Electronics & Electrical Eng. 2022 (snippet)](https://aimspress.com/article/doi/10.3934/electreng.2022019)
- Univ. Hohenheim: 2D scanner tilted 30° on a small robot between maize rows, scans assembled into 3D using total-station positioning; single-plant detection rate up to 70.7% with 3.6 cm RMSE, "precise enough to allow single plant treatment". — [Hohenheim (snippet)](https://www.uni-hohenheim.de/organisation/publikation/using-assembled-2d-lidar-data-for-single-plant-detection)
- Strawberry 2D-LiDAR phenotyping on a conveyor (stationary): LiDAR parameters correlate linearly with leaf area (individual plants in a controlled setup, not field row following). — [TU Berlin (snippet)](https://depositonce.tu-berlin.de/items/183e2d9a-475b-4ad4-8eef-6f880ddd2be6/full)
- Low-cost 2D LiDAR under-canopy navigation in corn/sorghum (TerraSentia-class) — row following, not plant counting. — [MDPI AgriEngineering 4(4):53 (snippet)](https://www.mdpi.com/2624-7402/4/4/53)
- Vineyard: forward-inclined 2D LiDAR builds row detection in 3D "thanks to the robot motion". — [HAL INRAE (snippet)](https://hal.inrae.fr/hal-04072510/document)
- Plant spacing: 30 cm in-row, double rows per bed in plasticulture (SARE; WSU). — [SARE (snippet)](https://projects.sare.org/media/pdf/S/t/r/Strawberry-Power-Point-about-SARE-project.pdf); [WSU](https://extension.wsu.edu/wam/effect-of-planting-density-on-growth-and-yield-of-sweet-sunrise-strawberry-2022/)
- Stop-and-go practice: Pheno-Robot autonomously positions around plants for high-quality data; TerraSentia-style continuous imaging suffers blur/occlusion; Thorvald counts continuously while performing UV-C passes. — [arXiv 2402.09685 (snippet)](https://arxiv.org/pdf/2402.09685); [US 11789453 (snippet)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11789453); [Saga Robotics (snippet)](https://www.sagarobotics.com/strawberries)
- Strawberry AMR counting used AprilTags along the rows as the positioning reference (13 mm) and speed-dependent keyframes rather than LiDAR plant detection. — [Frontiers (snippet)](https://public-pages-files-2025.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1162435/text)

### Inferences
- Why 2D LiDAR fails per-plant on strawberries: the scan plane at a fixed height intersects a continuous leaf canopy (plants 20–30 cm apart with 25–40 cm canopy diameters overlap) and the bed shoulder, returning a nearly continuous contour; RPLIDAR A1/A2 angular resolution (≈1° → 5 mm at 0.3 m) and ±1–2 cm range noise is fine geometrically, but there is no gap between plants to segment. It does give a very reliable bed-edge distance for keeping the camera at a constant 30–60 cm (which also fixes D for KQ10 option (a)) and detects row ends.
- Recommended trigger hierarchy: (1) pose-based: start the station sequence at the first plant (detected by camera) and stop every 0.30 m of along-row distance from the slam_toolbox pose or wheel odometry (a 1–2% odometry error over a 30 m row is 0.3–0.6 m ≈ 1–2 plant pitches, so re-anchor on the camera every few plants or on row ends from LiDAR); (2) camera-based: stop when the detector reports new fruit IDs/clusters entering the central image band or every N pixels of image displacement (N ≈ frame width x 0.6 for ~40% overlap) measured by ByteTrack mean box motion or sparse optical flow; (3) plant-level stops by crown detection (train a "crown"/"plant" class or use a green-mass vertical profile in the image) if per-plant attribution is needed. Option (2) with image-displacement spacing is what the keyframe-interval formulation in the strawberry AMR work does continuously; stopping instead of keyframing is a small change.
- What field robots do: commercial monitoring (Thorvald, Bitwise) images continuously with multiple cameras and lights and accepts some blur/over-count, because throughput matters; research phenotyping platforms (Pheno-Robot, arm-based probing) stop per plant for quality. For a Pi 5-class system the stop-and-shoot approach trades throughput for lower compute and better image quality and is the better fit.

### Gaps
- No paper found that segments individual strawberry plants in a field bed with 2D LiDAR; the negative conclusion is inferred from the row-following literature and canopy geometry.
- No source documents a camera-displacement-triggered stop policy explicitly; it is an adaptation of the keyframe-interval rule.

---

## Consolidated recommendations (synthesis — inferences unless cited above)

1. Detector/runtime: nano model (YOLO26n/YOLO11n) exported to NCNN (67 ms on Pi 5 for YOLO26n, Ultralytics benchmark) or run via yolo_ros ONNX-CPU (slower, ≈126–147 ms, but C++ and ROS-native). Expect 5–10 FPS end-to-end on CPU with SLAM running; ≈30 FPS camera-limited with AI HAT+/Hailo-8L (chip 157–202 FPS for nano models).
2. Tracker: ByteTrack (or OC-SORT) with `persist=True`; avoid BoT-SORT `sparseOptFlow` GMC and ReID on the CPU path (GMC ≈ 37 ms/frame on Jetson NX ARM; ReID doubles tracker-stage latency on a 3090). Use odometry-predicted pixel shift as motion compensation.
3. Frame rate vs speed: keep inter-frame displacement ≤ 0.5 box size; for 60 cm range and ≤ 0.3 m/s, 5–8 FPS suffices for fruit ≥ 100 px, borderline at 50 px; at 30 cm / 0.5 m/s plan on ≥ 16–33 FPS or pose-aided prediction.
4. Camera (ordinary rolling-shutter RGB, per user constraint): Camera Module 3 over a USB webcam (manual exposure via libcamera, 11.6–17 ms readout in binned modes, hardware timestamps, 56–120 fps modes). Moving at 0.3 m/s: fix exposure ≤ 2 ms (30 cm) / ≤ 4 ms (60 cm) for ≤ 1 px blur at 640 px, add LED lighting; rolling-shutter shear is ≈1–2 px within a fruit (negligible for detection) and 6–14 px across the frame (correct only if triangulating). Capture rate is never the bottleneck; the detector is.
5. Stop-and-shoot is recommended for the counting/ripeness pass: stop ≈1–2 s every 0.30 m (plant pitch) using the SLAM/odometry pose, re-anchored by camera events; this removes exposure/skew constraints, allows multi-frame fusion and a 5–10 cm lateral jog for pseudo-stereo, and halves throughput vs 0.3 m/s continuous. 2D LiDAR cannot be relied on to detect individual strawberry plants (only bed edges/row ends); use it to hold the camera at a constant bed distance.
6. Counting without depth: transform detections with tf2 at the image stamp; first estimate position from the fixed camera-to-bed distance (along-row error ≤ 2 cm for central detections), refine by two-view triangulation with the SLAM pose (≈2–5 cm at 10 cm baseline), reject background with a size gate; merge within 3–5 cm; count after N views. Literature: 2D-tracking-only ≈10% MAPE (strawberry), pose/3D-aided 2–4% error.
7. Resources: SLAM ≈ 0.5–0.6 GB and a fraction of a core on Pi 5 (anecdotal); perception gets 2–3 cores and ≈2.5 GB — use NCNN/ORT rather than PyTorch, Pi OS Lite, active cooling.
