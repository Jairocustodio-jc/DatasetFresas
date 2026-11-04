# Raspberry Pi 5 (4 GB) inference cost and deployment options for a two-stage pipeline (YOLO-nano detector + small crop classifier), with LiDAR+SLAM sharing the CPU

Scope note on sources: during this research session `docs.ultralytics.com`, `raspberrypi.com`, `arxiv.org`, `community.ultralytics.com`, `community.hailo.ai`, `wiki.seeedstudio.com`, `learnopencv.com` and `developer.arm.com` could not be fetched directly (DNS blocked in the sandbox). Wherever possible the *same documents* were read from their canonical GitHub source files (Ultralytics docs markdown at pinned tags, Hailo Model Zoo `.rst` tables, ncnn `benchmark/README.md`, Arm model cards on Hugging Face). Items that could only be confirmed via search-result snippets are explicitly flagged "(snippet only)". All times are inference-only unless stated otherwise.

---

## Key question 1 — Ultralytics official Raspberry Pi 5 benchmarks (YOLOv8n / YOLO11n / YOLO26n at 640 px, per export format) and imgsz-scaling data

### Takeaway
On a stock Pi 5 at FP32/640 the nano detectors cost roughly 67–95 ms/image with NCNN (the fastest format every generation), 105–200 ms with ONNX Runtime/OpenVINO, and 300–510 ms with PyTorch; each Ultralytics generation has trimmed NCNN latency (YOLOv8n 94 ms → YOLO11n 88 ms → YOLO26n 67 ms). Ultralytics publishes no Pi 5 numbers at imgsz 1024 and no thread-count information; nothing above 640 exists in any official table.

### Cited Findings

**Current docs page (Ultralytics 8.4.108, YOLO26, Pi 5, FP32, imgsz=640, "inference time does not include pre/post-processing")** — [Ultralytics docs source, main](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/guides/raspberry-pi.md)

| YOLO26n format | Size on disk (MB) | mAP50-95(B) (coco128) | Inference (ms/im) |
|---|---|---|---|
| PyTorch | 5.3 | 0.4760 | 299.09 |
| TorchScript | 9.8 | 0.4734 | 353.20 |
| ONNX | 9.5 | 0.4734 | 125.99 |
| OpenVINO | 9.6 | 0.4734 | 104.55 |
| LiteRT (TFLite) | 9.8 | 0.4730 | 123.30 |
| MNN | 9.4 | 0.4749 | 91.87 |
| NCNN | 9.4 | 0.4784 | **67.03** |
| ExecuTorch | 9.4 | 0.4772 | 144.83 |

- YOLO26s on the same setup: NCNN 172.88 ms, OpenVINO 281.78 ms, ONNX 355.69 ms, PyTorch 848.65 ms — [same source](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/guides/raspberry-pi.md)
- ONNX head-to-head at 640 (Ultralytics 8.4.14, Pi 5): YOLO26n 128.42 ms (mAP 40.1) vs YOLO11n 147.20 ms (mAP 39.5); YOLO26s 352.84 vs YOLO11s 366.83 ms — [same source](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/guides/raspberry-pi.md)
- The page states only n and s scales were benchmarked because larger models "are too big to run on the Pis and do not give decent performance"; it recommends NCNN as "best inference performance" on Raspberry Pi because it is optimised for ARM; it does **not** state batch size, thread count, Python version, cooling or PSU used for the benchmark, and does not mention the AI HAT/Hailo or AI Camera — [same source](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/guides/raspberry-pi.md)
- `format=litert` export does not run on the Pi (Linux x86_64 / macOS only); export elsewhere and copy the `.tflite` — [same source](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/guides/raspberry-pi.md)
- Optional Pi 5 overclock documented: `arm_freq=3000`, `gpu_freq=1000`, `force_turbo=1` (2.4→3.0 GHz), with the warning that proper cooling is needed and to step down 100 MHz if unstable; Pi 5 max draw listed as 5 A @ 5 V (PD) — [same source](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/guides/raspberry-pi.md)

**YOLO11n table (Ultralytics v8.3.39, Pi 5, FP32, imgsz=640)** — [Ultralytics docs source at tag v8.3.40](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.40/docs/en/guides/raspberry-pi.md)

| YOLO11n format | Size (MB) | mAP50-95(B) | ms/im |
|---|---|---|---|
| PyTorch | 5.4 | 0.6100 | 405.238 |
| TorchScript | 10.5 | 0.6082 | 526.628 |
| ONNX | 10.2 | 0.6082 | 168.082 |
| OpenVINO | 10.4 | 0.6082 | 81.192 |
| TF SavedModel | 25.8 | 0.6082 | 377.968 |
| TF GraphDef | 10.3 | 0.6082 | 487.244 |
| TF Lite | 10.3 | 0.6082 | 317.398 |
| PaddlePaddle | 20.4 | 0.6082 | 561.892 |
| MNN | 10.1 | 0.6106 | 112.554 |
| NCNN | 10.2 | 0.6106 | **88.026** |

- YOLO11s (same run): NCNN 194.858 ms, OpenVINO 179.324 ms, ONNX 324.17 ms, PyTorch 1011.60 ms — [same source](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.40/docs/en/guides/raspberry-pi.md)

**YOLOv8n table (Ultralytics 8.2-era docs, Pi 5 and Pi 4, FP32, imgsz=640; ultralytics version not stated on the page)** — [Ultralytics docs source at tag v8.2.100](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.2.100/docs/en/guides/raspberry-pi.md)

| YOLOv8n format | Pi 5 ms/im | Pi 4 ms/im (older board, reference only) |
|---|---|---|
| PyTorch | 508.61 | 1068.42 |
| TorchScript | 558.38 | 1248.01 |
| ONNX | 198.69 | 560.04 |
| OpenVINO | 704.70 (anomalously slow in that release) | 534.93 |
| TF SavedModel | 367.64 | 816.50 |
| TF GraphDef | 473.22 | 1007.57 |
| TF Lite | 380.67 | 950.29 |
| PaddlePaddle | 703.51 | 1507.75 |
| NCNN | **94.28** | 414.73 |

- YOLOv8s Pi 5: NCNN 211.26 ms, ONNX 451.37 ms, PyTorch 969.49 ms; Pi 4 NCNN 1042.39 ms — [same source](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.2.100/docs/en/guides/raspberry-pi.md)
- Independent C++ NCNN measurement (Qengineering, inference only, excludes frame grab/post-processing/drawing; column headers "RPi 4 1950" and "RPi 5 2900" i.e. overclocked MHz; thread count/precision/ncnn version not stated): YOLOv8 nano 640×640 = **20.0 FPS on Pi 5 (≈50 ms)** vs 3.1 FPS on Pi 4; YOLOv8 small 11.0 FPS (≈91 ms); YOLOv5 nano 13.6 FPS; YOLOv6 nano 15.8 FPS; YOLOv7 tiny 14.4 FPS; YoloX nano 416 38.6 FPS; NanoDet 320 43.2 FPS — [Qengineering README](https://raw.githubusercontent.com/Qengineering/YoloV8-ncnn-Raspberry-Pi-4/main/README.md)
- Community measurement (Pi 5, Raspberry Pi OS Bookworm, Python 3.11.2, Ultralytics v8.2.80, YOLOv8n, validation on coco128 averaged over 10 iterations): NCNN 85.06 ms (4 GB) / 105.88 ms (simulated 2 GB); ONNX 173.32 / 198.23 ms; PyTorch 670.90 / 710.33 ms (snippet only; thread is DNS-blocked) — [Ultralytics community thread](https://community.ultralytics.com/t/raspberry-pi-new-cheaper-raspberry-pi-released/208)
- Server-class reference for the same models (Ultralytics perf macros, "CPU ONNX", CPU type and ORT version not stated in the macro file): YOLO26n 38.9 ± 0.7 ms at 640, mAP 40.9 (e2e 40.1), 2.4 M params, 5.5 B FLOPs; YOLO26s 87.2 ms — [yolo-det-perf macro](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/macros/yolo-det-perf.md). YOLO26 page claims "up to 43% faster CPU ONNX inference" than YOLO11n on an Intel Xeon @ 2.00 GHz, and documents the NMS-free one-to-one head (`nms=False` output `(N,300,6)`, max 300 detections) vs the default one-to-many head `(N, nc+4, 8400)` that needs NMS — [yolo26.md](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/models/yolo26.md)
- A search for Pi 5 YOLO11n/NCNN at imgsz=1024 found no measured number anywhere — [search result summary citing the Ultralytics guide](https://docs.ultralytics.com/guides/raspberry-pi)

### Inferences
- Across three generations, NCNN on Pi 5 is consistently 1.3–2× faster than OpenVINO/ONNX and 4–5× faster than PyTorch for nano models at 640; a working planning figure for a nano detector at 640 with NCNN FP32 is **~70–95 ms/image (≈10–14 FPS, inference only)**, or ~50 ms if the board is overclocked to 2.9–3.0 GHz with good cooling (Qengineering number).
- The Pi 4 → Pi 5 speed-up on identical software is ~4.4× for YOLOv8n/NCNN (414.73 → 94.28 ms) and ~2.8× for ONNX; older Pi 4 figures from the literature can therefore be divided by roughly 3–4.5 to estimate Pi 5.
- The Ultralytics Python benchmark numbers already include a thin Python wrapper around NCNN; the C++ Qengineering figure (~50 ms, overclocked) suggests ~20–30% further headroom from a native pipeline and the overclock.
- Because pre/post-processing is excluded from every Ultralytics figure, end-to-end per-frame time (letterbox resize, NMS for v8/v11, drawing) will be higher; YOLO26's NMS-free head removes the NMS step, which is relevant on a CPU shared with SLAM.
- Note the mAP columns in the Pi tables are measured on coco128 (different values per release: 0.60–0.61 for YOLO11n, 0.47–0.48 for YOLO26n) and are not comparable to COCO val mAP.

### Gaps
- No official Pi 5 benchmarks exist for YOLOv9t, YOLOv10n or YOLO12n in any export format (Ultralytics only benchmarked v8, 11 and 26 n/s scales on the Pi).
- No official imgsz=1024 or any imgsz-scaling table for Pi 5; thread count and batch size used by Ultralytics are not documented.
- The Ultralytics `benchmark` mode's thread default for NCNN on the Pi could not be confirmed from the fetched pages.

---

## Key question 2 — Measured numbers for yolo11n-cls / yolov8n-cls, MobileNetV3, EfficientNet-B0, ResNet18 at 224 px on Pi 5 CPU

### Takeaway
A 224-px MobileNetV3 costs ~7 ms (4 threads) to ~12 ms (1 thread) in FP32-class NCNN on a Pi 5, EfficientNet-B0 ~12–24 ms, ResNet18 ~20–30 ms; with INT8 + XNNPACK/KleidiAI (ExecuTorch), MobileNetV3-Small drops to ~2.1 ms and ResNet-18 to ~8.8 ms. No Pi 5 measurement of yolo11n-cls / yolov8n-cls / yolo26n-cls was found; only server CPU figures exist.

### Cited Findings

**ncnn official `benchncnn` on Raspberry Pi 5 (BCM2712), loop_count=10, powersave=0, gpu_device=-1, cooling_down=1; ncnn benchmark model set (mobilenet_v3 variant/width not specified in the table; FP16 settings not stated in the text), avg ms** — [Tencent/ncnn benchmark README](https://raw.githubusercontent.com/Tencent/ncnn/master/benchmark/README.md)

| Model (224-px class) | Pi 5, 4 threads (min/avg ms) | Pi 5, 1 thread (avg ms) | Pi 4, 4 threads (avg) | Pi 4, 1 thread (avg) |
|---|---|---|---|---|
| mobilenet (v1) | 9.00 / 33.88 (noisy max 72.98) | 20.30 | 61.17 | 124.32 |
| mobilenet_v2 | 10.46 / 10.52 | 16.38 | 61.88 | 93.48 |
| mobilenet_v3 | 7.30 / **7.35** | 11.80 | 47.53 | 72.81 |
| efficientnet_b0 | 12.28 / **12.33** | 24.20 | 74.19 | 129.73 |
| resnet18 | 19.84 / **19.96** | 29.68 | 123.12 | 191.10 |
| squeezenet | 6.74 / 7.38 | 11.87 | 46.65 | 73.96 |
| shufflenet_v2 | 3.37 / 3.39 | 7.14 | 24.56 | 42.12 |
| yolov4-tiny (416) | 44.45 / 46.29 | 68.02 | 207.22 | 372.99 |
| nanodet_m (320) | 11.01 / 11.18 | 19.69 | 68.38 | 103.45 |

**Arm's ExecuTorch 1.1.0 model cards, reference device Raspberry Pi 5 (Cortex-A76 ×4 @ 2.4 GHz, 8 GB, Raspberry Pi OS 64-bit Debian 13 "Trixie"), CPU backend XNNPACK + KleidiAI, batch 1, 224×224 float32 input tensor, 10 warm-up + 100 runs; thread count not stated on the MobileNetV3/ResNet cards (the sibling YOLOv5s card states 1 thread — snippet only)**
- MobileNetV3-Small: FP32 baseline p50 3.083 ms (324 FPS), INT8 static PTQ p50 **2.12 ms** (471.68 FPS), p90 2.146 ms; model 9.74 → 2.639 MB; peak RAM 16.56 → 10.31 MB; load time 6.5 ms; INT8 scheme per-channel symmetric weights / per-tensor affine activations, calibrated on 2,000 ImageNet-1k images, some layers kept FP32 — [Arm/mobilenet-v3-small-int8-xnnpack-executorch](https://huggingface.co/Arm/mobilenet-v3-small-int8-xnnpack-executorch)
- ResNet-18: FP32 p50 31.563 ms (31.68 FPS), INT8 p50 **8.751 ms** (114.28 FPS); model 44.6 → 11.2 MB; peak RAM 83.89 → 23.97 MB; all layers quantised, calibrated on 1,500 images — [Arm/resnet-18-int8-xnnpack-executorch](https://huggingface.co/Arm/resnet-18-int8-xnnpack-executorch)
- Other 224-px INT8 models on the same Arm harness (snippet only): SqueezeNet 1.1 p50 2.31 ms, ShuffleNet V2 x1.0 3.63 ms — [Arm developer model pages (search summary)](https://developer.arm.com/ai/models/hugging-face/Arm/mobilenet-v3-small-int8-xnnpack-executorch/mobilenetv3-small-int8-pte)
- The MobileNetV3-Large and EfficientNet-B0 Arm cards returned HTTP 401 and could not be read.
- Blog claim with no stated conditions (low authority): MobileNetV3-Small ~200 FPS and MobileNetV3-Large ~80 FPS with TFLite on Pi 5 — [zbotic.in (snippet only)](https://zbotic.in/image-classification-on-edge-device-mobilenet-and-pi-5/)
- Ultralytics classification table (server "CPU ONNX", conditions not in the macro): YOLO26n-cls 224 px top-1 71.4 / top-5 90.1, **5.0 ± 0.3 ms CPU ONNX**, 2.8 M params, 0.4 B FLOPs; YOLO26s-cls 76.0 / 92.9, 7.9 ms, 6.7 M, 1.5 B — [yolo-cls-perf macro](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/macros/yolo-cls-perf.md). YOLO11n-cls: 70.0 top-1 / 89.4 top-5, 5.0 ± 0.3 ms CPU ONNX, 1.1 ms T4; speeds "averaged over ImageNet val images using an Amazon EC2 P4d instance"; param count reported as 2.8 M in one docs version and 1.6 M in another (snippet only) — [Ultralytics classify task docs](https://docs.ultralytics.com/zh/tasks/classify)
- Sony IMX500 (AI Camera) export table lists yolov8n-cls / yolo11n-cls at 224 px at 33.31 ms, but validated on ImageNet10 with top-1 0.25 — a tiny toy set, so not a meaningful accuracy datum (see Q3) — [sony-imx500.md](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/integrations/sony-imx500.md)

### Inferences
- Per-crop cost budget on Pi 5 CPU (inference only, 224 px): FP32 NCNN MobileNetV3 ≈ 7 ms (4 thr) / 12 ms (1 thr); EfficientNet-B0 ≈ 12 / 24 ms; ResNet18 ≈ 20 / 30 ms. For 5–15 crops per frame this is **~37–110 ms (MobileNetV3, 4 thr)**, 60–185 ms (EfficientNet-B0) or 100–300 ms (ResNet18) — i.e. with ResNet18 the classifier stage can cost more than the detector. With INT8 MobileNetV3-Small via XNNPACK the whole crop batch is ~10–32 ms; with INT8 ResNet-18 ~44–130 ms.
- Multithreading gives only ~1.5–1.6× for these tiny nets (mobilenet_v3 11.80 → 7.35 ms; resnet18 29.68 → 19.96 ms), so the classifier runs almost as well pinned to one core, which is the natural way to leave cores free for SLAM (see Q7).
- Running at 128/160 px instead of 224 cuts FLOPs by ~3× / ~2×; no Pi 5 measurement at those sizes was found, so treat it as an extrapolation to verify.
- yolo11n-cls / yolo26n-cls on Pi 5: no measurement exists. On the same server CPU, yolo26n-cls (5.0 ms) is ~7.8× cheaper than yolo26n-det at 640 (38.9 ms); applying that ratio to the Pi 5 NCNN detector figure (67 ms) suggests roughly **~9 ms per 224-px crop** for yolo26n-cls with NCNN, i.e. comparable to MobileNetV3 — an extrapolation, not a measurement.

### Gaps
- No Pi 5 measurement for yolo11n-cls, yolov8n-cls or yolo26n-cls in NCNN/ONNX/TFLite.
- No Pi 5 TFLite/LiteRT+XNNPACK number for MobileNetV3 with stated thread count; the only TFLite claim is an unqualified blog.
- No Pi 5 figures for MobileNetV3-Large or EfficientNet-B0 in INT8 (Arm cards inaccessible), and no 128/160-px classifier measurements.
- The ncnn benchmark does not say whether `mobilenet_v3` is the Large or Small variant, nor the FP16 settings used.

---

## Key question 3 — Hailo-8L AI HAT (and Hailo-8 AI HAT+, AI Camera IMX500) numbers for YOLOv8n/11n and classifiers; constraints

### Takeaway
Hailo's own Model Zoo lists YOLOv8n at 202 FPS (batch 1) and YOLO11n at 157 FPS on Hailo-8L, and ~1,000–1,900 FPS for 224-px classifiers, but those are "hardware-only" figures measured from an x86 host over PCIe Gen3 x4; on a Pi 5 (single PCIe lane, weak CPU doing pre/post-processing) community reports show far lower end-to-end FPS, especially when NMS runs on the CPU. The AI Camera (IMX500) runs YOLOv8n/YOLO11n INT8 at ~59 ms per frame on-sensor and only supports those two detectors.

### Cited Findings

**Hailo Model Zoo, Hailo-8L, COCO detection (Dataflow Compiler v2.19.0; host Intel Core i5-9400, PCIe Gen3 x4, room temperature)** — [HAILO8L_object_detection.rst](https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst)

| Network | float mAP | hardware (quantised) mAP | FPS b=1 | FPS b=8 | Input | Params M | OPS G |
|---|---|---|---|---|---|---|---|
| yolov8n | 37.0 | 36.4 | **202** | 438 | 640×640 | 3.2 | 8.74 |
| yolov8s | 44.6 | 43.9 | 110 | 208 | 640×640 | 11.2 | 28.6 |
| yolov11n | 39.0 | 37.5 | 157 | 371 | 640×640 | 2.6 | 6.55 |
| yolov10n | 38.5 | 36.6 | 150 | 359 | 640×640 | 2.3 | 6.8 |
| yolov6n | 34.3 | 32.4 | 356 | 356 | 640×640 | 4.32 | 11.12 |
| yolov5s | 35.3 | 34.0 | 124 | 243 | 640×640 | 7.46 | 17.44 |

**Hailo-8 (26 TOPS, AI HAT+ 26) same conditions** — [HAILO8_object_detection.rst](https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/docs/public_models/HAILO8/HAILO8_object_detection.rst): yolov8n 1036 FPS; yolov8s 491; yolov11n 185 (b=1) / 541 (b=8); yolov10n 194 / 567; yolov6n 1250.

**Hailo-8L classification (ImageNet, same conditions)** — [HAILO8L_classification.rst](https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/docs/public_models/HAILO8L/HAILO8L_classification.rst)

| Network | float top-1 | hw top-1 | FPS b=1 | Input | Params M | OPS G |
|---|---|---|---|---|---|---|
| mobilenet_v3 (single entry; small/large not specified) | 72.2 | 71.8 | 1935 | 224 | 4.07 | 2 |
| mobilenet_v2_1.0 | 71.8 | 70.9 | 1738 | 224 | 3.49 | 0.62 |
| efficientnet_lite0 | 75.0 | 73.9 | 1057 | 224 | 4.63 | 0.78 |
| efficientnet_s | 77.6 | 77.2 | 548 | 224 | 5.41 | 4.72 |
| resnet_v1_18 | 71.3 | 71.1 | 1281 | 224 | 11.68 | 3.64 |
| resnet_v1_50 | 75.2 | 74.7 | 138 (502 at b=8) | 224 | 25.53 | 6.98 |
| squeezenet_v1.1 | 59.8 | 59.3 | 2514 | 224 | 1.24 | 0.78 |

- Model Zoo structure/toolchain: Hailo-8/8L are served by Model Zoo v2.x with Dataflow Compiler v3.x; the `master` branch (DFC 5.4.0, HailoRT 5.4.0, TensorFlow 2.19.1, CUDA 12.5.1, Python 3.10–3.12) targets Hailo-10H/15; the flow is parse → optimize ("numeric translation of the input model into a compressed integer representation") → compile to HEF → evaluate/profile — [hailo_model_zoo README](https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/README.rst)
- Compile command for custom models on Pi 5 AI Kit: `hailomz compile yolov8n --hw-arch hailo8l` producing a `.hef` (snippet only) — [Seeed tutorial](https://wiki.seeedstudio.com/cn/tutorial_of_ai_kit_with_raspberrypi5_about_yolov8n_object_detection/)
- Operator constraints reported on Hailo's forum (snippet only, community posts not official docs): concat limited to 4 inputs; transposes swapping H↔W or H↔features only work when the tensor is < 1.5 MB quantised; LayerNormalization not a native op in DFC ≥ 3.30 (must be rebuilt from primitives); ReduceMean must reduce over one spatial dim; STFT unsupported; GeLU in preview; self-attention/BERT-style not supported on Hailo-8/8L (LLMs targeted at Hailo-10H) — [Hailo community: supported operators](https://community.hailo.ai/t/supported-operators/5046), [model parsing error thread](https://community.hailo.ai/t/model-parsing-error-reducemean-unsqueeze/1064), [Hailo8l max input dimension](https://community.hailo.ai/t/hailo8l-maximum-input-dimension/14909), [Concat17 error](https://community.hailo.ai/t/concat17-error-during-custom-yolov8m-pt-compilation-using-dfc/12896)
- Real Pi 5 throughput vs Model Zoo (snippet only): a user got ~16 FPS for YOLOv8m 640 on Pi 5 + Hailo-8 vs 130–150 expected; Hailo community reply: on weak CPUs pre/post-processing can be the bottleneck, and with batch size 2 they measured 49 FPS on Pi 5 + Hailo-8; another reply attributes the gap to the Model Zoo being measured on PCIe Gen3 x2 while the Pi 5 has Gen3 x1 and to the figures being `hw_only` with synthetic data; a different user saw "zero effect" from enabling PCIe Gen3; another measured YOLOv8m 31 FPS / YOLOv8l 19 FPS at batch 1 on Pi 5 + Hailo-8 — [Low FPS YOLOv8m thread](https://community.hailo.ai/t/low-fps-with-yolov8m-640x640-on-raspberry-pi-5-only-16-fps-vs-130-expected/17313), [performance not as official thread](https://community.hailo.ai/t/the-performance-on-the-raspberry-pi-5-with-the-hailo-8-chip-seems-not-good-as-he-official-results/17473), [benchmark differs thread](https://community.hailo.ai/t/benchmark-performance-differs-from-hailo-model-zoo-results-on-raspberry-pi-5-hailo-8l/19293)
- For pose models CPU-side post-processing "ends up taking more time than the inferencing itself", in contrast to detection HEFs where "the nms postprocessing step is integrated into the HEF file" (snippet only) — [Hailo community pose/NMS thread](https://community.hailo.ai/t/running-yolo-pose-with-nms-postprocessing-on-hailo8/15800)
- Seeed CM4 reComputer R1000 + Hailo-8L: 0.75 FPS CPU → 29.5 FPS accelerated for YOLOv8 detection; pose 0.5 → 27 FPS (variant not named; CM4 not Pi 5; snippet only) — [Seeed wiki](https://wiki.seeedstudio.com/yolov8_object_detection_on_recomputer_r1000_with_hailo_8l/)
- hailo-rpi5-examples: Pi 5 only; Hailo-8 = 26 TOPS, Hailo-8L = 13 TOPS; GStreamer-based pipelines; a `detection_simple.py` (YOLOv6 Nano) exists "minimizing CPU load"; HailoRT Python API link v4.18.0; repo marked outdated in favour of `hailo-apps` — [hailo-rpi5-examples](https://github.com/hailo-ai/hailo-rpi5-examples)
- AI HAT+ 2 (Hailo-10H, 40 TOPS INT4, 8 GB dedicated LPDDR4X, vision performance "comparable to the original AI HAT+ 26 TOPS", US$130; original AI HAT+ 26 TOPS US$110, 13 TOPS variant US$70; retail release 2026-01-15) — retailer/press only, no official page fetched — [Tecnoblog](https://tecnoblog.net/noticias/raspberry-pi-anuncia-placa-ai-hat-2-com-40-tops-para-ia-preco-e-alto/), [Switch Science](https://www.switch-science.com/products/10916), [buyzero.de](https://buyzero.de/en/products/raspberry-pi-ai-hat-2-hailo-10h-ki-beschleuniger-8-gb-ram)

**Raspberry Pi AI Camera (Sony IMX500) via Ultralytics `format=imx`** — [sony-imx500.md](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/integrations/sony-imx500.md)
- Benchmarks "run by the Ultralytics team on a Raspberry Pi AI Camera" (INT8, 640): YOLOv8n packerOut 2.1 MB, mAP50-95 0.470 (coco128), **58.79 ms/im**; YOLO11n 2.2 MB, 0.517, 58.82 ms; pose 58.79 / 62.50 ms; cls at 224 px 33.31 ms.
- Constraints: only YOLOv8n and YOLO11n supported ("other families, scales and architectures are not"); `quantize` must be 8 (INT8 PTQ via Sony MCT, auto-enabled; FP32/FP16 not supported); `nms=True` required for detect/segment/pose; export on Linux only, Python 3.10–3.12, Java 17+; deployment scripts set camera to 16 fps for detection (17 for pose/segmentation) "for maximum FPS"; the sensor outputs metadata only, reducing data to the host.
- Anecdotal forum report: biggest model loadable on the IMX500 ~8 MB (snippet only) — [Core Electronics forum](https://forum.core-electronics.com.au/t/raspberry-pi-ai-camera-quick-start-guide/21090?page=2); official sensor modes 2028×1520 @ 30 fps binned, 4056×3040 @ 10 fps (snippet only) — [Elektor](https://elektormagazine.com/articles/raspberry-pi-ai-camera)

### Inferences
- For a nano detector the Hailo-8L's raw capacity (150–200 FPS) is 10–15× beyond what the Pi 5 CPU can do; in practice the pipeline will be bounded by Pi-side camera decode, resize and (if not compiled into the HEF) NMS, so expect tens of FPS, not hundreds, and plan to compile detectors with NMS inside the HEF and use batch 2+ where latency allows.
- Classifiers on Hailo-8L are effectively free (≈0.5–1 ms per 224-px crop hardware time), so a two-stage pipeline could put both stages on the HAT, leaving the CPU to SLAM — but each crop incurs a PCIe round-trip, so batching crops (b=8 tables exist) matters.
- Hailo quantisation losses on the official zoo: yolov8n −0.6 mAP, yolov11n −1.5, yolov10n −1.9, yolov6n −1.9; mobilenet_v3 −0.4 top-1, resnet18 −0.2, efficientnet_lite0 −1.1 — i.e. modest but non-zero, worse for the newer YOLO heads.
- The AI Camera is a detector-only offload option (YOLOv8n/11n at ~17 fps, INT8 only, NMS on-sensor); it cannot run the crop classifier, so stage 2 would still run on the CPU (or HAT). It frees the CPU from stage 1 entirely, which is valuable when SLAM shares the board.
- Compiling for Hailo requires the Dataflow Compiler toolchain (x86 Linux host with TensorFlow/CUDA listed in the README) — the Pi only runs HEFs.

### Gaps
- No official Pi-5-specific FPS table from Hailo or Raspberry Pi for YOLOv8n/11n on Hailo-8L (all zoo numbers are x86-host, Gen3 x4); community numbers are for larger models.
- Official Raspberry Pi documentation pages for AI HAT+/AI Camera (requirements, PCIe Gen3 setting, cooling) could not be fetched (DNS + GitHub path 404).
- No Hailo-8L number for MobileNetV3-Small vs Large separately, nor for EfficientNet-B0 (only efficientnet_lite0/s/m).
- IMX500 memory limit for models is not stated officially; the ~8 MB figure is anecdotal.

---

## Key question 4 — RAM footprint of ultralytics + torch vs NCNN/ONNX-only runtimes on Pi 5 (4 GB shared with SLAM)

### Takeaway
The only Pi 5 measurement found shows the full Ultralytics+PyTorch path peaking at ~1.2 GB system RAM during YOLOv8n validation versus ~475 MB with the ONNX or NCNN backends (idle baseline ~445 MB); standalone runtimes such as ExecuTorch/XNNPACK need only 10–90 MB of process memory for nano/small models.

### Cited Findings
- Pi 5 (Ultralytics v8.2.80, Bookworm, Python 3.11.2, YOLOv8n, coco128 validation): "Max memory used when running validation on PyTorch ~1.2GB"; "Max memory used when running validation on ONNX, NCNN ~475MB"; idle baseline ~445 MB; the 2 GB case was simulated by limiting RAM and ran slower in all formats (snippet only) — [Ultralytics community thread](https://community.ultralytics.com/t/raspberry-pi-new-cheaper-raspberry-pi-released/208)
- User report, Pi 5 4 GB, YOLO11n NCNN with Ultralytics v8.3.202 inside a larger app: >5× faster than PyTorch, memory "stayed under about 1 GB"; the same user reported the Pi freezing when re-loading an NCNN model (unresolved in the excerpt) (snippet only) — [Ultralytics community crash thread](https://community.ultralytics.com/t/raspberry-pi-5-crash-when-running-yolov11n-ncnn/1525)
- ExecuTorch process memory on Pi 5: MobileNetV3-Small FP32 peak 16.56 MB / INT8 10.31 MB; ResNet-18 FP32 83.89 MB / INT8 23.97 MB; YOLOv8s 640 FP32 141.06 MB / INT8 90.28 MB — [Arm MobileNetV3-Small card](https://huggingface.co/Arm/mobilenet-v3-small-int8-xnnpack-executorch), [Arm ResNet-18 card](https://huggingface.co/Arm/resnet-18-int8-xnnpack-executorch), [Arm YOLOv8s card](https://huggingface.co/Arm/yolov8s-int8-xnnpack-executorch)
- Model files on disk (Pi 5 docs): YOLO26n PyTorch 5.3 MB, NCNN 9.4 MB, ONNX 9.5 MB; YOLO11n 5.4 / 10.2 / 10.2 MB; YOLOv8n 6.2 / 12.2 / 12.2 MB; TF SavedModel ~26–31 MB; PaddlePaddle ~20–24 MB — [raspberry-pi.md main](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/guides/raspberry-pi.md), [v8.3.40](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.40/docs/en/guides/raspberry-pi.md), [v8.2.100](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.2.100/docs/en/guides/raspberry-pi.md)
- Ultralytics recommends Raspberry Pi OS Lite (headless) to save RAM for vision processing, and an SSD over SD card for 24×7 operation — [raspberry-pi.md main](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/guides/raspberry-pi.md)
- slam_toolbox async mapping on a Pi 5 reported at ~1% of one CPU and ~500 MB RSS, +0.4 W (anecdotal forum post, snippet only) — [Dexter Industries forum](https://forum.dexterindustries.com/t/raspberry-pi-5-continued-ros-2-humble-in-docker-on-pios-bookworm/9561)

### Inferences
- Even with the NCNN backend, importing the `ultralytics` Python package drags in torch (the package depends on it), so the "~475 MB" figure likely reflects the Python process with torch loaded but idle; a torch-free deployment (ncnn Python bindings / C++ ncnn, onnxruntime, or LiteRT) should bring the vision process well under ~150 MB, which on a 4 GB board shared with ROS 2 + SLAM (≈0.5 GB) and the OS (~0.4 GB) leaves comfortable margin. This is an inference from the above, not a direct measurement.
- Keeping torch out of the runtime image also avoids the ~1.2 GB PyTorch-path peak, which would otherwise consume ~30% of the 4 GB before SLAM.

### Gaps
- No current (2025–2026) official RAM comparison from Ultralytics for Pi 5; the only measured comparison is v8.2.80 (2024) and was snippet-only.
- No measurement of RAM for a torch-free NCNN pipeline running detector + classifier concurrently on a Pi 5.

---

## Key question 5 — INT8 accuracy loss for YOLO nano and for MobileNet-class classifiers

### Takeaway
For YOLO nano, naive static full-integer PTQ typically costs 3–7 mAP50-95 points (e.g. YOLOv8n 37.4 → 33.9 per-channel, 32.9 per-tensor; YOLO11n −6.6 on LiteRT), whereas dynamic-range quantisation (~−0.1) or static INT8 that keeps the detection head in FP32 (YOLOv8s −0.45) and Hailo's calibrated flow (yolov8n −0.6, yolov11n −1.5) lose under 2 points. For classifiers, ResNet-18 INT8 is essentially lossless (−0.2 pp), while MobileNetV3 (hard-swish) is the fragile case: −0.4 (Hailo), −2.0 pp (Arm PTQ, Small), and catastrophic collapse with naive per-tensor PTQ reported for the MobileNet family.

### Cited Findings

**YOLO nano**
- YOLOv8n TFLite export: FP32 (SiLU) 37.4 mAP50-95; INT8 per-channel 33.9; INT8 per-tensor 32.9; ReLU6-variant INT8 31.4 (ReLU6 "has less accuracy drop after quantization than SiLU" relative to its own float baseline) (snippet only; site DNS-blocked) — [Cloudflight: Quantized YOLO for Edge Solutions](https://engineering.cloudflight.io/quantized-yolo-for-edge-solutions)
- Static INT8 TensorRT engines of YOLOv8n/s on clean COCO show "a moderate baseline accuracy drop of approximately 3–7% absolute mAP50-95", FP16 negligible loss; under medium synthetic noise the relative mAP50-95 drop for YOLOv8n static INT8 exceeded 60% (snippet only) — [arXiv 2508.19600, Quantization Robustness to Input Degradations for Object Detection](https://arxiv.org/pdf/2508.19600)
- YOLO11n on Android LiteRT: full-integer INT8 mAP loss 6.6 vs FP32; dynamic-range quantisation loses only 0.1 mAP (max across models); a further 0.2–0.4 mAP loss attributed to converting YOLO11's newer layers to LiteRT (snippet only) — [arXiv 2511.13453](https://arxiv.org/pdf/2511.13453)
- Ultralytics TFLite export naming trap: `yolo11n_int8.tflite` is dynamic-range (weights INT8 for storage, compute FP32); the true full-integer model is `yolo11n_full_integer_quant.tflite` and needs manual input/output scaling (snippet only) — [Ultralytics community thread](https://community.ultralytics.com/t/exporting-to-tflite-with-int8-quantization-for-edge-deployment/1523)
- YOLOv8s INT8 static PTQ on Pi 5 (ExecuTorch/XNNPACK): mAP50-95 44.31 → 43.86 (−0.45 pp), mAP50 60.70 → 60.39, with the detection-head prediction convs, DFL decode, sigmoid, concat and first conv kept in FP32; C2f/SPPF concats rewritten as split convs; calibrated on 2,000 COCO images; 95.43% of ops delegated to XNNPACK; latency 1018.84 → 457.03 ms (2.23×), model 42.78 → 13.24 MB — [Arm YOLOv8s card](https://huggingface.co/Arm/yolov8s-int8-xnnpack-executorch)
- Hailo-8L calibrated 8-bit: yolov8n 37.0 → 36.4; yolov8s 44.6 → 43.9; yolov11n 39.0 → 37.5; yolov10n 38.5 → 36.6; yolov6n 34.3 → 32.4 — [HAILO8L_object_detection.rst](https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst)
- IMX500 INT8 (Sony MCT PTQ) on coco128: YOLOv8n 0.470 and YOLO11n 0.517 mAP50-95, vs the FP32 coco128 values in the Pi docs of 0.609–0.638 (YOLOv8n) and 0.608–0.610 (YOLO11n) — [sony-imx500.md](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/integrations/sony-imx500.md), [v8.2.100 Pi docs](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.2.100/docs/en/guides/raspberry-pi.md), [v8.3.40 Pi docs](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.40/docs/en/guides/raspberry-pi.md)
- Quantised YOLOv4-Tiny on Pi 5 "maintained comparable accuracy with 36% faster inference and 43% lower power draw" (no mAP figures in snippet) — [arXiv 2506.09300](https://arxiv.org/pdf/2506.09300)

**MobileNet-class / ResNet classifiers**
- MobileNetV3-Small INT8 static PTQ (per-channel weights, per-tensor activations, some layers kept FP32, 2,000 calibration images), ImageNet-1k val: top-1 66.97 → 64.94 (−2.03 pp), top-5 87.13 → 85.92 (−1.21 pp) — [Arm MobileNetV3-Small card](https://huggingface.co/Arm/mobilenet-v3-small-int8-xnnpack-executorch)
- ResNet-18 INT8 static PTQ (all layers quantised, 1,500 calibration images): top-1 69.76 → 69.57 (−0.19 pp), top-5 89.08 → 89.00 — [Arm ResNet-18 card](https://huggingface.co/Arm/resnet-18-int8-xnnpack-executorch)
- Hailo-8L 8-bit: mobilenet_v3 72.2 → 71.8; mobilenet_v2_1.0 71.8 → 70.9; efficientnet_lite0 75.0 → 73.9; efficientnet_s 77.6 → 77.2; resnet_v1_18 71.3 → 71.1; resnet_v1_50 75.2 → 74.7 — [HAILO8L_classification.rst](https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/docs/public_models/HAILO8L/HAILO8L_classification.rst)
- Architectures using Hard-Swish (MobileNetV3) "are more vulnerable to INT8 degradation"; simple equalisation is insufficient and KD fine-tuning or AdaRound helped (snippet only) — [arXiv 2607.18101](https://arxiv.org/pdf/2607.18101)
- Naive 8-bit weight quantisation can collapse the MobileNet family on ImageNet "from 70.60% ~ 71.87% to 0.1%" (per-tensor, no equalisation; MobileNetV3 not isolated) (snippet only) — [Weight Equalizing Shift Scaler-Coupled PTQ](https://cs.paperswithcode.com/paper/weight-equalizing-shift-scaler-coupled-post)
- PaddleClas QAT: mobilenet_v3_large_x1_0 (distilled, 78.9% top-1) → 75.9% after quantisation (snippet only; QAT not PTQ) — [PaddleClas quantization docs](https://paddleclas.readthedocs.io/en/latest/extension/paddle_quantization_en.html)
- STMicroelectronics MobileNetV2 model cards: a100 73.17 → 72.76 % int8; a075 70.78 → 70.33 %; w035 61.02 → 60.09 % (snippet only) — [STM mobilenetv2_pt](https://huggingface.co/STMicroelectronics/mobilenetv2_pt/blob/main/README.md)

### Inferences
- For the fruit-detection use case the safest CPU-side INT8 recipe, per the evidence, is: per-channel weights, keep the detection head/DFL/sigmoid in FP32, calibrate on ≥1–2k representative images; expect ≤1 mAP point loss. Fully-integer TFLite exports (as needed for IMX500/Edge-TPU-style targets) should be budgeted at 3–7 points unless QAT is applied.
- On the Pi 5 CPU specifically, INT8 is *not* needed for the detector to hit ~10–15 FPS with NCNN FP32, so the accuracy risk can be avoided on stage 1; INT8 pays off mainly for stage 2 (classifier) via XNNPACK (1.45× for MobileNetV3-Small, 3.6× for ResNet-18) and for the Hailo/IMX500 paths where it is mandatory.
- ResNet-18 is the most quantisation-robust classifier option (−0.2 pp) at ~8.8 ms INT8 per crop; MobileNetV3 is the fastest but the most quantisation-sensitive; EfficientNet-lite variants sit between (−1.1 pp on Hailo).

### Gaps
- No PTQ ImageNet figure for MobileNetV3-Large (the Arm card was inaccessible, Hailo lists a single unspecified `mobilenet_v3`).
- No INT8 accuracy data for YOLOv9t, YOLO12n or YOLO26n (the NMS-free head's quantisation behaviour is undocumented in the sources found).
- The Cloudflight and arXiv numbers are from search snippets; exact toolchain/calibration-set sizes could not be verified.

---

## Key question 6 — Is YOLO at 1024 px on Pi 5 CPU practical, and is tiling/cropping an alternative?

### Takeaway
No measured Pi 5 figure exists for any YOLO nano at 1024 px; scaling the official 640-px NCNN numbers by the 2.56× pixel ratio gives an estimate of ~170–240 ms/frame (≈4–6 FPS) for YOLO26n/YOLO11n, which is workable for a slow-moving robot but roughly triples the detector's CPU share. Tiling (SAHI-style) buys small-object recall at a latency cost proportional to tile count and is only worthwhile if the fruit are small relative to the frame.

### Cited Findings
- 640-px Pi 5 NCNN latencies to scale from: YOLO26n 67.03 ms, YOLO11n 88.03 ms, YOLOv8n 94.28 ms; ONNX 126–199 ms — [raspberry-pi.md main](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/guides/raspberry-pi.md), [v8.3.40](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.40/docs/en/guides/raspberry-pi.md), [v8.2.100](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.2.100/docs/en/guides/raspberry-pi.md)
- Search for YOLO11n imgsz 1024 NCNN Pi 5 returned no measured data; the only available reasoning is the 2.56× pixel ratio extrapolation (≈225 ms for YOLO11n NCNN) — [search summary citing Ultralytics guide](https://docs.ultralytics.com/guides/raspberry-pi)
- An IoT small-object study found tiling "carried a significant time cost" on an edge device and presents tiling as "a valid alternative" when raising input resolution is blocked by memory (snippet only) — [arXiv 2311.07163](https://arxiv.org/pdf/2311.07163)
- A UAV maritime study: static SAHI with 640-px tiles raised small-object mAP50 to 0.42 but cut throughput to 0.60 FPS, judged impractical (non-Pi hardware; snippet only) — [arXiv 2511.19728](https://arxiv.org/pdf/2511.19728)
- Practitioner guidance: SAHI "won't always be faster than one large model due to tiling overhead, but it is far more accurate on small objects in high-res images" (snippet only) — [Roboflow forum](https://discuss.roboflow.com/t/max-image-size-monochrome-training-yolo/10366/4); Ultralytics community discussion on downsampling vs tiling — [Ultralytics community](https://community.ultralytics.com/t/improving-object-detection-accuracy-downsampling-vs-tiling-in-yolo-models/1458)
- Hailo-8L input-size constraint: transposes involving H↔W only work when the quantised tensor is < 1.5 MB, and "very wide or tall inputs often fail" (community; snippet only) — [Hailo8l maximum input dimension](https://community.hailo.ai/t/hailo8l-maximum-input-dimension/14909)

### Inferences
- Extrapolated Pi 5 cost at 1024 px (FP32, inference only, not measured): NCNN YOLO26n ≈ 170 ms, YOLO11n ≈ 225 ms, YOLOv8n ≈ 240 ms; ONNX ≈ 320–510 ms; PyTorch ≈ 0.8–1.3 s. Activation memory also grows ~2.56×, still small in absolute terms for nano models (tens of MB).
- A 2×2 tiling of a 1280×960-class frame into four 640 tiles costs ~4× the 640 latency (≈270–380 ms NCNN) *plus* merge/NMS, i.e. more than a single 1024 pass, while a single 1024 pass processes 2.56× pixels for ~2.56× cost. Tiling is therefore only preferable when fruit are so small that 1024 still under-resolves them, or when memory/accelerator input limits (e.g. Hailo tensor-size constraints, IMX500 model size) prevent a large input.
- Given the robot runs SLAM concurrently, 1024 px on CPU is feasible only at ~3–5 FPS with the detector occupying most of 3–4 cores; a pragmatic alternative is 640/800 px detection plus the 224-px crop classifier (stage 2 already recovers fine detail per fruit), or offloading stage 1 to Hailo/IMX500.

### Gaps
- No measured Pi 5 latency for any YOLO at 800, 960 or 1024 px in any format; no Pi 5 SAHI benchmark.
- Whether Hailo-8L compiles YOLO nano at 1024×1024 within its tensor limits was not confirmed.

---

## Key question 7 — Practical notes: batching crops, threading, thermal throttling, power supply, and sharing the CPU with SLAM

### Takeaway
Nano classifiers scale poorly with threads on the Pi 5 (≈1.5–1.6× from 1→4 threads), so run the classifier on 1–2 pinned cores and the NCNN detector on the remaining cores; use the official 27 W (5 V/5 A) PSU and the active cooler because the Pi 5 begins throttling around 80–85 °C under sustained load; slam_toolbox itself is light (~1% of a core, ~0.5 GB), so the detector, not SLAM, is the dominant CPU consumer.

### Cited Findings
- Thread scaling on Pi 5 from ncnn's official benchmark (4 vs 1 thread, avg ms): mobilenet_v3 7.35 vs 11.80; mobilenet_v2 10.52 vs 16.38; efficientnet_b0 12.33 vs 24.20; resnet18 19.96 vs 29.68; yolov4-tiny 46.29 vs 68.02; nanodet_m 11.18 vs 19.69 — [ncnn benchmark README](https://raw.githubusercontent.com/Tencent/ncnn/master/benchmark/README.md)
- Hailo b=1 vs b=8 FPS on 8L: yolov8n 202 → 438, yolov11n 157 → 371; classifiers show no gain from batching (mobilenet_v3 1935 → 1936) — [HAILO8L_object_detection.rst](https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst), [HAILO8L_classification.rst](https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/docs/public_models/HAILO8L/HAILO8L_classification.rst)
- Arm cards measured at batch 1; ExecuTorch "time to first inference" 3.1 ms (MobileNetV3-Small INT8) and 17.1 ms (ResNet-18 INT8), model load 6.5 ms / 39 ms — [Arm MobileNetV3-Small card](https://huggingface.co/Arm/mobilenet-v3-small-int8-xnnpack-executorch), [Arm ResNet-18 card](https://huggingface.co/Arm/resnet-18-int8-xnnpack-executorch)
- A Pi 5 YOLOv8n optimisation write-up recommends reserving two cores for the ncnn thread pool and pinning camera/pre-processing threads to the other cores to avoid core migration and cache misses (no measured 2-vs-4 thread numbers; snippet only, site DNS-blocked) — [viblo.asia article](https://viblo.asia/p/trien-khai-yolov8n-tren-raspberry-pi-5-toi-uu-tu-kernel-den-assembly-de-dat-hieu-nang-toi-da-bNVQGN3AJvR)
- Ultralytics: overclock to 3.0 GHz is possible with `force_turbo=1` but requires proper cooling; Pi 5 max draw 5 A @ 5 V (PD); use Raspberry Pi OS Lite to save RAM; SSD recommended for 24×7 — [raspberry-pi.md main](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/guides/raspberry-pi.md)
- Official 27 W (5 V/5 A) USB-C supply "highly recommended to prevent under-voltage warnings when both the CPU and fan are at maximum load" (retailer page; snippet only) — [Core Electronics Active Cooler](https://core-electronics.com.au/raspberry-pi-5-active-cooler.html)
- Pi 5 thermal behaviour: soft limit ~80 °C where clocks start to drop, hard limit 85 °C; a passive heatsink alone does not keep a sustained workload below the throttle point (community/tutorial sources, low authority; snippet only) — [dev.to thermal showdown](https://dev.to/mahinshanazeer/raspberry-pi-5-thermal-showdown-active-cooler-vs-passive-cooling-2e2i), [evezone guide](https://evezone.evetech.co.za/build-lab/how-to-set-up-active-cooling-on-a-raspberry-pi-5-to-stop-throttling)
- IEEE SoutheastCon edge-AI benchmark: the Pi 5 "operates at significantly higher temperatures, reaching up to 80 °C without active cooling" (abstract-level; snippet only) — [Georgia Southern publication record](https://scholars.georgiasouthern.edu/en/publications/benchmarking-edge-ai-platforms-performance-analysis-of-nvidia-jet/)
- Quantised YOLOv4-Tiny on Pi 5: 36% faster and 43% lower power draw than FP32 (snippet only) — [arXiv 2506.09300](https://arxiv.org/pdf/2506.09300)
- Hailo pipelines: on weak CPUs pre/post-processing is the bottleneck; batch size 2 gave 49 FPS for YOLOv8m on Pi 5 + Hailo-8; detection HEFs integrate NMS, pose HEFs do not (snippet only) — [Hailo community](https://community.hailo.ai/t/low-fps-with-yolov8m-640x640-on-raspberry-pi-5-only-16-fps-vs-130-expected/17313), [pose/NMS thread](https://community.hailo.ai/t/running-yolo-pose-with-nms-postprocessing-on-hailo8/15800)
- SLAM load: slam_toolbox async mode on a Pi 5 reported at "1% of 1 cpu, 500MB, +0.4W (base is 2.6W)" (anecdotal; snippet only) — [Dexter Industries forum](https://forum.dexterindustries.com/t/raspberry-pi-5-continued-ros-2-humble-in-docker-on-pios-bookworm/9561); on a Pi 4 a user tracing dropped LiDAR scans found a *different* node was the top CPU consumer with 1-min load ≈41% of the 4-core CPU (snippet only) — [Dexter Industries forum](https://forum.dexterindustries.com/t/ros2-distance-sensor-node-using-68-cpu/9207); a replacement RPLIDAR driver "results in very little CPU time being consumed" and the C++ `sllidar_ros2` driver is recommended (snippet only) — [Husarion community](https://community.husarion.com/t/closed-rplidar-via-gpio-for-better-performance-why-howto-ros2-docker/1734)
- A 2026 peer-reviewed benchmark of 2D LiDAR SLAM (GMapping, Hector, Cartographer) with ROS 2 on Raspberry Pi 5 exists but is closed-access; its CPU figures were not retrievable (snippet only) — [İstün Üniversitesi repository](https://openaccess.istun.edu.tr/entities/publication/d1239799-7ccc-42f9-aab3-c351b7861bd9)
- Reported stability issue: a Pi 5 (4 GB) freezing when re-loading an NCNN YOLO11n model inside a larger app (Ultralytics v8.3.202) (snippet only) — [Ultralytics community](https://community.ultralytics.com/t/raspberry-pi-5-crash-when-running-yolov11n-ncnn/1525)

### Inferences
- Core allocation that follows from the thread-scaling data: give NCNN detector 2–3 threads (nano models do scale on 4 threads for the detector-size workloads, e.g. yolov4-tiny 68 → 46 ms), run the classifier single-threaded on one core processing crops sequentially (5–15 × 12 ms ≈ 60–180 ms FP32 MobileNetV3, or ~2–3 ms each INT8), and pin ROS 2/SLAM/LiDAR driver to the remaining core(s) via `taskset`/cgroups. Total per-frame budget ≈ 90 ms detector + 60–180 ms classifier → ~3–6 end-to-end FPS at 640 FP32 on CPU alone, more with INT8 classifier.
- Batching crops on CPU brings little benefit (XNNPACK/NCNN are already compute-bound at batch 1 on tiny inputs; Hailo classifiers show zero batch gain); batching matters mainly for the Hailo *detector* (2× at b=8) and to amortise PCIe round-trips.
- Load each model once and keep it resident (ExecuTorch load times 6–40 ms are small, but the NCNN re-load freeze report argues against repeated model re-instantiation).
- Sustained CPU inference will push a Pi 5 to its 80–85 °C throttle point without the active cooler; with a mobile robot's battery supply, verify 5 V/5 A capability (or accept the 600 mA USB peripheral limit on 3 A supplies implied by the retailer warning) and log `vcgencmd get_throttled` during tests.
- Power-wise, INT8 reduces energy per inference (43% lower draw reported for YOLOv4-Tiny on Pi 5), relevant for battery endurance on the robot.

### Gaps
- No measured Pi 5 benchmark that logs temperature/throttling during sustained YOLO inference with specified cooler and PSU.
- No measurement of detector + classifier + SLAM running concurrently on a Pi 5; SLAM CPU figures are anecdotal forum numbers.
- Official Raspberry Pi power-supply documentation (5 V/5 A vs 3 A peripheral-current behaviour) could not be fetched.
- No published 2-vs-4-thread or FP16/BF16 ncnn numbers for YOLOv8n/11n specifically on Pi 5.
