---
comments: true
description: Learn how to export Ultralytics YOLO models to MXQ format for high-performance, low-power inference on Mobilint ARIES NPUs.
keywords: Mobilint, MXQ, ARIES, Regulus, NPU, model export, Ultralytics, YOLO, edge AI, quantization, qbcompiler, qbruntime
---

# Mobilint MXQ Export and Deployment

[Mobilint](https://mobilint.com/) designs energy-efficient AI accelerators built on a proprietary dataflow architecture for [edge AI](https://www.ultralytics.com/glossary/edge-ai) workloads. Exporting [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models to the **MXQ** format compiles them for the **ARIES** family of NPUs, enabling high-FPS, low-power [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) inference at the edge.

<p align="center">
  <img width="50%" src="https://static.wixstatic.com/media/4ddc53_3e04651bd0be4008b396fe7798336132~mv2.png/v1/fill/w_1434,h_655,al_c,q_90,usm_0.66_1.00_0.01,enc_avif,quality_auto/Mobilint_Chip.png" alt="Mobilint ARIES NPU edge AI accelerator">
</p>

## Mobilint's qb SDK

The Mobilint software stack is built around two components that together let any supported neural network run on ARIES/Regulus NPUs as a single, self-contained `.mxq` artifact.

### qb Compiler

<p align="center">
  <img width="70%" src="https://static.wixstatic.com/media/4ddc53_149ace516d104abb93bfe7a825c3d0d4~mv2.png/v1/fill/w_831,h_588,fp_0.50_0.50,q_90,usm_0.66_1.00_0.01,enc_avif,quality_auto/Model%20Build(full).png" alt="Mobilint qb Compiler — Model build pipeline">
</p>

`qb Compiler` ingests trained models — **vision (detection, segmentation, pose, classification), language (LLMs), and vision-language (VLMs)** — from PyTorch or other source formats and lowers them onto the NPU. The compiler:

- Applies **state-of-the-art quantization techniques** to compress weights and activations with minimal accuracy loss.
- Performs NPU-aware graph scheduling: operator fusion, memory layout planning, multi-core partitioning, and dataflow-aware tiling.
- Emits a **single `.mxq` artifact** that bundles the quantized weights, the compiled execution schedule.

### qb Runtime

<p align="center">
  <img width="70%" src="https://static.wixstatic.com/media/4ddc53_bff6d30d00904eb9b1169b71fed25f41~mv2.png/v1/crop/x_47,y_71,w_2775,h_1998/fill/w_831,h_588,fp_0.50_0.50,q_90,usm_0.66_1.00_0.01,enc_avif,quality_auto/Runtime%20Execution.png" alt="Mobilint qb Runtime — Multi-core execution flow">
</p>

`qb Runtime` loads and executes the `.mxq` artifact on Mobilint hardware. The runtime:

- Implements **efficient multi-core orchestration** — it distributes work across the NPU's clusters and cores according to the `core_mode` recorded at compile time (`single` / `multi` / `global4` / `global8`).
- Handles async I/O, request queuing, and per-stream pipelining so concurrent inference requests are scheduled onto available cores with minimal host overhead.
- Exposes a stable C/C++/Python API (`qbruntime`) so the same `.mxq` runs identically across host applications.

In this Ultralytics integration, the `export` mode invokes `qb Compiler` (via the `qbcompiler` package), and `predict`/`val` modes load the compiled `.mxq` through `qb Runtime` (via the `qbruntime` package).


## Hardware Targets

Select the `target` argument matching your device:

| Target    | Description                                            |
| :-------- | :----------------------------------------------------- |
| `aries-rb`   | ARIES NPU (8 cores — supports all `core_mode` options) |
| `regulus-rb` | Regulus NPU (single physical core — `single` only)     |

## Core Modes

ARIES is built from **2 physical clusters of 4 cores each (8 cores total)**. The `core_mode` argument selects how a compiled MXQ model maps onto this hardware. Pick the mode that matches your workload's throughput vs. latency profile.


| Mode      | Cores Used     | Optimized For                              |
| :-------- | :------------- | :----------------------------------------- |
| `single`  | 1 of 8 cores   | Single-stream throughput on one core       |
| `multi`   | 4 cores        | Batch processing of 4 images at once       |
| `global4` | 1 cluster (4)  | Lowest latency for a single image          |
| `global8` | 2 clusters (8) | Lowest latency for a single image          |

- **`single`** — Each image is processed by **one of the 8 cores**. Multiple cores can run independent streams in parallel, so this mode scales out across many concurrent inputs. Async I/O and async post-processing have a large impact on overall throughput.
- **`multi`** — A **batch of 4 images** is processed together, optimized for offline batch inference and high-throughput pipelines.
- **`global4`** — A **single cluster (4 cores)** cooperates on **one image** to minimize per-image latency. Useful when latency matters more than aggregate throughput.
- **`global8`** — **Both clusters (all 8 cores)** cooperate on **one image** for the lowest possible per-image latency, at the cost of leaving no cores free for concurrent streams.

For more detail, see Mobilint's [Multi-Core Inference documentation](https://docs.mobilint.com/v1.2/en/multicore.html).

## Installation

!!! warning "Platform Requirements"

    - Operating System: Linux (Ubuntu 22.04 recommended)
    - Python: 3.10+
    - Hardware (export): x86_64 host, GPU optional (accelerates calibration)
    - Hardware (inference): A Mobilint ARIES NPU device
    - qbcompiler: v1.2.0
    - qbruntime: v1.2.0

### Ultralytics Installation

```bash
pip install ultralytics mobilint-qb-runtime
pip install mobilint-qb-compiler --index-url https://dl.mobilint.com
```

For detailed instructions, see the [Ultralytics Installation guide](../quickstart.md). If you encounter difficulties, consult the [Common Issues guide](../guides/yolo-common-issues.md).

### Mobilint Toolchain Installation

The MXQ export requires `qbcompiler`, and on-device inference requires `qbruntime`. Refer to the [Mobilint developer resources](https://github.com/mobilint) for the latest installation packages and driver setup matching your target hardware.

## Exporting YOLO Models to MXQ

Export your trained YOLO model with the standard Ultralytics `export` API. `target` and `core_mode` are required.

!!! example "Export to MXQ Format"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO model
        model = YOLO("yolo26s.pt")

        # Export to MXQ format
        model.export(
            format="mxq",
            target="aries-rb",        # 'aries-rb' | 'regulus-rb'
            core_mode="single",    # 'single' | 'multi' | 'global4' | 'global8' | 'all'
            data="coco8.yaml",   # calibration dataset (recommended)
        )  # creates 'yolo26s_mobilint_model/'
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26s.pt format=mxq target=aries-rb core_mode=single data=coco8.yaml
        ```

### Export Arguments

| Argument    | Type             | Default          | Description                                                                                                            |
| :---------- | :--------------- | :--------------- | :--------------------------------------------------------------------------------------------------------------------- |
| `format`    | `str`            | `'mxq'`          | Target format for the exported model.                                                                                  |
| `target`    | `str`            | **required**     | Mobilint hardware target: `aries-rb` or `regulus-rb`.                                                                        |
| `core_mode` | `str`            | **required**     | NPU core scheduling mode to compile for: `single`, `multi`, `global4`, `global8`, or `all`. Use `all` to compile one model that supports every runtime mode (see [Core Modes](#core-modes)). |
| `imgsz`     | `int` or `tuple` | `640`            | Input image size — integer for square or `(height, width)` tuple.                                                      |
| `batch`     | `int`            | `1`              | Calibration batch size and exported model batch dimension.                                                             |
| `data`      | `str`            | `'coco8.yaml'`   | [Dataset](https://docs.ultralytics.com/datasets/) YAML used to draw INT8 calibration samples (up to 100 images).       |
| `device`    | `str`            | `None`           | Export device: GPU (`device=0`) accelerates calibration; otherwise CPU.                                                |

!!! tip "Calibration"

    Providing `data=` is strongly recommended. If no calibration data is supplied, the compiler falls back to **random-input calibration**, which can degrade [accuracy](https://www.ultralytics.com/glossary/accuracy) significantly. 100 representative images is typically sufficient.

### Output

The exporter writes a self-contained folder next to the source weights:

```
yolo26s_mobilint_model/
├── yolo26s_mobilint_model        # Compiled MXQ artifact loaded by qbruntime
└── metadata.yaml      # names / task / imgsz / head params (reg_max, nl, nm)
```

The sidecar `metadata.yaml` is required for inference on custom-trained models — it carries the class names and post-processing parameters that cannot be inferred from the `.mxq` filename alone.

## Running Inference

Pass the exported folder (or a bare `.mxq` file from Hugging Face) to the Ultralytics API just like any other model. Pre-compiled YOLO MXQs are also published at https://huggingface.co/mobilint.

!!! example "Inference with an MXQ Model"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported MXQ model
        model = YOLO("yolo12s.mxq", task="detect")

        # Run inference — pass NPU placement on the predict() call.
        # See "Selecting Cores at Inference Time" below for all options.
        results = model.predict(
            "https://ultralytics.com/images/bus.jpg",
            target="aries-rb",
            core_mode="single",
            cluster_id=0,
            core_id=0,
        )

        for r in results:
            print(f"Detected {len(r.boxes)} objects")
            r.show()
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo12s.mxq task=detect \
            target=aries-rb core_mode=single cluster_id=0 core_id=0 \
            source='https://ultralytics.com/images/bus.jpg'
        ```

### Selecting Cores at Inference Time

Two additional arguments — `cluster_id` and `core_id` — control which physical cores the loaded model runs on. They are **inference-time only** (the compiled `.mxq` itself is core-agnostic; allocation happens when the model is loaded into `qbruntime`). Pass them to `model.predict(...)` / `model.val(...)`, alongside `target` and `core_mode`.

Recall ARIES has **2 clusters × 4 cores = 8 cores total**. Which arguments to provide depends on `core_mode`:

| `core_mode` | `cluster_id`              | `core_id`               | Notes                                                                                          |
| :---------- | :------------------------ | :---------------------- | :--------------------------------------------------------------------------------------------- |
| `single`    | `0` or `1`                | `0`, `1`, `2`, or `3`   | Picks one specific core. Lists are allowed to pin multiple streams to specific cores at once.  |
| `multi`     | `0` or `1` (or `[0, 1]`)  | _not used_              | Picks the cluster(s) the 4-image batch runs on. `core_id` is ignored.                          |
| `global4`   | `0` or `1`                | _not used_              | Picks the single cluster (4 cores) that cooperates on each image. `core_id` is ignored.        |
| `global8`   | _not used_                | _not used_              | All 8 cores cooperate on each image. Neither argument is required.                             |

!!! note "Runtime `core_mode` must be one of the four real modes"

    At inference time, `core_mode` must be `single`, `multi`, `global4`, or `global8` — **not** `all`. Models compiled with `core_mode="all"` at export still need a specific runtime mode chosen when loading (any of the four works).

!!! example "Basic per-mode usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26s_mobilint_model", task="detect")
        src = "https://ultralytics.com/images/bus.jpg"

        # single: run on Cluster0 / Core0
        model.predict(src, target="aries-rb", core_mode="single", cluster_id=0, core_id=0)

        # multi: 4-image batch on Cluster0
        model.predict(src, target="aries-rb", core_mode="multi", cluster_id=0)

        # global4: 1 cluster (4 cores) cooperates on each image
        model.predict(src, target="aries-rb", core_mode="global4", cluster_id=0)

        # global8: all 8 cores cooperate on each image — no IDs needed
        model.predict(src, target="aries-rb", core_mode="global8")
        ```

        !!! tip "Switching modes on the same `YOLO` instance"

            The qbruntime backend is initialized on the **first** `predict()`/`val()` call and reused thereafter, so the four blocks above effectively select the mode chosen on the first call. To switch `core_mode` / `cluster_id` / `core_id` at runtime, instantiate a new `YOLO("yolo26s_mobilint_model", task="detect")` per configuration.

    === "CLI"

        ```bash
        # single: Cluster0 / Core0
        yolo predict model=yolo26s_mobilint_model task=detect \
            target=aries-rb core_mode=single cluster_id=0 core_id=0 source=https://ultralytics.com/images/bus.jpg

        # multi: Cluster0
        yolo predict model=yolo26s_mobilint_model task=detect \
            target=aries-rb core_mode=multi cluster_id=0 source=https://ultralytics.com/images/bus.jpg

        # global4: Cluster0
        yolo predict model=yolo26s_mobilint_model task=detect \
            target=aries-rb core_mode=global4 cluster_id=0 source=https://ultralytics.com/images/bus.jpg

        # global8: no cluster/core args
        yolo predict model=yolo26s_mobilint_model task=detect \
            target=aries-rb core_mode=global8 source=https://ultralytics.com/images/bus.jpg
        ```

!!! example "Advanced — registering multiple cores for parallel workloads (`single` only)"

    In `single` mode, `cluster_id` and `core_id` can be lists so that **multiple cores are registered as a pool** at model load time. The two lists are zipped element-wise, so they must have the same length.

    The cores are not pinned to specific inputs — instead, when your application issues many concurrent inference calls (e.g. from a thread pool running "read input → `model.predict(...)` → save result" for each frame), the qbruntime scheduler distributes those calls across the registered cores. Throughput scales with the number of concurrent workers, up to the number of registered cores.

    === "Python"

        ```python
        from concurrent.futures import ThreadPoolExecutor
        from ultralytics import YOLO

        model = YOLO("yolo26s_mobilint_model", task="detect")

        # Register 5 cores on the first predict() — these are captured at backend
        # load time and reused for every subsequent call on this YOLO instance.
        mxq_cfg = dict(
            target="aries-rb",
            core_mode="single",
            cluster_id=[0, 0, 0, 1, 1],
            core_id=[0, 1, 3, 2, 3],
        )

        sources = ["frame0.jpg", "frame1.jpg", ...]  # many inputs

        def infer_one(src):
            return model.predict(src, **mxq_cfg)

        # 10 worker threads share the 5-core pool — qbruntime schedules each call
        # onto an available registered core.
        with ThreadPoolExecutor(max_workers=10) as ex:
            results = list(ex.map(infer_one, sources))
        ```

    === "CLI"

        ```bash
        # The yolo CLI runs a single inference loop, so multi-core registration mainly
        # helps when you drive predict() from your own threaded code (Python). The list
        # form is still valid on the CLI:
        yolo predict model=yolo26s_mobilint_model task=detect \
            target=aries-rb core_mode=single cluster_id=[0,0,0,1,1] core_id=[0,1,3,2,3] \
            source=/path/to/frames/
        ```

## Supported Tasks

| Task                                                              | Status |
| :---------------------------------------------------------------- | :----- |
| [Object Detection](https://docs.ultralytics.com/tasks/detect/)    | ✅     |
| [Segmentation](https://docs.ultralytics.com/tasks/segment/)       | ✅     |
| [Pose Estimation](https://docs.ultralytics.com/tasks/pose/)       | ✅     |
| [Classification](https://docs.ultralytics.com/tasks/classify/)    | ✅     |
| [Oriented Bounding Boxes](https://docs.ultralytics.com/tasks/obb/) | ✅     |

### Object Detection Task
```
yolo export model=yolo26s.pt format=mxq target=aries-rb core_mode=all data=coco8.yaml
yolo predict model=yolo26s_mobilint_model task=detect \
    target=aries-rb core_mode=single cluster_id=0 core_id=0 \
    source=https://ultralytics.com/images/bus.jpg
yolo val model=yolo26s_mobilint_model task=detect target=aries-rb core_mode=global8 data=coco8.yaml
```

### Segmentation Task
```
yolo export model=yolo26s-seg.pt format=mxq target=aries-rb core_mode=all data=coco8-seg.yaml
yolo predict model=yolo26s-seg_mobilint_model task=segment \
    target=aries-rb core_mode=single cluster_id=0 core_id=0 \
    source=https://ultralytics.com/images/bus.jpg
yolo val model=yolo26s-seg_mobilint_model task=segment target=aries-rb core_mode=global8 data=coco8-seg.yaml
```

### Pose Estimation Task
```
yolo export model=yolo26s-pose.pt format=mxq target=aries-rb core_mode=all data=coco8-pose.yaml
yolo predict model=yolo26s-pose_mobilint_model task=pose \
    target=aries-rb core_mode=single cluster_id=0 core_id=0 \
    source=https://ultralytics.com/images/bus.jpg
yolo val model=yolo26s-pose_mobilint_model task=pose target=aries-rb core_mode=global8 data=coco8-pose.yaml
```

### Classification Task
```
yolo export model=yolo26s-cls.pt imgsz=224 format=mxq target=aries-rb core_mode=all data=imagenet100
yolo predict model=yolo26s-cls_mobilint_model task=classify imgsz=224 \
    target=aries-rb core_mode=single cluster_id=0 core_id=0 \
    source=https://ultralytics.com/images/bus.jpg
yolo val model=yolo26s-cls_mobilint_model task=classify imgsz=224 target=aries-rb core_mode=global8 data=imagenet100
```

## Real-World Applications

YOLO on Mobilint NPUs enables a range of edge deployments:

- **Smart Surveillance**: High-FPS multi-stream object detection for [security analytics](https://www.ultralytics.com/blog/computer-vision-applications-ai-in-security-systems).
- **Industrial Inspection**: Low-latency defect detection on the production line, fully on-device.
- **Retail Analytics**: Real-time [object counting](https://docs.ultralytics.com/guides/object-counting/) and [heatmaps](https://docs.ultralytics.com/guides/heatmaps/) without cloud dependency.
- **Robotics & Mobility**: Vision-based perception on power-constrained autonomous platforms.

## Benchmarks

Each cell shows the **MXQ measurement** (on ARIES) outside the parentheses, and the **official `.pt` benchmark** inside the parentheses.  
[official Ultralytics YOLO26 benchmarks](https://docs.ultralytics.com/models/yolo26/).

### Detect (COCO val2017)

| Model         | Size |     mAPval 50-95 |
| ------------- | ---- | ---------------: |
| `yolo26n.mxq` | 640  | _TBD_ (40.9) |
| `yolo26s_mobilint_model` | 640  | _TBD_ (48.6) |
| `yolo26m.mxq` | 640  | _TBD_ (53.1) |
| `yolo26l.mxq` | 640  | _TBD_ (55.0) |
| `yolo26x.mxq` | 640  | _TBD_ (57.5) |

### Segment (COCO val2017)

| Model             | Size | mAPval 50-95 (B) | mAPval 50-95 (M) |
| ----------------- | ---- | ---------------: | ---------------: |
| `yolo26n-seg.mxq` | 640  | _TBD_ (39.6) | _TBD_ (33.9) |
| `yolo26s-seg_mobilint_model` | 640  | _TBD_ (47.3) | _TBD_ (40.0) |
| `yolo26m-seg.mxq` | 640  | _TBD_ (52.5) | _TBD_ (44.1) |
| `yolo26l-seg.mxq` | 640  | _TBD_ (54.4) | _TBD_ (45.5) |
| `yolo26x-seg.mxq` | 640  | _TBD_ (56.5) | _TBD_ (47.0) |

### Pose (COCO val2017)

| Model              | Size | mAPval 50-95 (B) | mAPval 50-95 (P) |
| ------------------ | ---- | ---------------: | ---------------: |
| `yolo26n-pose.mxq` | 640  | _TBD_ (57.2) | _TBD_ (83.3) |
| `yolo26s-pose_mobilint_model` | 640  | _TBD_ (63.0) | _TBD_ (86.6) |
| `yolo26m-pose.mxq` | 640  | _TBD_ (68.8) | _TBD_ (89.6) |
| `yolo26l-pose.mxq` | 640  | _TBD_ (70.4) | _TBD_ (90.5) |
| `yolo26x-pose.mxq` | 640  | _TBD_ (71.6) | _TBD_ (91.6) |

### Classify (ImageNet val)

| Model             | Size |       Top-1 Acc |       Top-5 Acc |
| ----------------- | ---- | --------------: | --------------: |
| `yolo26n-cls.mxq` | 224  | _TBD_ (71.4) | _TBD_ (90.1) |
| `yolo26s-cls_mobilint_model` | 224  | _TBD_ (76.0) | _TBD_ (92.9) |
| `yolo26m-cls.mxq` | 224  | _TBD_ (78.1) | _TBD_ (94.2) |
| `yolo26l-cls.mxq` | 224  | _TBD_ (79.0) | _TBD_ (94.6) |
| `yolo26x-cls.mxq` | 224  | _TBD_ (79.9) | _TBD_ (95.0) |

### OBB (DOTAv1 test)

| Model             | Size | mAPtest 50-95 |  mAPtest 50 |
| ----------------- | ---- | ------------: | ----------: |
| `yolo26n-obb.mxq` | 1024 |  _TBD_ (52.4) | _TBD_ (78.9) |
| `yolo26s-obb.mxq` | 1024 |  _TBD_ (54.8) | _TBD_ (80.9) |
| `yolo26m-obb.mxq` | 1024 |  _TBD_ (55.3) | _TBD_ (81.0) |
| `yolo26l-obb.mxq` | 1024 |  _TBD_ (56.2) | _TBD_ (81.6) |
| `yolo26x-obb.mxq` | 1024 |  _TBD_ (56.7) | _TBD_ (81.7) |


## Recommended Workflow

1. **Train** your model with the Ultralytics [Train Mode](https://docs.ultralytics.com/modes/train/).
2. **Export** to MXQ with `model.export(format="mxq", target=..., core_mode=..., data=...)`.
3. **Validate** with `yolo val` to quantify any quantization-induced accuracy change.
4. **Deploy** the `.mxq` file to your Mobilint device and run inference via `qbruntime`.

## Summary

This guide showed how to export Ultralytics YOLO models to Mobilint's **MXQ** format and run them on **ARIES** NPUs. The combination of Ultralytics YOLO and Mobilint hardware delivers efficient, real-time inference for production edge AI deployments.

For more Ultralytics integrations, see the [integration guide page](../integrations/index.md). For the Mobilint toolchain itself, visit the [Mobilint GitHub](https://github.com/mobilint) and the [Mobilint Model Zoo](https://github.com/mobilint/mblt-model-zoo).

## FAQ

### How do I export an Ultralytics YOLO model to MXQ format?

Use the standard `export` API with `format="mxq"`, plus the required `target` and `core_mode` arguments:

```python
from ultralytics import YOLO

model = YOLO("yolo26s.pt")
model.export(format="mxq", target="aries-rb", core_mode="single", data="coco8.yaml")
```

This produces a single `yolo26s_mobilint_model` file ready for deployment.

### Which Mobilint hardware targets are supported?

`aries-rb` and `regulus-rb`.

### What does `core_mode` control?

`core_mode` selects how a compiled MXQ model maps onto the ARIES NPU's 8 cores:

- `single` — each image runs on **one** core (use multiple cores from your application via threads for higher throughput).
- `multi` — a **batch of 4 images** runs across the chosen cluster(s).
- `global4` — **one cluster (4 cores)** cooperates on a single image — low per-image latency.
- `global8` — **both clusters (all 8 cores)** cooperate on a single image — lowest per-image latency.

Choose based on whether you need highest aggregate throughput (`single` + concurrency) or lowest per-image latency (`global4` / `global8`). See [Core Modes](#core-modes) for details.

### When should I export with `core_mode="all"`?

Use `core_mode="all"` when shipping a **production artifact that needs to switch core modes dynamically** at runtime — a single `.mxq` can be loaded under any of `single` / `multi` / `global4` / `global8`. The trade-off is **compile time**: `all` builds all four modes in one pass, so it is slower than compiling a single mode.

During development, prefer compiling a **specific mode** (`single`, `multi`, `global4`, or `global8`) so iteration stays fast. Switch to `all` only when the deployment target genuinely requires runtime flexibility. `all` is an export-only value — runtime/predict still requires picking one of the four real modes.

### How many calibration images should I use?

Fewer than 100 images is typically enough for stable quantization, and the Ultralytics MXQ exporter samples up to that many from the dataset YAML by default. For **accuracy-critical products**, increase the number of calibration images gradually and validate accuracy at each step until the curve plateaus.

### Where can I find pre-compiled MXQ models?

The [`mobilint` Hugging Face organization](https://huggingface.co/mobilint) hosts pre-compiled YOLO models.
