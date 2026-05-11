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
| `aries`   | ARIES NPU (8 cores — supports all `core_mode` options) |
| `regulus` | Regulus NPU (single physical core — `single` only)     |

## Core Modes

ARIES is built from **2 physical clusters of 4 cores each (8 cores total)**. The `core_mode` argument selects how a compiled MXQ model maps onto this hardware. Pick the mode that matches your workload's throughput vs. latency profile.

!!! note "Regulus is single-core"

    Regulus has **only one physical core**, so `target=regulus` supports `core_mode=single` only. The multi-core modes (`multi`, `global4`, `global8`) apply to ARIES.

| Mode      | Cores Used     | Optimized For                              |
| :-------- | :------------- | :----------------------------------------- |
| `single`  | 1 of 8 cores   | Single-stream throughput on one core       |
| `multi`   | 4 cores        | Batch processing of 4 images at once       |
| `global4` | 1 cluster (4)  | Lowest latency for a single image          |
| `global8` | 2 clusters (8) | Lowest latency for a single image (all HW) |

- **`single`** — Each image is processed by **one of the 8 cores**. Multiple cores can run independent streams in parallel, so this mode scales out across many concurrent inputs. Async I/O and async post-processing have a large impact on overall throughput.
- **`multi`** — A **batch of 4 images** is processed together, optimized for offline batch inference and high-throughput pipelines.
- **`global4`** — A **single cluster (4 cores)** cooperates on **one image** to minimize per-image latency. Useful when latency matters more than aggregate throughput.
- **`global8`** — **Both clusters (all 8 cores)** cooperate on **one image** for the lowest possible per-image latency, at the cost of leaving no cores free for concurrent streams.

!!! tip "`all` — export-only versatile compile"

    Passing `core_mode="all"` to **export** compiles a single `.mxq` that can be loaded under **any** of the four runtime modes (`single`, `multi`, `global4`, `global8`). It is not itself a runtime mode — at inference time you still pick one of the four. Use `all` when you want one artifact that can serve different deployment scenarios; use a specific mode at export when you know exactly how the model will be served and want the smallest, most-tightly-scheduled binary.

For more detail, see Mobilint's [Multi-Core Inference documentation](https://docs.mobilint.com/v1.2/en/multicore.html).

## Supported Tasks

| Task                                                              | Status |
| :---------------------------------------------------------------- | :----- |
| [Object Detection](https://docs.ultralytics.com/tasks/detect/)    | ✅     |
| [Segmentation](https://docs.ultralytics.com/tasks/segment/)       | ✅     |
| [Pose Estimation](https://docs.ultralytics.com/tasks/pose/)       | ✅     |
| [Classification](https://docs.ultralytics.com/tasks/classify/)    | ✅     |

## Installation

!!! warning "Platform Requirements"

    - **Operating System**: Linux (Ubuntu 22.04 recommended)
    - **Python**: 3.10+
    - **Hardware (export)**: x86_64 host, GPU optional (accelerates calibration)
    - **Hardware (inference)**: A Mobilint ARIES NPU device with the matching driver and `qbruntime` installed

### Ultralytics Installation

```bash
pip install ultralytics
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
        model = YOLO("yolo11s.pt")

        # Export to MXQ format
        model.export(
            format="mxq",
            target="aries",        # 'aries' | 'regulus'
            core_mode="single",    # 'single' | 'multi' | 'global4' | 'global8' | 'all'
            data="coco8.yaml",     # calibration dataset (recommended)
        )  # creates 'yolo11s.mxq'
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11s.pt format=mxq target=aries core_mode=single data=coco8.yaml
        ```

### Export Arguments

| Argument    | Type             | Default          | Description                                                                                                            |
| :---------- | :--------------- | :--------------- | :--------------------------------------------------------------------------------------------------------------------- |
| `format`    | `str`            | `'mxq'`          | Target format for the exported model.                                                                                  |
| `target`    | `str`            | **required**     | Mobilint hardware target: `aries` or `regulus`.                                                                        |
| `core_mode` | `str`            | **required**     | NPU core scheduling mode to compile for: `single`, `multi`, `global4`, `global8`, or `all`. Use `all` to compile one model that supports every runtime mode (see [Core Modes](#core-modes)). |
| `imgsz`     | `int` or `tuple` | `640`            | Input image size — integer for square or `(height, width)` tuple.                                                      |
| `batch`     | `int`            | `1`              | Calibration batch size and exported model batch dimension.                                                             |
| `data`      | `str`            | `'coco8.yaml'`   | [Dataset](https://docs.ultralytics.com/datasets/) YAML used to draw INT8 calibration samples (up to 100 images).       |
| `device`    | `str`            | `None`           | Export device: GPU (`device=0`) accelerates calibration; otherwise CPU.                                                |

!!! tip "Calibration"

    Providing `data=` is strongly recommended. If no calibration data is supplied, the compiler falls back to **random-input calibration**, which can degrade [accuracy](https://www.ultralytics.com/glossary/accuracy) significantly. 100 representative images is typically sufficient.

### Output

The compiled model is written next to the source weights, alongside a small sidecar carrying class names, task type, image size, and other export metadata:

```
yolo11s.mxq     # Compiled MXQ model
yolo11s.yaml    # Sidecar metadata (class names, task, imgsz, etc.) — auto-loaded at inference
```

Keep the `.yaml` next to the `.mxq` when distributing the model so the Ultralytics MXQ backend can restore class names and other metadata. If the sidecar is missing, inference still runs but visualizations fall back to numeric class labels (`class0`, `class1`, ...).

## Running Inference

Load the `.mxq` file with the Ultralytics API just like any other model.

!!! example "Inference with an MXQ Model"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported MXQ model
        model = YOLO("yolo11s.mxq",
                     task="detect",
                     target="aries",
                     core_mode="single",
                     cluster_id=0,
                     core_id=0)

        # Run inference — explicitly pick the NPU core to run on.
        # See "Selecting Cores at Inference Time" below for all options.
        results = model("https://ultralytics.com/images/bus.jpg")

        for r in results:
            print(f"Detected {len(r.boxes)} objects")
            r.show()
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo11s.mxq task=detect \
            target=aries core_mode=single cluster_id=0 core_id=0 \
            source='https://ultralytics.com/images/bus.jpg'
        ```

### Pre-Compiled Models from Hugging Face

If you do not have a local `.mxq` file, Ultralytics will auto-download a pre-compiled model from the [`mobilint`](https://huggingface.co/mobilint) Hugging Face organization. Provide `target` and `core_mode` to select the variant, plus `cluster_id` / `core_id` to pick the NPU core(s) it will run on:

!!! example "Auto-Download Pre-Compiled MXQ"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Downloads mobilint/YOLO12m/aries/single/yolo12m.mxq
        model = YOLO("yolo12m.mxq",
                     task="detect",
                     target="aries",
                     core_mode="single",
                     cluster_id=0,
                     core_id=0)
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo12m.mxq task=detect \
            target=aries core_mode=single cluster_id=0 core_id=0 \
            source='https://ultralytics.com/images/bus.jpg'
        ```

### Selecting Cores at Inference Time

Two additional arguments — `cluster_id` and `core_id` — control which physical cores the loaded model runs on. They are **inference-time only** (the compiled `.mxq` itself is core-agnostic; allocation happens when the model is loaded into `qbruntime`). Pass them on the `YOLO(...)` constructor, alongside `target` and `core_mode`.

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

        # single: run on Cluster0 / Core0
        mxq_cfg = dict(target="aries", core_mode="single", cluster_id=0, core_id=0)
        model = YOLO("yolo11s.mxq", task="detect", **mxq_cfg)
        model.predict("bus.jpg")

        # multi: 4-image batch on Cluster0
        mxq_cfg = dict(target="aries", core_mode="multi", cluster_id=0)
        model = YOLO("yolo11s.mxq", task="detect", **mxq_cfg)
        model.predict("bus.jpg")

        # global4: 1 cluster (4 cores) cooperates on each image
        mxq_cfg = dict(target="aries", core_mode="global4", cluster_id=0)
        model = YOLO("yolo11s.mxq", task="detect", **mxq_cfg)
        model.predict("bus.jpg")

        # global8: all 8 cores cooperate on each image — no IDs needed
        mxq_cfg = dict(target="aries", core_mode="global8")
        model = YOLO("yolo11s.mxq", task="detect", **mxq_cfg)
        model.predict("bus.jpg")
        ```

    === "CLI"

        ```bash
        # single: Cluster0 / Core0
        yolo predict model=yolo11s.mxq task=detect \
            target=aries core_mode=single cluster_id=0 core_id=0 source=bus.jpg

        # multi: Cluster0
        yolo predict model=yolo11s.mxq task=detect \
            target=aries core_mode=multi cluster_id=0 source=bus.jpg

        # global4: Cluster0
        yolo predict model=yolo11s.mxq task=detect \
            target=aries core_mode=global4 cluster_id=0 source=bus.jpg

        # global8: no cluster/core args
        yolo predict model=yolo11s.mxq task=detect \
            target=aries core_mode=global8 source=bus.jpg
        ```

!!! example "Advanced — registering multiple cores for parallel workloads (`single` only)"

    In `single` mode, `cluster_id` and `core_id` can be lists so that **multiple cores are registered as a pool** at model load time. The two lists are zipped element-wise, so they must have the same length.

    The cores are not pinned to specific inputs — instead, when your application issues many concurrent inference calls (e.g. from a thread pool running "read input → `model.predict(...)` → save result" for each frame), the qbruntime scheduler distributes those calls across the registered cores. Throughput scales with the number of concurrent workers, up to the number of registered cores.

    === "Python"

        ```python
        from concurrent.futures import ThreadPoolExecutor
        from ultralytics import YOLO

        # Register 5 cores at load time
        model = YOLO("yolo11s.mxq",
                     task="detect",
                     target="aries",
                     core_mode="single",
                     cluster_id=[0, 0, 0, 1, 1],
                     core_id=[0, 1, 3, 2, 3])

        sources = ["frame0.jpg", "frame1.jpg", ...]  # many inputs

        def infer_one(src):
            return model.predict(src)

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
        yolo predict model=yolo11s.mxq task=detect \
            target=aries core_mode=single cluster_id=[0,0,0,1,1] core_id=[0,1,3,2,3] \
            source=/path/to/frames/
        ```

!!! warning "Required at inference time"

    `core_mode` is **always required** for MXQ inference, and `cluster_id` / `core_id` are required whenever the mode uses them (see the table above):

    - `single` — both `cluster_id` and `core_id` required.
    - `multi`, `global4` — `cluster_id` required.
    - `global8` — neither is needed.

    The backend will raise a `ValueError` if a required argument is missing — there is no silent fallback to a default core. Be explicit so multiple models running on the same device do not collide.

### Validation

Verify quantization accuracy with `yolo val`:

```bash
yolo val model=yolo12m.mxq task=detect target=aries core_mode=global8 data=coco8.yaml
```

## Real-World Applications

YOLO on Mobilint NPUs enables a range of edge deployments:

- **Smart Surveillance**: High-FPS multi-stream object detection for [security analytics](https://www.ultralytics.com/blog/computer-vision-applications-ai-in-security-systems).
- **Industrial Inspection**: Low-latency defect detection on the production line, fully on-device.
- **Retail Analytics**: Real-time [object counting](https://docs.ultralytics.com/guides/object-counting/) and [heatmaps](https://docs.ultralytics.com/guides/heatmaps/) without cloud dependency.
- **Robotics & Mobility**: Vision-based perception on power-constrained autonomous platforms.

## Benchmarks

<!-- TODO: fill in measured numbers from internal benchmarking -->

!!! tip "Performance"

    | Model    | Target  | mAP50-95(B) | Inference time (ms/im) |
    | -------- | ------- | ----------- | ---------------------- |
    | YOLO26n  | `aries` | _TBD_       | _TBD_                  |
    | YOLO26s  | `aries` | _TBD_       | _TBD_                  |
    | YOLO26m  | `aries` | _TBD_       | _TBD_                  |
    | YOLO26l  | `aries` | _TBD_       | _TBD_                  |
    | YOLO26x  | `aries` | _TBD_       | _TBD_                  |

    Validation done on COCO val2017. Inference time excludes pre/post-processing.

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

model = YOLO("yolo11s.pt")
model.export(format="mxq", target="aries", core_mode="single", data="coco8.yaml")
```

This produces a single `yolo11s.mxq` file ready for deployment.

### Which Mobilint hardware targets are supported?

`aries` and `regulus`.

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

The [`mobilint` Hugging Face organization](https://huggingface.co/mobilint) hosts pre-compiled YOLO models. Ultralytics will auto-download them when you load a known model name (e.g. `YOLO("yolo12m.mxq", target="aries", core_mode="single")`). The full catalog is in the [Mobilint Model Zoo](https://github.com/mobilint/mblt-model-zoo).
