# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
from copy import deepcopy
from pathlib import Path

import torch

from ultralytics.utils import LOGGER

# Mobilint HF repos follow the layout `mobilint/<RepoName>/<device>/<core_mode>/<filename>.mxq`.
# `core_mode` is one of `single | multi | global4 | global8` and corresponds to qbcompiler's
# `inference_scheme` parameter at compile time.


def onnx2mxq(
    onnx_file: str | Path,
    save_path: str | Path,
    target: str,
    core_mode: str,
    task: str | None = None,
    calib_path: str | None = None,
    use_random_calib: bool = False,
    imgsz: int | tuple[int, int] = (640, 640),
    device: str = "cpu",
    prefix: str = "",
) -> None:
    """Compile an ONNX model to Mobilint MXQ format using qbcompiler.

    Args:
        onnx_file (str | Path): Path to the ONNX model to compile.
        save_path (str | Path): Output path for the compiled .mxq file.
        target (str): Mobilint hardware tier (e.g. `"aries"` | `"regulus"`).
            for future use; It depends on pip package versions of qbcompiler.
        core_mode (str): Compile inference scheme — one of `"single" | "multi" | "global4" |
            "global8" | "all"`. Forwarded to qbcompiler as `inference_scheme`.
        task (str | None, optional): Ultralytics task of the source model — one of `"detect"`,
            `"segment"`, `"pose"`, `"classify"`. Available for task-specific compile choices
            (e.g. preprocessing pipeline, output layout); pass `None` to keep generic defaults.
        calib_path (str | None, optional): Directory containing calibration `.npy` samples. If
            None, calibration data is not used.
        use_random_calib (bool, optional): If True, use random data for calibration. Mutually
            exclusive with `calib_path`.
        device (str, optional): Compile device, "cpu" or "gpu".
        imgsz (tuple[int, int], optional): (height, width) used by the letterbox preprocessing op.
        prefix (str, optional): Prefix string for log messages.
    """
    from qbcompiler import (
        CalibrationConfig,
        PreprocessingConfig,
        Uint8InputConfig,
        mxq_compile,
    )

    onnx_file = str(onnx_file)
    save_path = str(save_path)
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)
    height, width = imgsz
    maxsize = max(height, width)

    pipeline = []
    calib_per_ch = 1
    if task == "classify":
        calib_per_ch = 0
        pipeline.append({"op": "resize", "size": int(maxsize * (256 / 224)), "mode": "bilinear"})
        pipeline.append({"op": "centerCrop", "height": height, "width": width})
    else:
        pipeline.append({"op": "letterbox", "height": height, "width": width, "padValue": 114})

    preprocessing_config = PreprocessingConfig(
        apply=True,
        auto_convert_format=True,
        pipeline=pipeline,
        input_configs={},
    )
    calibration_config = CalibrationConfig(
        method=1,  # 0 for per tensor, 1 for per channel
        output=calib_per_ch,  # 0 for layer, 1 for channel
        mode=1,  # maxpercentile
        max_percentile={
            "percentile": 0.9999,
            "topk_ratio": 0.01,
        },
    )

    LOGGER.info(f"{prefix} mxq_compile running...")
    mxq_compile(
        model=onnx_file,
        calib_data_path=calib_path,
        use_random_calib=use_random_calib,
        save_path=save_path,
        inference_scheme=core_mode,
        image_channels=3,
        backend="onnx",
        device=device,
        preprocessing_config=preprocessing_config,
        uint8_input_config=Uint8InputConfig(apply=True, inputs=[]),
        calibration_config=calibration_config,
    )
    LOGGER.info(f"{prefix} mxq_compile completed successfully.")


def _filename_to_mobilint_repo(name: str) -> str | None:
    """Convert a local `.mxq` filename into its `mobilint/<RepoName>` HF repo id.

    Examples (lowercase `yolo` prefix becomes uppercase `YOLO`, rest preserved):
        yolov10s.mxq    -> mobilint/YOLOv10s
        yolo11s.mxq     -> mobilint/YOLO11s
        yolov8s-seg.mxq -> mobilint/YOLOv8s-seg

    Returns None if the filename does not start with `yolo`, signalling that the caller
    should not attempt an HF download.
    """
    stem = Path(name).stem
    if not stem.lower().startswith("yolo"):
        return None
    return f"mobilint/YOLO{stem[4:]}"


def attempt_download_mxq(file: str | Path, device: str, core_mode: str) -> str:
    """Resolve a `.mxq` model path, downloading from Hugging Face Hub if needed.

    Resolution order:
      1. Return the path if the file already exists.
      2. Return `<weights_dir>/<name>` if cached there from a prior download.
      3. Derive the HF repo from the filename (`yolo*.mxq` -> `mobilint/YOLO*`) and
         download `<device>/<core_mode>/<name>` into `<weights_dir>`.
      4. If the filename does not match the `yolo*` convention, return the path unchanged
         (caller will surface the missing-file error).

    Args:
        file (str | Path): Local `.mxq` path or bare filename (e.g. `"yolo11s.mxq"`).
        device (str): Mobilint hardware tier directory (e.g. `"aries"`).
        core_mode (str): Variant subdirectory — one of
            `"single" | "multi" | "global4" | "global8"`.

    Returns:
        (str): Path to the resolved (and possibly downloaded) file.
    """
    from ultralytics.utils import SETTINGS  # scoped to avoid circular import

    path = Path(str(file).strip().replace("'", ""))
    if path.exists():
        return str(path)

    weights_dir = Path(SETTINGS["weights_dir"])
    cached = weights_dir / path.name
    if cached.exists():
        return str(cached)

    repo_id = _filename_to_mobilint_repo(path.name)
    if repo_id is None:
        return str(path)  # not a yolo*.mxq; let the caller raise a clear error
    in_repo_path = f"{device}/{core_mode}/{path.name}"

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        from ultralytics.utils.checks import check_requirements

        check_requirements("huggingface_hub")
        from huggingface_hub import hf_hub_download

    weights_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Downloading {in_repo_path} from Hugging Face repo {repo_id}...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=in_repo_path,
        local_dir=str(weights_dir),
    )
    return str(downloaded)


def get_mobilint_model_zoo_post_cfg(model_name: str) -> dict:
    """Get mblt-model-zoo postprocess config for an official YOLO model name."""

    def import_cfg_classes(module_path: str) -> list:
        spec = importlib.util.find_spec(module_path)
        if spec is None or not spec.origin or spec.origin in {"built-in", "frozen"}:
            raise ModuleNotFoundError(module_path)

        cfg_classes = []
        for yolo_file in Path(spec.origin).parent.glob("yolo*.py"):
            tree = ast.parse(yolo_file.read_text(encoding="utf-8"), filename=str(yolo_file))
            module = importlib.import_module(f"{module_path}.{yolo_file.stem}")
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    cfg_classes.append(getattr(module, node.name))
        return cfg_classes

    def normalize_cfg_name(name: str) -> str:
        return name.lower().replace("_set", "").replace("_", "")

    def normalize_model_name(name: str) -> str:
        value = Path(name).stem.lower()
        value = value.replace("-ghost", "")
        value = value.replace("-resnet18", "")
        value = value.replace("-resnet50", "")
        value = value.replace("-resnet101", "")
        if "yolov5" in value:
            value = value.replace("-p6", "6")
        value = value.replace("-p6", "p6").replace("-p2", "p2")
        value = value.replace("-spp", "spp").replace("-tiny", "tiny")
        value = value.replace("-seg", "seg")
        value = value.replace("-pose", "pose")
        value = value.replace("-cls", "cls")
        value = value.replace("-obb", "obb")
        value = value.replace("yoloe-", "yolo")
        value = value.replace("-worldv2", "")
        value = value.replace("-world", "")
        if "yolov6" in value:
            value = re.sub(r"yolov6([nsmlx])", "yolov7x", value)
        if "yolo12" in value and ("cls" in value or "pose" in value):
            value = value.replace("yolo12", "yolo11")
        return value

    def guess_task(name: str) -> str:
        normalized = normalize_model_name(name)
        if "seg" in normalized:
            return "seg"
        if "pose" in normalized:
            return "pose"
        if "cls" in normalized:
            return "cls"
        if "obb" in normalized:
            return "obb"
        if "yolo" in normalized:
            return "det"
        return "unknown"

    modules = {
        "det": "mblt_model_zoo.vision.object_detection",
        "seg": "mblt_model_zoo.vision.instance_segmentation",
        "pose": "mblt_model_zoo.vision.pose_estimation",
        "cls": "mblt_model_zoo.vision.image_classification",
    }
    task = guess_task(model_name)
    if task not in modules:
        return {}

    cfgs = {}
    for cfg_class in import_cfg_classes(modules[task]):
        if hasattr(cfg_class, "DEFAULT"):
            cfgs[normalize_cfg_name(cfg_class.__name__)] = cfg_class

    normalized_name = normalize_model_name(model_name)
    if normalized_name in cfgs:
        return deepcopy(cfgs[normalized_name].DEFAULT.value.post_cfg)

    if "p6" in normalized_name or "p2" in normalized_name:
        alt_name = normalized_name.replace("p6", "").replace("p2", "")
        if alt_name in cfgs:
            cfg = deepcopy(cfgs[alt_name].DEFAULT.value.post_cfg)
            cfg["nl"] = 4
            return cfg

    return {}


def pad_mobilint_detections(detections: list[torch.Tensor]) -> torch.Tensor:
    """Pad mblt-model-zoo per-image detections to Ultralytics end-to-end BNC format."""
    if not detections:
        return torch.zeros((0, 0, 6))

    cols = max((det.shape[1] for det in detections if det.ndim == 2), default=6)
    rows = max((det.shape[0] for det in detections if det.ndim == 2), default=0)
    padded = detections[0].new_zeros((len(detections), rows, cols))
    for i, det in enumerate(detections):
        if det.numel():
            padded[i, : det.shape[0], : det.shape[1]] = det
    return padded


def format_mobilint_segmentation(segments: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    """Adapt mblt-model-zoo segmentation output to Ultralytics' existing segment predictor input format.

    mblt-model-zoo emits per-detection masks at the input resolution (e.g. 640x640). Standard
    Ultralytics YOLO segment expects protos at imgsz/4 (stride-4), and `SegmentationValidator`
    relies on this convention via `imgsz = [4 * x for x in proto.shape[2:]]`. Downsample the
    masks accordingly so the unmodified val/predict postprocess chain handles them correctly.
    """
    import torch.nn.functional as F

    if not segments:
        return [torch.zeros((0, 0, 7)), torch.zeros((0, 1, 1, 1))]

    max_masks = max((item[0].shape[0] for item in segments), default=0)
    max_masks = max(max_masks, 1)
    device = segments[0][0].device
    dtype = segments[0][0].dtype

    raw_h, raw_w = next((item[1].shape[-2:] for item in segments if item[1].numel()), (4, 4))
    proto_h, proto_w = max(raw_h // 4, 1), max(raw_w // 4, 1)
    mask_shape = (proto_h, proto_w)

    preds = torch.zeros((len(segments), max_masks, 6 + max_masks), dtype=dtype, device=device)
    protos = torch.zeros((len(segments), max_masks, *mask_shape), dtype=dtype, device=device)

    for batch_idx, (det, masks) in enumerate(segments):
        num_masks = det.shape[0]
        if num_masks == 0:
            continue
        preds[batch_idx, :num_masks, :6] = det[:, :6]
        preds[batch_idx, torch.arange(num_masks, device=device), 6 + torch.arange(num_masks, device=device)] = 1
        m = masks[:num_masks].to(device=device)
        if m.shape[-2:] != mask_shape:
            m = F.interpolate(m.unsqueeze(0).float(), mask_shape, mode="bilinear", align_corners=False).squeeze(0)
        protos[batch_idx, :num_masks] = m.to(dtype=dtype)

    return [preds, protos]


def format_mobilint_postprocess_output(output, task: str):
    """Adapt mblt-model-zoo postprocess output to the tensors expected by Ultralytics predictors."""
    if task == "segment":
        return format_mobilint_segmentation(output)
    return pad_mobilint_detections(output)
