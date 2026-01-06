# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
from copy import deepcopy
from pathlib import Path

import torch

# Mapping: local .mxq filename -> Hugging Face repo id under the `mobilint/` org.
# The compiled model is expected at `aries/<filename>` inside each repo.
MXQ_HF_REPOS: dict[str, str] = {
    "yolov10s.mxq": "mobilint/YOLOv10s",
}


def attempt_download_mxq(file: str | Path) -> str:
    """Resolve a `.mxq` model path, downloading from Hugging Face Hub if needed.

    Resolution order:
      1. Return the path if the file already exists.
      2. Return `<weights_dir>/<name>` if cached there from a prior download.
      3. Look up `<name>` in `MXQ_HF_REPOS` and download `aries/<name>` from the matching
         Hugging Face repo into `<weights_dir>`.
      4. If the name is unknown, return the original path unchanged (caller will surface the
         missing-file error).

    Args:
        file (str | Path): Local `.mxq` path or bare filename (e.g. `"yolov10s.mxq"`).

    Returns:
        (str): Path to the resolved (and possibly downloaded) file.
    """
    from ultralytics.utils import LOGGER, SETTINGS  # scoped to avoid circular import

    path = Path(str(file).strip().replace("'", ""))
    if path.exists():
        return str(path)

    weights_dir = Path(SETTINGS["weights_dir"])
    cached = weights_dir / path.name
    if cached.exists():
        return str(cached)

    repo_id = MXQ_HF_REPOS.get(path.name)
    if repo_id is None:
        return str(path)  # unknown name; let the caller raise a clear error

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        from ultralytics.utils.checks import check_requirements

        check_requirements("huggingface_hub")
        from huggingface_hub import hf_hub_download

    weights_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Downloading {path.name} from Hugging Face repo {repo_id}...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=f"aries/{path.name}",
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
    """Adapt mblt-model-zoo segmentation output to Ultralytics' existing segment predictor input format."""
    if not segments:
        return [torch.zeros((0, 0, 7)), torch.zeros((0, 1, 1, 1))]

    max_masks = max((item[0].shape[0] for item in segments), default=0)
    max_masks = max(max_masks, 1)
    device = segments[0][0].device
    dtype = segments[0][0].dtype
    mask_shape = next((item[1].shape[-2:] for item in segments if item[1].numel()), (1, 1))

    preds = torch.zeros((len(segments), max_masks, 6 + max_masks), dtype=dtype, device=device)
    protos = torch.zeros((len(segments), max_masks, *mask_shape), dtype=dtype, device=device)

    for batch_idx, (det, masks) in enumerate(segments):
        num_masks = det.shape[0]
        if num_masks == 0:
            continue
        preds[batch_idx, :num_masks, :6] = det[:, :6]
        preds[batch_idx, torch.arange(num_masks, device=device), 6 + torch.arange(num_masks, device=device)] = 1
        protos[batch_idx, :num_masks] = masks[:num_masks].to(device=device, dtype=dtype)

    return [preds, protos]


def format_mobilint_postprocess_output(output, task: str):
    """Adapt mblt-model-zoo postprocess output to the tensors expected by Ultralytics predictors."""
    if task == "segment":
        return format_mobilint_segmentation(output)
    return pad_mobilint_detections(output)
