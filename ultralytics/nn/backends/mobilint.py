# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.export.mxq import (
    format_mobilint_postprocess_output,
    get_mobilint_model_zoo_post_cfg,
)

from .base import BaseBackend

_TASK_MAP = {
    "object_detection": "detect",
    "instance_segmentation": "segment",
    "pose_estimation": "pose",
    "image_classification": "classify",
}

_DEFAULT_DATA_FOR_TASK = {
    "detect": "coco.yaml",
    "segment": "coco.yaml",
    "pose": "coco-pose.yaml",
    "classify": "ImageNet.yaml",
}


def _as_list(v: int | list[int] | None) -> list[int]:
    """Normalize an int|list|None into a list[int] (None → [])."""
    if v is None:
        return []
    return [v] if isinstance(v, int) else list(v)


class MobilintBackend(BaseBackend):
    """Mobilint mxq inference backend for Mobilint AI accelerators.

    Loads compiled Mobilint models (.mxq files) and runs inference using the Mobilint runtime SDK.
    Postprocessing is delegated to mblt-model-zoo and adapted to the tensor shapes Ultralytics
    predictors expect via `format_mobilint_postprocess_output`.

    Inference is dispatched onto the ARIES NPU according to `core_mode`, optionally constrained to
    specific cores via `cluster_id` / `core_id`. ARIES is laid out as 2 clusters of 4 cores each
    (8 cores total). See https://docs.mobilint.com/v1.2/en/multicore.html for details.
    """

    def __init__(
        self,
        weight: str | torch.nn.Module,
        device: torch.device | str,
        fp16: bool = False,
        core_mode: str | None = None,
        cluster_id: int | list[int] | None = None,
        core_id: int | list[int] | None = None,
    ):
        """Initialize the MXQ backend, capturing core-allocation args before model load.

        Args:
            weight: Path to a `.mxq` file or directory containing one.
            device: Torch device (unused by qbruntime but kept for BaseBackend compat).
            fp16: Unused; MXQ models are always INT8.
            core_mode: 'single' | 'multi' | 'global4' | 'global8'. Required.
            cluster_id: Cluster ID(s) — int or list. Required except for `core_mode='global8'`.
            core_id: Core ID(s) within cluster — int or list. Required for `core_mode='single'`.
        """
        if core_mode is None:
            raise ValueError(
                "MXQ inference requires 'core_mode=' "
                "(one of 'single' | 'multi' | 'global4' | 'global8')."
            )
        self._core_mode = core_mode
        self._cluster_id = cluster_id
        self._core_id = core_id
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str | Path) -> None:
        """Load a Mobilint MXQ model and resolve task / metadata.

        The .mxq itself contains only compiled head features (boxes, scores, optional mask
        coefficients, optional proto). Class names, task type, and head dimensions come from
        — in priority order:

        1. Sidecar `<stem>.yaml` written by the Ultralytics MXQ exporter.
        2. mblt-model-zoo `post_cfg` looked up by model stem (for HF pre-compiled YOLO models).
        3. Hardcoded YOLO11/12 defaults (`reg_max=16`, `nl=3`) + COCO/ImageNet class names.

        Args:
            weight (str | Path): Path to a `.mxq` file or a directory containing one.
        """
        check_requirements("mobilint-qb-runtime==1.2.0")
        check_requirements("mblt_model_zoo==1.4.2")
        from mblt_model_zoo.vision.utils.postprocess import build_postprocess
        from qbruntime import Accelerator, Model, ModelConfig

        w = Path(weight)
        mxq_file = next(w.rglob("*.mxq"), None) if w.is_dir() else (w if w.suffix == ".mxq" else None)
        if mxq_file is None or not mxq_file.exists():
            raise FileNotFoundError(
                f"No .mxq file found at: {w}. For yolo*.mxq, pass `core_mode=` (and optional "
                "`target=`) to `model.predict(...)` to trigger Hugging Face auto-download, or "
                "manually call `attempt_download_mxq(name, device=..., core_mode=...)` and pass "
                "the local path."
            )

        accelerator = Accelerator()
        model_config = ModelConfig()
        self._apply_core_mode(model_config)
        self.model = Model(str(mxq_file), model_config)
        self.model.launch(accelerator)

        # mblt-model-zoo postprocess emits NMS-fused output, so the predictor's NMS must use
        # its end2end shortcut. Force end2end=True regardless of what the sidecar reports —
        # the original .pt's end2end flag describes the pre-export model, not the MXQ output.
        self.end2end = True

        # ── 2. mblt-model-zoo post_cfg (defaults: nc / nl / reg_max for known YOLO models) ──
        post_cfg = get_mobilint_model_zoo_post_cfg(mxq_file.stem)

        # Build the mblt-model-zoo postprocess pipeline used by forward()
        pre_cfg = {"LetterBox": {"img_size": [640, 640]}}
        self._mobilint_pp = build_postprocess(pre_cfg, post_cfg)

        # ── 3. Resolve task (sidecar > post_cfg) ────────────────────────────────────────────
        if not self.task:
            zoo_task = _TASK_MAP.get(post_cfg.get("task"))
            if zoo_task is None:
                raise NotImplementedError(
                    f"Cannot determine task for {mxq_file.name}. Ensure a sidecar "
                    f"{mxq_file.with_suffix('.yaml').name} is present (written by the "
                    f"Ultralytics MXQ exporter), or use a model name recognized by "
                    f"mblt-model-zoo."
                )
            self.task = zoo_task

        if self.task == "pose" and not getattr(self, "kpt_shape", None):
            self.kpt_shape = [17, 3]

        # ── 4. Fallback class names (HF pre-compiled / sidecar without names) ───────────────
        if not self.names:
            from ultralytics.nn.autobackend import default_class_names

            default_data = _DEFAULT_DATA_FOR_TASK.get(self.task)
            if default_data is not None:
                self.names = default_class_names(default_data)
                LOGGER.info(
                    f"MXQ: no sidecar class names for {mxq_file.name}; using defaults "
                    f"from {default_data}. For custom-trained models, re-export with the "
                    f"Ultralytics MXQ exporter to embed your model's class names."
                )

    def _apply_core_mode(self, model_config) -> None:
        """Configure qbruntime ModelConfig for the requested core_mode / cluster_id / core_id.

        Mapping to qbruntime:
            single  → set_single_core_mode(core_ids=[CoreId(cluster, core), ...])
            multi   → set_multi_core_mode([Cluster, ...])
            global4 → set_global4_core_mode([Cluster])
            global8 → set_global8_core_mode()
        """
        from qbruntime import Cluster, Core, CoreId

        cluster_map = {0: Cluster.Cluster0, 1: Cluster.Cluster1}
        core_map = {0: Core.Core0, 1: Core.Core1, 2: Core.Core2, 3: Core.Core3}

        def to_cluster(i: int):
            if i not in cluster_map:
                raise ValueError(f"cluster_id={i} is invalid; ARIES has clusters {sorted(cluster_map)}.")
            return cluster_map[i]

        def to_core(i: int):
            if i not in core_map:
                raise ValueError(f"core_id={i} is invalid; ARIES cluster has cores {sorted(core_map)}.")
            return core_map[i]

        mode = self._core_mode
        clusters = _as_list(self._cluster_id)
        cores = _as_list(self._core_id)

        if mode == "single":
            if self._cluster_id is None or self._core_id is None:
                raise ValueError(
                    "core_mode='single' requires both cluster_id (0-1) and core_id (0-3); "
                    f"got cluster_id={self._cluster_id!r}, core_id={self._core_id!r}."
                )
            if len(clusters) != len(cores):
                raise ValueError(
                    f"core_mode='single' requires cluster_id and core_id to have the same length; "
                    f"got cluster_id={self._cluster_id!r}, core_id={self._core_id!r}."
                )
            core_ids = [CoreId(to_cluster(cl), to_core(co)) for cl, co in zip(clusters, cores)]
            model_config.set_single_core_mode(core_ids=core_ids)
        elif mode == "multi":
            if self._cluster_id is None:
                raise ValueError(
                    "core_mode='multi' requires cluster_id (0, 1, or [0, 1]); "
                    f"got cluster_id={self._cluster_id!r}."
                )
            if cores:
                LOGGER.warning("MXQ: core_id is ignored for core_mode='multi'.")
            model_config.set_multi_core_mode([to_cluster(c) for c in clusters])
        elif mode == "global4":
            if self._cluster_id is None:
                raise ValueError(
                    "core_mode='global4' requires cluster_id (0 or 1); "
                    f"got cluster_id={self._cluster_id!r}."
                )
            if len(clusters) != 1:
                raise ValueError(
                    f"core_mode='global4' requires exactly one cluster_id (e.g. cluster_id=0); "
                    f"got cluster_id={self._cluster_id!r}."
                )
            if cores:
                LOGGER.warning("MXQ: core_id is ignored for core_mode='global4'.")
            model_config.set_global4_core_mode([to_cluster(clusters[0])])
        elif mode == "global8":
            if clusters or cores:
                LOGGER.warning("MXQ: cluster_id/core_id are ignored for core_mode='global8' (uses all 8 cores).")
            model_config.set_global8_core_mode()
        elif mode == "all":
            raise ValueError(
                "core_mode='all' is an export-only value; at inference time pick one of "
                "'single' | 'multi' | 'global4' | 'global8' (models compiled with 'all' "
                "support any of them)."
            )
        else:
            raise ValueError(
                f"core_mode={mode!r} is invalid. Use 'single' | 'multi' | 'global4' | 'global8'."
            )

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run inference on the Mobilint hardware accelerator.

        Args:
            im (torch.Tensor): Input image tensor in BHWC format, normalized to [0, 1].

        Returns:
            (list): Model predictions as a list of output arrays.
        """
        if im.dtype != torch.uint8:
            im = im.cpu().numpy() * 255
            im = im.astype("uint8")
        output = self.model.infer(im)
        if self.task == "classify":
            # qbruntime returns BHWC arrays; classify's output is (B, 1, 1, nc). The
            # ClassificationPredictor expects a (B, nc) tensor, so flatten the trailing dims.
            out = output[0] if isinstance(output, (list, tuple)) else output
            if not isinstance(out, torch.Tensor):
                out = torch.from_numpy(out)
            return out.reshape(out.shape[0], -1)
        # TODO: Remove iou/conf thresholds from the backend for validation.
        # Implement end2end=false postprocess function in mblt-model-zoo.
        output = self._mobilint_pp(output, 0.25, 0.45)
        return format_mobilint_postprocess_output(output, self.task)
