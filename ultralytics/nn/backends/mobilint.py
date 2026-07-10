# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.export.mxq import format_mobilint_postprocess_output

from .base import BaseBackend

# Ultralytics task name → mblt-model-zoo `post_cfg["task"]` value expected by `build_postprocess`.
_TASK_TO_MBLT = {
    "detect": "object_detection",
    "segment": "instance_segmentation",
    "pose": "pose_estimation",
    "classify": "image_classification",
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
        coefficients, optional proto). All other information — class names, task type,
        input size, head params (`reg_max` / `nl` / `nm` / `kpt_shape`) — comes from a
        `metadata.yaml` sidecar written by the Ultralytics MXQ exporter and located next
        to the `.mxq` file (typically inside a `*_mobilint_model/` directory).

        The sidecar is **required**. Filename-based heuristics (e.g. `mblt-model-zoo`
        lookup keyed by stem) are intentionally not used — they fail on custom-named
        artifacts and silently mis-decode custom-trained models with `nc != 80`.

        Args:
            weight (str | Path): Path to a `*_mobilint_model/` directory (preferred), or
                directly to the `.mxq` file inside one. The directory must also contain
                `metadata.yaml`.
        """
        check_requirements("mobilint-qb-runtime==1.2.0")
        check_requirements("mblt_model_zoo==2.1.2")
        from mblt_model_zoo.vision.utils.postprocess import build_postprocess
        from qbruntime import Accelerator, Model, ModelConfig

        w = Path(weight)
        mxq_file = next(w.rglob("*.mxq"), None) if w.is_dir() else (w if w.suffix == ".mxq" else None)
        if mxq_file is None or not mxq_file.exists():
            raise FileNotFoundError(
                f"No .mxq file found at: {w}. For yolo*.mxq, pass `target=` (and `core_mode=` "
                "for inference) to `model.predict(...)` to trigger Hugging Face auto-download, or "
                "manually call `attempt_download_mxq(name, device=...)` and pass the local path."
            )

        sidecar = mxq_file.parent / "metadata.yaml"
        if not sidecar.exists():
            raise FileNotFoundError(
                f"MXQ sidecar `metadata.yaml` not found at {sidecar}. The Ultralytics MXQ "
                "exporter writes it alongside the `.mxq` inside `<stem>_mobilint_model/`. "
                "Re-export with `model.export(format='mxq', ...)` to produce the sidecar, "
                "or move the `.mxq` into a folder containing its matching `metadata.yaml`."
            )
        self.apply_metadata(YAML.load(sidecar))

        accelerator = Accelerator()
        model_config = ModelConfig()
        self._apply_core_mode(model_config)
        self.model = Model(str(mxq_file), model_config)
        self.model.launch(accelerator)

        # mblt-model-zoo postprocess currently emits NMS-fused output, so Ultralytics'
        # predictor must take its end2end shortcut (skip NMS) — hence True. Flip to False
        # once mblt-model-zoo is updated to emit raw NMS-free output and the backend
        # forwards that through unchanged.
        self.end2end = True

        if self.task is None:
            raise ValueError(
                f"`metadata.yaml` at {sidecar} is missing the required `task` field. "
                "Re-export with the Ultralytics MXQ exporter."
            )
        if self.task == "pose" and not getattr(self, "kpt_shape", None):
            self.kpt_shape = [17, 3]

        # mblt-model-zoo postprocess pipeline used by forward()
        imgsz = getattr(self, "imgsz", None) or [640, 640]
        pre_cfg = {"LetterBox": {"img_size": list(imgsz)}}
        self._mobilint_pp = build_postprocess(pre_cfg, self._build_post_cfg())

    def _build_post_cfg(self) -> dict:
        """Assemble the mblt-model-zoo `post_cfg` for `build_postprocess`.

        All values come from `self.metadata` (sidecar yaml). `nc` is derived from
        `self.names` so custom-trained models with non-COCO class counts split the head
        channels correctly. `reg_max <= 1` is mapped to mblt-model-zoo's `dflfree` flag
        (YOLO26-style raw box deltas).
        """
        meta = self.metadata or {}
        cfg = {
            "task": _TASK_TO_MBLT[self.task],
            "nc": len(self.names),
            "nl": meta.get("nl", 3),
        }
        reg_max = meta.get("reg_max")
        if reg_max is not None and reg_max > 1:
            cfg["reg_max"] = reg_max
        else:
            cfg["dflfree"] = True
        if self.task == "segment":
            cfg["n_extra"] = meta.get("nm", 32)
        elif self.task == "pose":
            kh, kw = self.kpt_shape or (17, 3)
            cfg["n_extra"] = kh * kw
        return cfg

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
