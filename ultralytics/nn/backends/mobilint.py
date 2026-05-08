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
}


class MobilintBackend(BaseBackend):
    """Mobilint mxq inference backend for Mobilint AI accelerators.

    Loads compiled Mobilint models (.mxq files) and runs inference using the Mobilint runtime SDK.
    Postprocessing is delegated to mblt-model-zoo and adapted to the tensor shapes Ultralytics
    predictors expect via `format_mobilint_postprocess_output`.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a Mobilint model and build the matching postprocess pipeline.

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
        model_config.set_single_core_mode(1)
        self.model = Model(str(mxq_file), model_config)
        self.model.launch(accelerator)

        # Resolve postprocess config from mblt-model-zoo by model name; fall back to YOLO detection defaults
        model_name = mxq_file.stem
        post_cfg = get_mobilint_model_zoo_post_cfg(model_name)

        self._post_task = post_cfg.get("task")
        task = _TASK_MAP.get(self._post_task)
        if task is None:
            raise NotImplementedError(f"Mobilint MXQ inference does not support task={self._post_task!r}.")
        self.task = task
        if self.task == "pose":
            self.kpt_shape = [17, 3]

        pre_cfg = {"LetterBox": {"img_size": [640, 640]}}
        self._mobilint_pp = build_postprocess(pre_cfg, post_cfg)

        # Optional metadata.yaml beside the .mxq file overrides defaults
        metadata_file = mxq_file.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run inference on the Mobilint hardware accelerator.

        Args:
            im (torch.Tensor): Input image tensor in BHWC format, normalized to [0, 1].

        Returns:
            (list): Model predictions as a list of output arrays.
        """
        im = im.cpu().numpy() * 255
        im = im.astype("uint8")
        output = self.model.infer(im)
        output = self._mobilint_pp(output, 0.25, 0.45)
        return format_mobilint_postprocess_output(output, self.task)
