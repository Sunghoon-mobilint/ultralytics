# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

from ultralytics.utils import LOGGER


def onnx2mxq(
    onnx_file: str | Path,
    save_path: str | Path,
    calib_path: str | None = None,
    use_random_calib: bool = False,
    device: str = "cpu",
    imgsz: tuple[int, int] = (640, 640),
    prefix: str = "",
) -> None:
    """Compile an ONNX model to Mobilint MXQ format using qbcompiler.

    Args:
        onnx_file (str | Path): Path to the ONNX model to compile.
        save_path (str | Path): Output path for the compiled .mxq file.
        calib_path (str | None, optional): Directory containing calibration `.npy` samples. If
            None, calibration data is not used.
        use_random_calib (bool, optional): If True, use random data for calibration. Mutually
            exclusive with `calib_path`.
        device (str, optional): Compile device, "cpu" or "gpu".
        imgsz (tuple[int, int], optional): (height, width) used by the letterbox preprocessing op.
        prefix (str, optional): Prefix string for log messages.
    """
    from qbcompiler import CalibrationConfig, PreprocessingConfig, mxq_compile

    onnx_file = str(onnx_file)
    save_path = str(save_path)
    height, width = imgsz

    preprocessing_config = PreprocessingConfig(
        apply=True,
        auto_convert_format=True,
        pipeline=[
            {"op": "letterbox", "height": height, "width": width, "padValue": 114},
            {"op": "normalize", "mean": [0, 0, 0], "std": [255, 255, 255]},
        ],
        input_configs={},
    )
    calibration_config = CalibrationConfig(
        method=1,  # 0 for per tensor, 1 for per channel
        output=1,  # 0 for layer, 1 for channel
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
        inference_scheme="all",
        backend="onnx",
        device=device,
        preprocessing_config=preprocessing_config,
        calibration_config=calibration_config,
    )
    LOGGER.info(f"{prefix} mxq_compile completed successfully.")
