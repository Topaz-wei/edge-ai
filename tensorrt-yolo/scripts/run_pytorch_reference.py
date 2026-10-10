#!/usr/bin/env python3
"""Run the underlying YOLOv8m model on a frozen fixed640 FP32 tensor; no NMS."""
import argparse
import json
import os
from pathlib import Path
import platform

import numpy as np

from prepare_yolo_reference import ROOT, sha256


def validate_input(tensor):
    if tensor.shape != (1, 3, 640, 640) or tensor.dtype != np.float32:
        raise ValueError("Expected FP32 [1,3,640,640]")
    if not np.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
        raise ValueError("Input must be finite RGB FP32 in [0,1]")
    return np.ascontiguousarray(tensor)


def select_raw_output(output):
    # Non-export Detect.forward returns (decoded predictions, feature maps).
    if isinstance(output, (tuple, list)):
        output = output[0]
    result = output.detach().float().cpu().numpy()
    if result.shape != (1, 84, 8400) or not np.isfinite(result).all():
        raise ValueError("Expected finite decoded pre-NMS [1,84,8400], got %s" % (result.shape,))
    return np.ascontiguousarray(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "data/reference")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/yolov8m.pt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error("Weights do not exist; automatic download is disabled")
    directory = args.reference_dir
    outputs = ["pytorch_output.npy", "pytorch_output.bin", "pytorch_metadata.json"]
    if any((directory / name).exists() for name in outputs):
        parser.error("Golden output already exists; use a new reference directory")
    metadata = json.loads((directory / "metadata.json").read_text())
    for name in ("input.npy", "input.bin"):
        if sha256(directory / name) != metadata["artifacts_sha256"][name]:
            raise ValueError("Reference input hash mismatch: " + name)
    tensor = validate_input(np.load(directory / "input.npy", allow_pickle=False))
    binary = np.fromfile(directory / "input.bin", dtype="<f4")
    if binary.size != tensor.size or not np.array_equal(tensor.ravel(), binary):
        raise ValueError("input.bin and input.npy disagree")

    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/tensorrt-yolo-ultralytics")
    import torch
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != "8.3.173":
        raise RuntimeError("Reference requires ultralytics==8.3.173, got " + ultralytics.__version__)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; request GPU access or explicitly choose --device cpu")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    model = YOLO(str(args.weights)).model.to(device).float().eval()
    x = torch.from_numpy(tensor).to(device)
    with torch.inference_mode():
        result = select_raw_output(model(x))
        repeated = select_raw_output(model(x))
    if not np.array_equal(result, repeated):
        raise RuntimeError("Repeated same-process inference is not bitwise reproducible")
    np.save(directory / "pytorch_output.npy", result, allow_pickle=False)
    result.astype("<f4", copy=False).tofile(directory / "pytorch_output.bin")
    summary = {
        "schema_version": 1, "contract": metadata["contract"],
        "weights_path": str(args.weights.resolve()), "weights_sha256": sha256(args.weights),
        "input_npy_sha256": sha256(directory / "input.npy"), "input_bin_sha256": sha256(directory / "input.bin"),
        "preprocess_metadata_sha256": sha256(directory / "metadata.json"),
        "shape": list(result.shape), "dtype": "float32", "binary_dtype": "<f4",
        "layout": "N,C,anchors", "semantics": "decoded pre-NMS: xywh in model pixels + 80 class probabilities; no separate objectness",
        "device": args.device, "device_name": torch.cuda.get_device_name(device) if args.device == "cuda" else platform.processor(),
        "tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True,
        "model_mode": "eval, float32, normal Detect.forward; output[0]; no Predictor, AMP or NMS",
        "model_state": {"is_fused": bool(model.is_fused()), "head_class": type(model.model[-1]).__name__, "head_export": bool(getattr(model.model[-1], "export", False))},
        "same_process_repeat_bitwise_equal": True,
        "min": float(result.min()), "max": float(result.max()),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "ultralytics": ultralytics.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
        "script_sha256": sha256(__file__),
        "artifacts_sha256": {name: sha256(directory / name) for name in outputs[:2]},
    }
    (directory / "pytorch_metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("shape", "dtype", "device", "device_name", "min", "max", "same_process_repeat_bitwise_equal")}, indent=2))


if __name__ == "__main__":
    main()
