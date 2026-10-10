#!/usr/bin/env python3
"""Freeze the v21 business preprocessing + fixed640 ONNX input contract.

Camera images need transpose/resize; saved data_collect images already have
business geometry. The stage is deliberately explicit. No model is run here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HEADREST_POLYGON = np.array([[240, 0], [360, 0], [360, 200], [300, 100]], np.int32)
WINDOW_POLYGON = np.array([[0, 150], [65, 106], [95, 237], [6, 327]], np.int32)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resize_business_frame(frame):
    frame = np.transpose(frame, (1, 0, 2))
    h, w = frame.shape[:2]
    # imutils.resize performs this call even when the dimensions are unchanged.
    return cv2.resize(frame, (int(w * (640.0 / h)), 640), interpolation=cv2.INTER_AREA)


def mean_region_mask(rgb, polygon):
    output = rgb.copy()
    # Match historical per-channel float64 mean -> assignment to uint8 exactly.
    means = np.array([np.mean(output[:, :, c]) for c in range(3)]).astype(np.uint8)
    mask = np.zeros(output.shape[:2], np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    output[mask != 0] = means
    return output


def apply_business_masks(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = mean_region_mask(rgb, HEADREST_POLYGON)
    rgb = mean_region_mask(rgb, WINDOW_POLYGON)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def letterbox_fixed(image):
    h, w = image.shape[:2]
    ratio = min(640 / h, 640 / w)
    new_w, new_h = round(w * ratio), round(h * ratio)
    dw, dh = (640 - new_w) / 2, (640 - new_h) / 2
    if (new_w, new_h) != (w, h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left, right = round(dw - 0.1), round(dw + 0.1)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image, ratio, (left, top, right, bottom)


def to_tensor(image_bgr):
    tensor = np.ascontiguousarray(image_bgr[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= np.float32(255.0)
    return tensor


def prepare_reference(original, input_stage):
    if original.dtype != np.uint8 or original.ndim != 3 or original.shape[2] != 3 or not original.size:
        raise ValueError("Expected a nonempty uint8 HWC BGR image")
    if input_stage not in ("camera", "business"):
        raise ValueError("input_stage must be camera or business")
    business = resize_business_frame(original) if input_stage == "camera" else original.copy()
    # The historical masks use fixed coordinates; do not silently apply them to
    # another geometry. Supporting another camera needs a separate contract.
    if business.shape != (640, 360, 3):
        raise ValueError("v21 masks require business HWC=(640,360,3), got %s; check input stage/geometry" % (business.shape,))
    masked = apply_business_masks(business)
    letterboxed, ratio, padding = letterbox_fixed(masked)
    tensor = to_tensor(letterboxed)
    return business, masked, letterboxed, tensor, ratio, padding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/images/test.jpg")
    parser.add_argument("--input-stage", choices=("camera", "business"), required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/reference")
    args = parser.parse_args()
    # Never accidentally replace an already approved input or leave stale golden
    # outputs beside a different input. Use a new directory for another sample.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; use a new reference directory")
    original = cv2.imread(str(args.input), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if original is None:
        parser.error("Cannot decode image: %s" % args.input)
    business, masked, letterboxed, tensor, ratio, padding = prepare_reference(original, args.input_stage)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, pixels in (("business", business), ("masked", masked), ("letterbox", letterboxed)):
        if not cv2.imwrite(str(args.output_dir / (name + ".png")), pixels):
            raise RuntimeError("Failed to save %s" % name)
    np.save(args.output_dir / "input.npy", tensor, allow_pickle=False)
    tensor.astype("<f4", copy=False).tofile(args.output_dir / "input.bin")
    files = ["business.png", "masked.png", "letterbox.png", "input.npy", "input.bin"]
    metadata = {
        "schema_version": 1, "contract": "vh3-v21-fixed640-v1",
        "source_path": str(args.input.resolve()), "source_sha256": sha256(args.input),
        "input_stage": args.input_stage, "source_shape_hwc": list(original.shape),
        "business_shape_hwc": list(business.shape), "model_shape": list(tensor.shape),
        "camera_transform_applied": args.input_stage == "camera",
        "camera_transform": "transpose H/W, resize height=640 INTER_AREA" if args.input_stage == "camera" else None,
        "source_to_business_scale_xy_after_transpose": [business.shape[1] / original.shape[0], business.shape[0] / original.shape[1]] if args.input_stage == "camera" else None,
        "mask_order": ["headrest", "window"], "headrest_polygon_xy": HEADREST_POLYGON.tolist(),
        "window_polygon_xy": WINDOW_POLYGON.tolist(), "mask_fill": "current whole-image per-channel mean, truncate to uint8; recompute for each mask",
        "letterbox": {"shape_hw": [640, 640], "auto": False, "center": True, "scale_fill": False, "scaleup": True, "ratio": ratio, "padding_ltrb": list(padding), "color_bgr": [114, 114, 114], "resize_interpolation": "INTER_LINEAR"},
        "tensor": {"layout": "NCHW", "color": "RGB", "dtype": "float32", "binary_dtype": "<f4", "normalization": "uint8 / 255; no mean/std", "contiguous": bool(tensor.flags.c_contiguous), "min": float(tensor.min()), "max": float(tensor.max()), "binary_bytes": tensor.nbytes},
        "image_decoder": "cv2.IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION; no automatic EXIF rotation",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
        "script_sha256": sha256(__file__),
        "artifacts_sha256": {name: sha256(args.output_dir / name) for name in files},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"source": list(original.shape), "business": list(business.shape), "tensor": list(tensor.shape), "dtype": str(tensor.dtype), "ratio": ratio, "padding_ltrb": padding, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
