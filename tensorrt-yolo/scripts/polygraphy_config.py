"""TensorRT 8.5 config callback for the project-private Polygraphy launcher."""

import json
import os
from pathlib import Path
import tensorrt as trt

# TensorRT holds native references; keep the Python owners alive through builds.
_KEEP = []


class CacheOnlyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, path):
        super().__init__()
        self.data = Path(path).read_bytes()
        if not self.data:
            raise ValueError("Empty calibration cache")

    def get_batch_size(self):
        return 1

    # A rejected cache must abort, not silently calibrate on the debug image.
    def get_batch(self, names):
        raise RuntimeError(
            "Calibration cache rejected; single-image recalibration is forbidden"
        )

    def read_calibration_cache(self):
        return self.data

    def write_calibration_cache(self, cache):
        raise RuntimeError("Cache-only calibration must not recalibrate")


def load_config(builder, network):
    spec = json.loads(Path(os.environ["YOLO_PRECISION_SPEC"]).read_text())
    eligible_types = {
        trt.LayerType.CONVOLUTION,
        trt.LayerType.ACTIVATION,
        trt.LayerType.ELEMENTWISE,
        trt.LayerType.SOFTMAX,
        trt.LayerType.POOLING,
        trt.LayerType.RESIZE,
    }
    # Polygraphy selects network ranges. Export only the eligible compute
    # layers actually constrained, using the same policy as the C++ builder.
    names = []
    for layer in network:
        outputs = [layer.get_output(i) for i in range(layer.num_outputs)]
        eligible = (
            layer.type in eligible_types
            and any(t.dtype in (trt.float32, trt.float16) for t in outputs)
            and all(t.is_execution_tensor and not t.is_shape_tensor for t in outputs)
        )
        if not eligible:
            layer.reset_precision()
        elif layer.precision_is_set:
            if layer.precision != trt.float16:
                raise ValueError("Expected FP16 constraint")
            names.append(layer.name)
    Path("fp16_layers.txt").write_text("".join(name + "\n" for name in names))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    if names:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    calibrator = CacheOnlyCalibrator(spec["cache"])
    config.int8_calibrator = calibrator
    # Profiling timings and activation calibration scales are separate caches.
    timing = Path("timing.cache")
    cache = config.create_timing_cache(timing.read_bytes() if timing.exists() else b"")
    if not config.set_timing_cache(cache, False):
        raise ValueError("Incompatible timing cache")
    _KEEP.extend([calibrator, cache])
    return config
