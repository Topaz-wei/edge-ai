#!/usr/bin/env python3
"""Prepare (default) or execute a pinned Polygraphy coarse single-image search."""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from precision_search import ROOT, validate_cache, sha
from precision_check import add_gate_args, gate_kwargs, load_output, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["bisect", "linear"], default="bisect")
    p.add_argument("--direction", choices=["forward", "reverse"], default="reverse")
    p.add_argument("--max-builds", type=int, default=12)
    p.add_argument(
        "--run", action="store_true", help="Actually build engines (can take hours)"
    )
    p.add_argument("--model", type=Path, default=ROOT / "models/yolov8m.onnx")
    p.add_argument(
        "--cache", type=Path, default=ROOT / "cache/int8-baseline-v1/yolov8m_int8.cache"
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "data/splits/person_temporal_v1/calibration",
    )
    p.add_argument("--input", type=Path, default=ROOT / "data/reference/input.bin")
    p.add_argument(
        "--reference", type=Path, default=ROOT / "data/outputs/tensorrt_fp16_output.bin"
    )
    add_gate_args(p)
    args = p.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.max_builds < 1:
        p.error("max-builds must be positive")
    if args.work_dir.exists():
        p.error("work-dir must be new")

    # Pin the project-local tool instead of inheriting a system installation.
    private = ROOT / ".deps/polygraphy"
    sys.path.insert(0, str(private))
    import polygraphy

    if polygraphy.__version__ != "0.49.9":
        raise ValueError("Expected project-private Polygraphy 0.49.9")
    n = validate_cache(args.model, args.cache, args.calibration)
    ref = load_output(str(args.reference))
    evaluate(ref, ref, **gate_kwargs(args))
    if args.input.stat().st_size != 4915200:
        raise ValueError("Invalid fixed input size")

    # Record the immutable inputs before preparing a deferred or immediate run.
    args.work_dir.mkdir(parents=True)
    spec = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    spec["polygraphy"] = polygraphy.__version__
    spec["calibration_count"] = n
    spec["hashes"] = {
        k: sha(getattr(args, k)) for k in ("model", "cache", "input", "reference")
    }
    spec_path = args.work_dir / "experiment.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(private)
    env["YOLO_PRECISION_SPEC"] = str(spec_path)
    env["POLYGRAPHY_AUTOINSTALL_DEPS"] = "0"
    env["PYTHONUNBUFFERED"] = "1"

    # Keep checker arguments separate: everything after --check belongs to it.
    checker = [
        sys.executable,
        str(ROOT / "scripts/precision_check.py"),
        "--engine",
        "polygraphy_debug.engine",
        "--input",
        str(args.input),
        "--reference",
        str(args.reference),
        "--report",
        "check.json",
    ]
    for key, value in gate_kwargs(args).items():
        checker.extend(["--" + key.replace("_", "-"), str(value)])

    command = [
        "/usr/bin/python3",
        str(ROOT / "scripts/polygraphy_entry.py"),
        "debug",
        "precision",
        str(args.model),
        "--int8",
        "--fp16",
        "--calibration-cache",
        str(args.cache),
        "--precision",
        "float16",
        "--mode",
        args.mode,
        "--direction",
        args.direction,
        "--trt-config-script",
        str(ROOT / "scripts/polygraphy_config.py"),
        "--save-timing-cache",
        "timing.cache",
        "--show-output",
        "--no-remove-intermediate",
        "--save-debug-replay",
        "replay.json",
        "--iter-info",
        "iteration.json",
        "--artifacts",
        "polygraphy_debug.engine",
        "fp16_layers.txt",
        "check.json",
        "check.output.bin",
        "check.infer.log",
        "--art-dir",
        "artifacts",
        "--check",
    ] + checker
    (args.work_dir / "command.json").write_text(
        json.dumps(
            {
                "command": command,
                "env": {
                    k: env[k]
                    for k in (
                        "PYTHONPATH",
                        "YOLO_PRECISION_SPEC",
                        "POLYGRAPHY_AUTOINSTALL_DEPS",
                        "PYTHONUNBUFFERED",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )

    # Both execution paths go through the same hash checks and log supervisor.
    script = (
        "#!/bin/bash\nset -euo pipefail\nexec "
        + shlex.join(
            [
                "/usr/bin/python3",
                str(ROOT / "scripts/polygraphy_supervise.py"),
                str(args.work_dir),
            ]
        )
        + "\n"
    )
    (args.work_dir / "run.sh").write_text(script)
    print("Prepared:", args.work_dir / "run.sh", flush=True)
    if not args.run:
        return 0
    from polygraphy_supervise import supervise

    code = supervise(args.work_dir)
    print("Polygraphy exit:", code, "(3 means build budget reached)")
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(2)
