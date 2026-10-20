#!/usr/bin/env python3
"""Bounded single-image layer deletion search through the deployment C++ builder."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import shutil
import sys
from pathlib import Path
from precision_check import add_gate_args, gate_kwargs, evaluate, load_output

ROOT = Path(__file__).resolve().parents[1]


def prune(seed, check, max_trials):
    if max_trials < 1:
        raise ValueError("max_trials must be positive")

    # First rebuild the seed with the deployment builder; a Polygraphy PASS
    # does not guarantee that a separately built C++ engine will also pass.
    calls = 1
    if not check(seed):
        return dict(best=None, trials=calls, budget_exhausted=False)
    best = list(seed)

    # Remove groups of constraints, then reduce the group size when no
    # deletion passes. This is a bounded greedy search, not a global optimum.
    width = max(1, math.ceil(len(best) / 2))

    while best:
        changed = False
        for begin in range(0, len(best), width):
            if calls >= max_trials:
                return dict(best=best, trials=calls, budget_exhausted=True)
            trial = best[:begin] + best[begin + width :]
            calls += 1
            if check(trial):
                best = trial
                changed = True
                break
        if changed:
            width = min(width, max(1, math.ceil(len(best) / 2)))
        elif width > 1:
            width = max(1, width // 2)
        else:
            break
    return dict(best=best, trials=calls, budget_exhausted=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_cache(model, cache, calibration):
    # Reusing scales is valid only for the recorded model, preprocessing,
    # calibration image contents and TensorRT configuration.
    cache = Path(cache)
    calibration = Path(calibration)
    meta = Path(str(cache) + ".meta").read_text().splitlines()
    if "ONNX=" + sha(model) not in meta or "CacheSHA256=" + sha(cache) not in meta:
        raise ValueError("Calibration cache/model fingerprint mismatch")
    if (
        "TensorRT=8.5.2.2" not in meta
        or "Flags=INT8,FP16-fallback,TF32-off" not in meta
    ):
        raise ValueError(
            "Calibration configuration mismatch (expected local TRT8.5.2.2 baseline)"
        )
    if "CUDA-preprocess=" + sha(ROOT / "src/cuda_yolo.cu") not in meta:
        raise ValueError("Calibration preprocessing source changed")
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    actual = {
        p.resolve(): sha(p)
        for p in calibration.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes
    }
    expected = {}
    for line in meta:
        fields = line.rsplit(" ", 1)
        if len(fields) == 2 and len(fields[1]) == 64:
            path = (calibration / fields[0]).resolve()
            if path in expected:
                raise ValueError("Duplicate calibration source")
            expected[path] = fields[1]
    if not actual or actual != expected:
        raise ValueError("Calibration image set/hash mismatch")
    return len(actual)


def run_logged(command, log):
    import shlex
    import time

    with Path(log).open("x") as stream:
        stream.write("Command: " + shlex.join(list(map(str, command))) + "\n")
        stream.flush()
        start = time.monotonic()
        result = subprocess.run(
            list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT
        )
        stream.write(
            "\nExit code: %d; elapsed seconds: %.3f\n"
            % (result.returncode, time.monotonic() - start)
        )
    if result.returncode:
        raise RuntimeError("Command failed; see " + str(log))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "models/yolov8m.onnx")
    p.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "data/splits/person_temporal_v1/calibration",
    )
    p.add_argument(
        "--cache", type=Path, default=ROOT / "cache/int8-baseline-v1/yolov8m_int8.cache"
    )
    p.add_argument("--input", type=Path, default=ROOT / "data/reference/input.bin")
    p.add_argument(
        "--reference", type=Path, default=ROOT / "data/outputs/tensorrt_fp16_output.bin"
    )
    p.add_argument("--builder", type=Path, default=ROOT / "build/yolo_build")
    p.add_argument("--runner", type=Path, default=ROOT / "build/yolo_infer_raw")
    p.add_argument(
        "--seed-layers",
        type=Path,
        help="Omit to start with all eligible layers; supplied seed is always rebuilt and checked",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="New directory; never overwrite an experiment",
    )
    p.add_argument("--max-builds", type=int, default=12)
    add_gate_args(p)
    args = p.parse_args()
    if args.max_builds < 1:
        p.error("--max-builds must be positive")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.work_dir.exists():
        p.error("work-dir already exists; choose a new run")
    ref = load_output(str(args.reference))
    evaluate(ref, ref, **gate_kwargs(args))
    if args.input.stat().st_size != 4915200:
        raise ValueError("Invalid fixed input size")
    count = validate_cache(args.model, args.cache, args.calibration)

    args.work_dir.mkdir(parents=True)
    files = {
        k: {"path": str(getattr(args, k)), "sha256": sha(getattr(args, k))}
        for k in ("model", "cache", "input", "reference", "builder", "runner")
    }
    (args.work_dir / "experiment.json").write_text(
        json.dumps(
            {
                "files": files,
                "calibration_count": count,
                "thresholds": gate_kwargs(args),
                "max_builds": args.max_builds,
            },
            indent=2,
        )
        + "\n"
    )

    run_logged(
        [
            args.builder,
            args.model,
            args.work_dir / "unused.plan",
            "int8",
            "--inspect-only",
            "--dump-layers",
            args.work_dir / "network.tsv",
        ],
        args.work_dir / "inspect.log",
    )
    with (args.work_dir / "network.tsv").open() as f:
        eligible = [
            r["name"] for r in csv.DictReader(f, delimiter="\t") if r["eligible"] == "1"
        ]
    if len(eligible) != len(set(eligible)):
        raise ValueError("Ambiguous network names")
    seed = eligible
    if args.seed_layers:
        seed = [
            s.strip()
            for s in args.seed_layers.read_text().splitlines()
            if s.strip() and not s.lstrip().startswith("#")
        ]
        if len(seed) != len(set(seed)) or set(seed) - set(eligible):
            raise ValueError("Invalid seed layer list")

    # Each experiment owns a copy so candidate builds cannot alter the baseline.
    local_cache = args.work_dir / "calibration.cache"
    shutil.copyfile(args.cache, local_cache)
    shutil.copyfile(str(args.cache) + ".meta", str(local_cache) + ".meta")
    attempts = []

    def check(names):
        trial = args.work_dir / ("trial-%03d" % len(attempts))
        trial.mkdir()
        layers = trial / "fp16_layers.txt"
        layers.write_text("".join(x + "\n" for x in names))
        run_logged(
            [
                args.builder,
                args.model,
                trial / "yolov8m_int8.plan",
                "int8",
                args.calibration,
                local_cache,
                "--fp16-layers",
                layers,
                "--timing-cache",
                args.work_dir / "timing.cache",
            ],
            trial / "build.log",
        )
        run_logged(
            [
                args.runner,
                trial / "yolov8m_int8.plan",
                args.input,
                trial / "output.bin",
            ],
            trial / "infer.log",
        )
        result = evaluate(
            ref, load_output(str(trial / "output.bin")), **gate_kwargs(args)
        )
        (trial / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        attempts.append(
            {
                "trial": str(trial),
                "layers": len(names),
                "pass": result["pass"],
                "engine_sha256": sha(trial / "yolov8m_int8.plan"),
            }
        )
        (args.work_dir / "attempts.json").write_text(
            json.dumps(attempts, indent=2) + "\n"
        )
        # Persist the latest passing set immediately, including budget-limited runs.
        if result["pass"]:
            (args.work_dir / "fp16_layers.txt").write_text(layers.read_text())
            (args.work_dir / "best.json").write_text(
                json.dumps(attempts[-1], indent=2) + "\n"
            )
        print(
            trial.name,
            "PASS" if result["pass"] else "FAIL",
            "layers=",
            len(names),
            flush=True,
        )
        return result["pass"]

    result = prune(seed, check, args.max_builds)
    (args.work_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["best"] is not None else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(2)
