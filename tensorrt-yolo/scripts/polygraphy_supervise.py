#!/usr/bin/env python3
"""Run a prepared experiment, aborting on infrastructure errors instead of pruning layers."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def supervise(work):
    work = Path(work).resolve()
    spec = json.loads((work / "command.json").read_text())
    env = os.environ.copy()
    env.update(spec["env"])
    # run.sh may be started much later than preparation: recheck inputs now.
    if "YOLO_PRECISION_SPEC" in spec["env"]:
        from precision_search import sha, validate_cache

        experiment = json.loads(Path(spec["env"]["YOLO_PRECISION_SPEC"]).read_text())
        for key, digest in experiment["hashes"].items():
            if sha(experiment[key]) != digest:
                raise ValueError("Prepared experiment input changed: " + key)
        validate_cache(
            experiment["model"], experiment["cache"], experiment["calibration"]
        )
    # Prevent stale pass/fail artifacts and accidental overwrites on reruns.
    with (work / "search.log").open("x") as log:
        process = subprocess.Popen(
            spec["command"],
            cwd=work,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                # Infrastructure failures must not become accuracy FAIL samples.
                if "ERROR:" in line or "Failed to create network or engine" in line:
                    raise RuntimeError("Infrastructure failure; see search.log")
            return process.wait()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == "__main__":
    try:
        sys.exit(supervise(sys.argv[1]))
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(2)
