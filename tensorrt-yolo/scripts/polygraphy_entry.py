#!/usr/bin/env python3
"""Pinned Polygraphy CLI adapter: stop after completed iterations, not during TRT teardown."""
import gc
import json
import os
import sys


def main():
    import polygraphy

    if polygraphy.__version__ != "0.49.9":
        raise RuntimeError("This adapter requires Polygraphy 0.49.9")
    from polygraphy.tools.debug.subtool.precision import Precision
    from polygraphy.tools import main as cli

    budget = json.loads(open(os.environ["YOLO_PRECISION_SPEC"]).read())["max_builds"]
    # Stop at an iteration boundary so checked artifacts are already saved.
    original = Precision.step
    state = {"iterations": 0, "exhausted": False}

    def bounded_step(self, success):
        state["iterations"] += 1
        done = original(self, success)
        if not done and state["iterations"] >= budget:
            state["exhausted"] = True
            print("Build budget reached; keeping verified artifacts.", flush=True)
            return True
        return done

    Precision.step = bounded_step
    try:
        code = cli()
    finally:
        # Restore the adapter and release owners before TensorRT teardown.
        Precision.step = original
        module = sys.modules.get("polygraphy_config")
        if module is not None:
            module._KEEP.clear()
        gc.collect()
    return 3 if state["exhausted"] and code == 0 else code


if __name__ == "__main__":
    sys.exit(main())
