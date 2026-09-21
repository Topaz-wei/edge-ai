import numpy as np


def convert(src, dst):
    x = np.load(src).astype(np.float32)
    x.tofile(dst)

    print(
        f"{src} -> {dst}, "
        f"shape={x.shape}, "
        f"elements={x.size}"
    )


convert(
    "data/input.npy",
    "data/input.bin",
)

convert(
    "data/pytorch_output.npy",
    "data/pytorch_output.bin",
)