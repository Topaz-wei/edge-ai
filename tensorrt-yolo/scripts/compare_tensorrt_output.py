import numpy as np


shape = (1, 84, 8400)

pytorch = np.load(
    "data/reference/pytorch_output.npy",
    allow_pickle=False,
)

onnx = np.load(
    "data/reference/onnx_output.npy",
    allow_pickle=False,
)

trt = np.fromfile(
    "data/outputs/tensorrt_fp32_output.bin",
    dtype="<f4",
).reshape(shape)


def compare(name, reference, actual):
    diff = actual - reference

    print(f"\n=== {name} ===")
    print("Max abs :", np.max(np.abs(diff)))
    print("MAE     :", np.mean(np.abs(diff)))
    print("RMSE    :", np.sqrt(np.mean(diff * diff)))

    # bbox
    bbox_diff = np.abs(
        actual[:, :4, :] -
        reference[:, :4, :]
    )

    # class probabilities
    cls_diff = np.abs(
        actual[:, 4:, :] -
        reference[:, 4:, :]
    )

    print(
        "BBox max:",
        bbox_diff.max(),
    )

    print(
        "Cls max :",
        cls_diff.max(),
    )


compare(
    "TensorRT vs PyTorch",
    pytorch,
    trt,
)

compare(
    "TensorRT vs ONNX Runtime",
    onnx,
    trt,
)