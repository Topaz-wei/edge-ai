import numpy as np
import argparse


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

    bbox_max = bbox_diff.max()
    cls_max = cls_diff.max()

    # assert bbox_max < 0.05, (
    #     f"BBox mismatch too large: {bbox_max}"
    # )

    # assert cls_max < 1e-4, (
    #     f"Class score mismatch too large: {cls_max}"
    # )

    print("\nNumerical comparison completed.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    output_path = args.output

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
        output_path,
        dtype="<f4",
    ).reshape(shape)

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


if __name__ == "__main__":
    main()