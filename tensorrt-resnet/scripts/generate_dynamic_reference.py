import os

import numpy as np
import torch

from export_resnet18 import ResNet18


def main():
    os.makedirs("data/dynamic", exist_ok=True)

    # 必须和导出 ONNX 时保持一致
    torch.manual_seed(0)

    model = ResNet18().eval().cuda()

    for batch in [1, 4, 8]:
        # 单独固定输入随机种子，保证实验可重复
        torch.manual_seed(1000 + batch)

        x = torch.randn(
            batch,
            3,
            224,
            224,
            dtype=torch.float32,
        )

        with torch.no_grad():
            y = model(x.cuda()).cpu()

        assert torch.isfinite(x).all()
        assert torch.isfinite(y).all()

        x_np = x.numpy().astype(np.float32)
        y_np = y.numpy().astype(np.float32)

        x_np.tofile(
            f"data/dynamic/input_b{batch}.bin"
        )

        y_np.tofile(
            f"data/dynamic/reference_b{batch}.bin"
        )

        np.save(
            f"data/dynamic/input_b{batch}.npy",
            x_np,
        )

        np.save(
            f"data/dynamic/reference_b{batch}.npy",
            y_np,
        )

        print(
            f"batch={batch}: "
            f"input={x_np.shape}, "
            f"output={y_np.shape}"
        )


if __name__ == "__main__":
    main()