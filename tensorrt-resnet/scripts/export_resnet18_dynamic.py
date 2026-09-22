import onnx
import torch

from export_resnet18 import ResNet18


def main():
    torch.manual_seed(0)

    model = ResNet18().eval()

    x = torch.randn(1, 3, 224, 224)

    output_path = "models/resnet18_dynamic.onnx"

    torch.onnx.export(
        model,
        x,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={
            "input": {
                0: "batch",
            },
            "output": {
                0: "batch",
            },
        },
    )

    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)

    print("Exported:", output_path)

    for tensor in model_onnx.graph.input:
        print(
            "Input:",
            tensor.name,
            [
                d.dim_value if d.dim_value != 0 else d.dim_param
                for d in tensor.type.tensor_type.shape.dim
            ],
        )


if __name__ == "__main__":
    main()