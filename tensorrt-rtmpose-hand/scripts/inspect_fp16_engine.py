# 本脚本用实际 batch=2 的 context 导出 engine inspector；不执行推理或修改 engine。
# 使用系统 Python 的 TensorRT binding，和 ORT 所在 py38 环境分开运行。
import argparse
import hashlib
import json
from pathlib import Path

import tensorrt as trt

ROOT = Path(
    __file__
).resolve().parents[1]
engine_path = ROOT / 'engines/rtmpose_hand_fp16.engine'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        type=Path,
        default=engine_path
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation/trt-fp16/layers-context.json"
    )
    arguments = parser.parse_args()
    runtime = engine = context = inspector = None
    try:
        blob = arguments.engine.read_bytes()
        logger = trt.Logger(
            trt.Logger.WARNING
        )
        runtime = trt.Runtime(
            logger
        )
        engine = runtime.deserialize_cuda_engine(
            blob
        )
        assert engine is not None
        context = engine.create_execution_context()
        assert context is not None
        assert context.set_input_shape(
            'input',
            (2, 3, 256, 256)
        )
        assert context.infer_shapes() == []
        tensors = []
        for index in range(
            engine.num_io_tensors
        ):
            name = engine.get_tensor_name(
                index
            )
            shape = list(
                context.get_tensor_shape(
                    name
                )
            )
            assert all(
                (dimension > 0 for dimension in shape)
            )
            assert engine.get_tensor_dtype(
                name
            ) == trt.float32
            tensors.append(
                {'name': name, 'shape': shape, 'dtype': str(
                    engine.get_tensor_dtype(
                        name
                    )
                ),
                            'mode': str(
                    engine.get_tensor_mode(
                        name
                    )
                )}
            )
        inspector = engine.create_engine_inspector()
        inspector.execution_context = context
        assert inspector.execution_context is not None
        information = json.loads(
            inspector.get_engine_information(
                trt.LayerInformationFormat.JSON
            )
        )
        formats = [item.get('Format/Datatype', '') for layer in information['Layers']
                   for item in layer.get('Inputs', []) + layer.get('Outputs', [])]
        assert any('FP16' in value or 'FP32' in value for value in formats), 'Inspector formats unresolved'
        report = {'engine_sha256': hashlib.sha256(
            blob
        ).hexdigest(), 'tensorrt_version': trt.__version__,
                  'input_shape': [2, 3, 256, 256], 'tensors': tensors, 'inspector': information}
        output = arguments.output
        output.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2
            ) + '\n'
        )
        print(
            'Exported context-aware inspector for %d layers.' % len(
                information['Layers']
            )
        )
    finally:
        inspector = None
        context = None
        engine = None
        runtime = None


if __name__ == '__main__':
    main()
