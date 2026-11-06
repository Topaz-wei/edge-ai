# 调用实际 C++ audit，验证校准身份中的 shape 和精度设置可被标准 JSON 正确读取。
import json
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    output = Path(directory) / 'audit.json'
    subprocess.run(
        [str(root.parent / 'build-shared/tensorrt-rtmpose-hand/rtmpose_int8'),
         'audit', str(root), str(output)],
        check=True
    )
    identity = json.loads(output.read_text())
    config = json.loads((root / 'configs/rtmpose_hand_builder_int8.json').read_text())
    assert identity['calibration_shape'] == [1, 3, 256, 256]
    assert identity['fp16'] == int(config['fp16'])
    assert identity['inference_profile'] == '1/2/4'
    assert len(identity['ordered_tensors']) == 352
    assert identity['schema_version'] == 2
print('C++ audit identity matches active calibration configuration.')
