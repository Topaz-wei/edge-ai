# 本脚本在构建前锁定公共 builder、模型入口、配置和可执行程序身份。
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'tensorrt-rtmpose-hand'
FILES = [
    'trt-runtime/include/edge/trt/builder.hpp',
    'trt-runtime/src/builder.cpp',
    'trt-runtime/CMakeLists.txt',
    'tensorrt-rtmpose-hand/apps/build_engine.cpp',
    'tensorrt-rtmpose-hand/CMakeLists.txt',
    'tensorrt-rtmpose-hand/configs/rtmpose_hand_builder.json',
    'tensorrt-rtmpose-hand/configs/rtmpose_hand_trt_fp16_cpp_2g.json',
    'tensorrt-rtmpose-hand/models/rtmpose_hand.onnx',
    'tensorrt-rtmpose-hand/configs/rtmpose_hand_reference.json',
    'build-shared/tensorrt-rtmpose-hand/rtmpose_build',
]


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete():
    directory = MODULE / 'validation/trt-fp16-cpp-2g'
    provenance_path = directory / 'build-provenance.json'
    provenance = json.loads(provenance_path.read_text())
    for name, expected in provenance['files_sha256'].items():
        assert checksum(ROOT / name) == expected, 'Build input changed: ' + name
    engine = MODULE / 'engines/rtmpose_hand_fp16_cpp_2g.engine'
    log = directory / 'build.log'
    contents = log.read_text()
    config_sha = provenance['files_sha256'][
        'tensorrt-rtmpose-hand/configs/rtmpose_hand_builder.json'
    ]
    assert 'precision=fp16; TF32=off; workspace_mib=2048; config_sha256=' + config_sha in contents
    assert 'Engine saved: ' in contents
    assert '; sha256=' + checksum(engine) in contents
    result = {
        'engine_sha256': checksum(engine),
        'provenance_sha256': checksum(provenance_path),
        'log_sha256': checksum(log),
    }
    with (directory / 'build-complete.json').open('x') as stream:
        json.dump(
            result,
            stream,
            indent=2
        )
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--complete',
        action='store_true'
    )
    arguments = parser.parse_args()
    if arguments.complete:
        complete()
        return
    output = MODULE / 'validation/trt-fp16-cpp-2g/build-provenance.json'
    data = {
        'builder': 'edge_trt_builder',
        'workspace_mib': 2048,
        'precision': 'fp16',
        'files_sha256': {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in FILES
        },
    }
    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    with output.open('x') as stream:
        json.dump(
            data,
            stream,
            indent=2
        )
        stream.write('\n')


if __name__ == '__main__':
    main()
