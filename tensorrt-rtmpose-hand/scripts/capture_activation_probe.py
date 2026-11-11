# 将一次 C++ 激活采集与 engine、配置、可执行文件和原始输出哈希绑定，拒绝覆盖和身份变化。
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('RTMPOSE_PROBE_DIR', str(ROOT / 'validation/fp16-activations'))).resolve()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_capture(folder):
    record = json.loads((folder / 'capture.json').read_text())
    for name, sha in record['assets_sha256'].items():
        if digest(Path(name)) != sha:
            raise ValueError('Capture asset changed: ' + name)
    actual = {path.name for path in folder.glob('*.bin')}
    if actual != set(record['raw_sha256']):
        raise ValueError('Capture output set changed')
    for name, sha in record['raw_sha256'].items():
        if digest(folder / name) != sha:
            raise ValueError('Capture output changed: ' + name)
    return record


def capture(mode):
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for name, sha in manifest['sha256'].items():
        if digest(Path(name)) != sha:
            raise ValueError('Frozen probe input changed: ' + name)
    executable = ROOT.parent / 'build-shared/tensorrt-rtmpose-hand/rtmpose_probe'
    config = OUT / 'config.json'
    engine = OUT / (mode + '.engine')
    folder = OUT / (mode + '-verified')
    assets = {str(path): digest(path) for path in [executable, config, engine, OUT / 'manifest.json']}
    subprocess.run(
        [str(executable), 'run', str(config), str(engine), str(folder)],
        check=True
    )
    for name, sha in assets.items():
        if digest(Path(name)) != sha:
            raise ValueError('Asset changed during capture: ' + name)
    files = list(folder.glob('*.bin'))
    settings = json.loads(config.read_text())
    if len(files) != len(settings['samples']) * len(settings['outputs']):
        raise ValueError('Incomplete capture')
    if (folder / 'complete.txt').read_text().strip() != str(len(settings['samples'])):
        raise ValueError('Incomplete sample count')
    record = {'mode': mode, 'assets_sha256': assets,
              'raw_sha256': {path.name: digest(path) for path in files},
              'capture_script_sha256': digest(Path(__file__).resolve())}
    (folder / 'capture.json').write_text(json.dumps(record, indent=2) + '\n')
    verify_capture(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['fp32', 'fp16'])
    capture(parser.parse_args().mode)
