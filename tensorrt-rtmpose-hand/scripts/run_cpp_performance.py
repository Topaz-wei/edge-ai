# 编排 GPU 性能测量并记录输入身份与环境；不修改用户设置的电源或时钟。
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'tensorrt-rtmpose-hand'
OUTPUT = MODULE / 'validation/cpp-2g-performance'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def environment():
    result = {'time_unix': time.time(), 'sysfs': {}}
    paths = list(Path('/sys/class/devfreq').glob('*/cur_freq'))
    paths += list(Path('/sys/class/devfreq').glob('*/min_freq'))
    paths += list(Path('/sys/class/devfreq').glob('*/max_freq'))
    paths += list(Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_cur_freq'))
    paths += list(Path('/sys/class/thermal').glob('thermal_zone*/temp'))
    paths += list(Path('/sys/class/thermal').glob('thermal_zone*/type'))
    for path in paths:
        try:
            result['sysfs'][str(path)] = path.read_text().strip()
        except OSError as error:
            result['sysfs'][str(path)] = str(error)
    for command in [['nvpmodel', '-q'], ['uname', '-a']]:
        response = subprocess.run(command, capture_output=True, text=True)
        result[' '.join(command)] = {'returncode': response.returncode, 'stdout': response.stdout, 'stderr': response.stderr}
    return result


def main():
    timings = OUTPUT / 'timings.csv'
    if timings.exists() or (OUTPUT / 'benchmark-inputs.json').exists():
        raise RuntimeError('Experiment already exists; do not overwrite raw measurements')
    consistency = json.loads((OUTPUT / 'consistency.json').read_text())
    assert consistency['status'] == 'passed'
    paths = [
        MODULE / 'engines/rtmpose_hand_fp32_cpp_2g.engine',
        MODULE / 'engines/rtmpose_hand_fp16_cpp_2g.engine',
        ROOT / 'build-shared/tensorrt-rtmpose-hand/rtmpose_benchmark',
        ROOT / 'build-shared/tensorrt-rtmpose-hand/rtmpose_build',
        MODULE / 'apps/benchmark.cpp', MODULE / 'apps/build_engine.cpp',
        ROOT / 'trt-runtime/src/builder.cpp', ROOT / 'trt-runtime/src/runner.cpp',
        MODULE / 'src/reference.cpp', MODULE / 'configs/rtmpose_hand_reference.json',
        MODULE / 'configs/rtmpose_hand_builder.json', MODULE / 'configs/rtmpose_hand_builder_fp32_2g.json',
        MODULE / 'validation/trt-fp32/cases.json', OUTPUT / 'consistency.json',
        OUTPUT / 'build-fp32.log', OUTPUT / 'build-inputs.json', Path(__file__).resolve(),
    ]
    cases = json.loads((MODULE / 'validation/trt-fp32/cases.json').read_text())
    for sample in cases['samples']:
        paths += [Path(sample['tensor']), Path(sample['image'])]
    assert sha(paths[0]) == consistency['fp32_engine_sha256']
    assert sha(paths[1]) == consistency['fp16_engine_sha256']
    patch = subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT)
    (OUTPUT / 'tracked-changes.patch').write_bytes(patch)
    inputs = {
        'files_sha256': {str(path.relative_to(ROOT)): sha(path) for path in paths},
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'tracked_patch_sha256': sha(OUTPUT / 'tracked-changes.patch'),
        'software': {'jetson_release': Path('/etc/nv_tegra_release').read_text(),
                     'tensorrt': json.loads((MODULE / 'validation/trt-fp16-cpp-2g/layers-context.json').read_text())['tensorrt_version']},
        'clock_policy': 'MAXN and jetson_clocks enabled manually by user; tool does not modify them',
        'user_reported_frequencies': {'cpu_khz': 2188800, 'gpu_hz': 930750000, 'emc_hz': 3199000000},
    }
    write(OUTPUT / 'benchmark-inputs.json', inputs)
    write(OUTPUT / 'environment-before.json', environment())
    command = [str(ROOT / 'build-shared/tensorrt-rtmpose-hand/rtmpose_benchmark'),
               str(paths[0]), str(paths[1]), str(MODULE / 'validation/trt-fp32/cases.json'), str(timings)]
    with (OUTPUT / 'tegrastats.log').open('w') as telemetry, (OUTPUT / 'benchmark.log').open('w') as log:
        monitor = subprocess.Popen(['/usr/bin/tegrastats', '--interval', '1000'], stdout=telemetry, stderr=subprocess.STDOUT)
        try:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
            write(OUTPUT / 'environment-after.json', environment())
    for relative, expected in inputs['files_sha256'].items():
        assert sha(ROOT / relative) == expected, relative
    write(OUTPUT / 'benchmark-complete.json', {
        'timings_sha256': sha(timings), 'inputs_sha256': sha(OUTPUT / 'benchmark-inputs.json'),
        'environment_before_sha256': sha(OUTPUT / 'environment-before.json'),
        'environment_after_sha256': sha(OUTPUT / 'environment-after.json'),
        'telemetry_sha256': sha(OUTPUT / 'tegrastats.log'), 'log_sha256': sha(OUTPUT / 'benchmark.log'),
    })
    print('Benchmark completed; raw samples and telemetry preserved.')


if __name__ == '__main__':
    main()
