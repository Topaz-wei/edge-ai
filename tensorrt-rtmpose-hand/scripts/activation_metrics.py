# 以 FP64 比较同位置激活；非有限和零范数显式标记，不隐式过滤异常元素。
import numpy as np


def metrics(
    reference,
    actual
):
    if reference.shape != actual.shape or reference.size == 0:
        raise ValueError('Activation shape mismatch or empty tensor')
    x = reference.astype(np.float64).ravel()
    y = actual.astype(np.float64).ravel()
    result = {'elements': x.size}
    for name, values in [('reference', x), ('actual', y)]:
        count = int(np.count_nonzero(~np.isfinite(values)))
        result[name + '_nonfinite'] = count
        result[name + '_zero_fraction'] = float(np.mean(values == 0))
        result[name + '_min'] = None if count else float(values.min())
        result[name + '_max'] = None if count else float(values.max())
    keys = ['mae', 'rmse', 'max_abs', 'relative_l2', 'cosine', 'norm_ratio', 'reference_norm', 'actual_norm']
    if result['reference_nonfinite'] or result['actual_nonfinite']:
        result.update({key: None for key in keys})
        result['zero_norm_case'] = 'nonfinite'
        return result
    xn = float(np.linalg.norm(x))
    yn = float(np.linalg.norm(y))
    delta = y - x
    result.update({
        'mae': float(np.mean(np.abs(delta))),
        'rmse': float(np.sqrt(np.mean(delta * delta))),
        'max_abs': float(np.max(np.abs(delta))),
        'reference_norm': xn,
        'actual_norm': yn,
        'relative_l2': float(np.linalg.norm(delta) / xn) if xn else None,
        'norm_ratio': yn / xn if xn else None,
        'cosine': float(np.clip(np.dot(x, y) / (xn * yn), -1., 1.)) if xn and yn else None,
        'zero_norm_case': 'both' if not xn and not yn else ('reference' if not xn else ('actual' if not yn else 'neither')),
    })
    return result


def safe_final_comparison(
    actual,
    expected,
    samples
):
    from verify_trt_fp16 import compare_raw
    bad_actual = sum(int(np.count_nonzero(~np.isfinite(value))) for value in actual)
    bad_expected = sum(int(np.count_nonzero(~np.isfinite(value))) for value in expected)
    if bad_actual or bad_expected:
        return {'status': 'nonfinite', 'actual_nonfinite': bad_actual,
                'reference_nonfinite': bad_expected, 'decoded_metrics': None}
    return compare_raw(actual, expected, samples)
