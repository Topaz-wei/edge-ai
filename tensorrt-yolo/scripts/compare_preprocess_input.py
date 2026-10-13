import numpy as np

ref = np.fromfile(
    "data/reference/input.bin",
    dtype=np.float32
)

cpp = np.fromfile(
    "data/inputs/cpp_input.bin",
    dtype=np.float32
)

print("ref elements:", ref.size)
print("cpp elements:", cpp.size)

assert ref.size == cpp.size, (
    f"size mismatch: ref={ref.size}, cpp={cpp.size}"
)

diff = cpp - ref
abs_diff = np.abs(diff)

print("finite ref :", np.isfinite(ref).all())
print("finite cpp :", np.isfinite(cpp).all())

print("max abs    :", abs_diff.max())
print("MAE        :", abs_diff.mean())
print("RMSE       :", np.sqrt(np.mean(diff * diff)))

print("exact equal:", np.array_equal(ref, cpp))
print(
    "allclose   :",
    np.allclose(
        ref,
        cpp,
        rtol=0.0,
        atol=1e-7,
    )
)

# 找最大误差位置
idx = np.argmax(abs_diff)

print("\nmax diff index:", idx)
print("ref value     :", ref[idx])
print("cpp value     :", cpp[idx])
print("abs diff      :", abs_diff[idx])

# reshape 成模型输入
ref = ref.reshape(1, 3, 640, 640)
cpp = cpp.reshape(1, 3, 640, 640)

# 分 channel 看误差
for c, name in enumerate(["R", "G", "B"]):
    d = np.abs(
        cpp[0, c] - ref[0, c]
    )

    print(
        f"{name}: "
        f"max={d.max():.10f}, "
        f"mae={d.mean():.10f}"
    )