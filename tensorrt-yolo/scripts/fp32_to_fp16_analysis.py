import numpy as np

shape = (1, 84, 8400)

fp32 = np.fromfile(
    "data/outputs/tensorrt_fp32_output.bin",
    dtype="<f4"
).reshape(shape)

fp16 = np.fromfile(
    "data/outputs/tensorrt_fp16_output.bin",
    dtype="<f4"
).reshape(shape)

bbox32 = fp32[0, :4, :]
bbox16 = fp16[0, :4, :]

cls32 = fp32[0, 4:, :]
cls16 = fp16[0, 4:, :]

# 每个 candidate 的最佳类别与分数
class32 = np.argmax(cls32, axis=0)
class16 = np.argmax(cls16, axis=0)

score32 = np.max(cls32, axis=0)
score16 = np.max(cls16, axis=0)

bbox_diff = np.abs(bbox16 - bbox32)
cls_diff = np.abs(cls16 - cls32)

print("=== Raw FP16 vs FP32 ===")
print("BBox max :", bbox_diff.max())
print("BBox MAE :", bbox_diff.mean())
print("Cls max  :", cls_diff.max())
print("Cls MAE  :", cls_diff.mean())

# ------------------------------------------------
# 找 raw bbox 最大误差发生在哪里
# ------------------------------------------------
c, i = np.unravel_index(
    np.argmax(bbox_diff),
    bbox_diff.shape
)

names = ["x", "y", "w", "h"]

print("\n=== Largest bbox error ===")
print("candidate :", i)
print("component :", names[c])
print("FP32      :", bbox32[c, i])
print("FP16      :", bbox16[c, i])
print("abs diff  :", bbox_diff[c, i])

print("FP32 class:", class32[i])
print("FP32 score:", score32[i])
print("FP16 class:", class16[i])
print("FP16 score:", score16[i])

# ------------------------------------------------
# 高置信候选
# 使用 union，防止 FP16/FP32 跨过 0.7 阈值
# ------------------------------------------------
high = (score32 >= 0.7) | (score16 >= 0.7)

print("\n=== High-confidence candidates ===")
print("count:", high.sum())

if high.any():
    high_bbox_diff = bbox_diff[:, high]

    print("BBox max :", high_bbox_diff.max())
    print("BBox MAE :", high_bbox_diff.mean())
    print(
        "BBox p95 :",
        np.percentile(high_bbox_diff, 95)
    )
    print(
        "BBox p99 :",
        np.percentile(high_bbox_diff, 99)
    )

    print(
        "Class-ID mismatches:",
        np.sum(class32[high] != class16[high])
    )

# ------------------------------------------------
# 0.7 threshold crossing
# ------------------------------------------------
cross = (
    (score32 >= 0.7) !=
    (score16 >= 0.7)
)

print("\n=== Threshold stability ===")
print(
    "0.7 threshold crossings:",
    cross.sum()
)

# ------------------------------------------------
# 历史业务：
# 先最高分类，再要求 class == person(0)
# ------------------------------------------------
person32 = (
    (class32 == 0) &
    (score32 >= 0.7)
)

person16 = (
    (class16 == 0) &
    (score16 >= 0.7)
)

print("\n=== Historical person filter ===")
print(
    "FP32 person candidates:",
    np.where(person32)[0]
)

print(
    "FP16 person candidates:",
    np.where(person16)[0]
)

print(
    "candidate sets equal:",
    np.array_equal(
        np.where(person32)[0],
        np.where(person16)[0]
    )
)