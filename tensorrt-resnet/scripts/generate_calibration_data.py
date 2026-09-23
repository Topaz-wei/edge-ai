import numpy as np


NUM_SAMPLES = 100
SHAPE = (NUM_SAMPLES, 3, 224, 224)

rng = np.random.default_rng(2026)

data = rng.standard_normal(
    SHAPE,
    dtype=np.float32,
)

assert np.isfinite(data).all()

path = "calibration/calibration_100.bin"

data.tofile(path)

print("Saved:", path)
print("Shape:", data.shape)
print("Elements:", data.size)
print("Bytes:", data.nbytes)
print("Min:", data.min())
print("Max:", data.max())