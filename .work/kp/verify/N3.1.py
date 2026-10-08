"""N3.1 特征值的估计。运行：python3 .work/kp/verify/N3.1.py"""
import numpy as np

A = np.array(
    [[1, 0.1, 0.2, 0.3], [0.5, 3, 0.1, 0.2], [1, 0.3, -1, 0.5], [0.2, -0.3, -0.1, -4]],
    dtype=float,
)
radii = [0.6, 0.8, 1.8, 0.6]
centers = np.diag(A)
assert np.allclose([np.sum(np.abs(A[i])) - abs(A[i, i]) for i in range(4)], radii)
ev = np.linalg.eigvals(A)
for lam in ev:
    assert any(abs(lam - c) <= r + 1e-8 for c, r in zip(centers, radii))
# 孤立圆 G2、G4 各含一个实特征值
assert any(abs(lam - 3) <= 0.8 and abs(lam.imag) < 1e-8 for lam in ev)
assert any(abs(lam + 4) <= 0.6 and abs(lam.imag) < 1e-8 for lam in ev)

A2 = np.array([[0.9, 0.01, 0.12], [0.01, 0.8, 0.13], [0.01, 0.02, 0.4]])
P = np.diag([1.0, 1.0, 0.1])
B = np.linalg.inv(P) @ A2 @ P
expect = np.array([[0.9, 0.01, 0.012], [0.01, 0.8, 0.013], [0.1, 0.2, 0.4]])
assert np.allclose(B, expect)
assert abs(0.12 * 0.1 - 0.012) < 1e-12
assert abs(0.01 / 0.1 - 0.1) < 1e-12
ev2 = np.linalg.eigvals(A2)
# 书上的收紧界
bounds = [(0.9, 0.022), (0.8, 0.023), (0.4, 0.03)]
for lam in ev2:
    assert any(abs(lam - c) <= r + 1e-8 for c, r in bounds)
assert any(abs(lam - 0.9) <= 0.022 + 1e-8 for lam in ev2)

print("ALL OK")
