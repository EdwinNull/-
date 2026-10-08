"""N3.3 QR 方法。运行：python3 .work/kp/verify/N3.3.py"""
import numpy as np

A = np.array([[5.0, -2, -5, -1], [1, 0, -3, 2], [0, 2, 2, -3], [0, 0, 1, -2]])
ev = np.linalg.eigvals(A)
# -1, 4, 1±2i
assert any(abs(v + 1) < 1e-8 for v in ev)
assert any(abs(v - 4) < 1e-8 for v in ev)
assert any(abs(v - (1 + 2j)) < 1e-8 for v in ev)
assert any(abs(v - (1 - 2j)) < 1e-8 for v in ev)

B = np.array([[1.8789, -3.5910], [1.3290, 0.1211]])
bev = np.linalg.eigvals(B)
assert any(abs(v - (1 + 2j)) < 1e-4 for v in bev)
assert any(abs(v - (1 - 2j)) < 1e-4 for v in bev)

# 书上行列式方程与 A12 的 2×2 块一致
# det [[λ-1.8789, 3.5910], [-1.3290, λ-0.1211]] = 0 的根是 1±2i
disc = 4 - 4 * (1.8789 * 0.1211 + 3.5910 * 1.3290)
assert abs(disc + 16) < 1e-2

# 一步 QR 保持相似：特征值不变
Q, R = np.linalg.qr(A)
A2 = R @ Q
assert np.allclose(np.sort_complex(np.linalg.eigvals(A2)), np.sort_complex(ev))

# 上 Hessenberg 与上三角相乘仍是上 Hessenberg
H = np.array([[1.0, 2, 3], [4, 5, 6], [0, 7, 8]])
U = np.triu(np.array([[1.0, 1, 1], [2, 3, 4], [5, 6, 7]]))
HU = H @ U
assert abs(HU[2, 0]) < 1e-12

print("ALL OK")
