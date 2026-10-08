"""N3.2 幂法与反幂法。运行：python3 .work/kp/verify/N3.2.py"""
import numpy as np

def max_comp(x):
    i = int(np.argmax(np.abs(x)))
    return x[i]

# 例 1 第一步精确
A1 = np.array([[2.0, 4, 6], [3, 9, 15], [4, 16, 36]])
x1 = A1 @ np.ones(3)
assert np.allclose(x1, [12, 27, 56])
assert abs(max(np.abs(np.linalg.eigvals(A1))) - 43.88) < 0.01
# 四位小数规范化后再乘，应对上表 3.2-1 的第二行
y1 = np.array([0.2143, 0.4821, 1.0])
x2 = A1 @ y1
assert abs(x2[0] - 8.357) < 5e-4
assert abs(x2[2] - 44.57) < 5e-3

# 例 2
A2 = np.array([[-4.0, 14, 0], [-5, 13, 0], [-1, 0, 2]])
ev2 = np.sort(np.linalg.eigvals(A2).real)
assert np.allclose(ev2, [2, 3, 6])
def aitken(m0, m1, m2):
    return m2 - (m2 - m1) ** 2 / (m2 - 2 * m1 + m0)
assert abs(aitken(1, 10, 7.2) - 7.8644) < 5e-4
assert abs(aitken(10, 7.2, 6.5) - 6.2667) < 5e-4
assert abs(aitken(7.2, 6.5, 6.2312) - 6.0636) < 5e-4

# 例 3
A3 = np.array([[-3.0, 1, 0], [1, -3, -3], [0, -3, 4]])
ev3 = np.linalg.eigvals(A3).real
assert any(abs(v - 5.1247) < 5e-4 for v in ev3)
# 平移 a=-4，六次后 max≈9.1247
x = np.array([0.0, 0, 1])
As = A3 + 4 * np.eye(3)
for _ in range(6):
    y = x / max_comp(x)
    x = As @ y
assert abs(max_comp(x) - 9.1247) < 5e-4
assert abs(max_comp(x) - 4 - 5.1247) < 5e-4

# 例 4
A4 = np.array([[6.0, 2, 1], [2, 3, 1], [1, 1, 1]])
y = np.ones(3)
x = A4 @ y
R = (y @ x) / (y @ y)
assert np.allclose(x, [9, 6, 3])
assert abs(R - 6) < 1e-12
lam = max(np.abs(np.linalg.eigvals(A4)))
assert abs(lam - 7.288) < 2e-3

# 例 5 的收尾算术
assert abs(-6.42 - 1 / 937.875765 - (-6.421066)) < 5e-7
L = np.array([[1, 0, 0], [0.369004, 1, 0], [0.184502, 0.375148, 1.0]])
R = np.array([[5.42, 2, 1], [0, 1.681993, 0.630996], [0, 0, -1.218848e-3]])
S = np.array([[-1.0, 2, 1], [2, -4, 1], [1, 1, -6]]) + 6.42 * np.eye(3)
assert np.linalg.norm(L @ R - S) < 5e-4

# 对称正定时平移点取中点
# (λ2+λn)/2 使两端到 λ1 的距离比较
assert abs((3 + 2) / 2 - 2.5) < 1e-12

print("ALL OK")
