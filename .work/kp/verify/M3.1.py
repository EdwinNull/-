"""M3.1 向量范数 —— 讲解中数值的复算脚本。"""
import numpy as np

# 常用范数与单位球例子
x = np.array([0.5, 0.5])
assert abs(np.linalg.norm(x, 1) - 1.0) < 1e-12
assert abs(np.linalg.norm(x, 2) - 1 / np.sqrt(2)) < 1e-12
assert abs(np.linalg.norm(x, np.inf) - 0.5) < 1e-12
assert 0.5 <= 1 / np.sqrt(2) <= 1

# 三维 1-范数与无穷范数的等价界
x3 = np.array([1.0, -2.0, 3.0])
assert np.linalg.norm(x3, np.inf) <= np.linalg.norm(x3, 1) <= 3 * np.linalg.norm(x3, np.inf)

# 加权范数 A=[[2,1],[1,2]], x=[1,2]^T
A = np.array([[2.0, 1.0], [1.0, 2.0]])
xw = np.array([1.0, 2.0])
assert np.allclose(np.linalg.eigvalsh(A), [1.0, 3.0])
assert np.allclose(A @ xw, [4.0, 5.0])
q = xw @ A @ xw
assert abs(q - 14.0) < 1e-12
assert abs(np.sqrt(q) - np.sqrt(14.0)) < 1e-12

# 0<p<1 的反例，p=1/2
p = 0.5
u = np.array([1.0, 0.0])
v = np.array([0.0, 1.0])
pnorm = lambda z: np.sum(np.abs(z) ** p) ** (1 / p)
assert abs(pnorm(u) - 1.0) < 1e-12
assert abs(pnorm(v) - 1.0) < 1e-12
assert abs(pnorm(u + v) - 4.0) < 1e-12
assert pnorm(u + v) > pnorm(u) + pnorm(v)

# Frobenius 范数和迹公式
B = np.array([[1.0 + 0j, 1j], [2.0 + 0j, -1j]])
assert abs(np.linalg.norm(B, "fro") - np.sqrt(7)) < 1e-12
assert abs(np.trace(B.conj().T @ B).real - 7.0) < 1e-12

# 自测题
xs = np.array([1.0, -2.0, 2.0])
assert np.linalg.norm(xs, 1) == 5
assert np.linalg.norm(xs, 2) == 3
assert np.linalg.norm(xs, np.inf) == 2
C = np.diag([1.0, 3.0])
z = np.array([2.0, -1.0])
assert abs(z @ C @ z - 7.0) < 1e-12
assert abs(np.sqrt(z @ C @ z) - np.sqrt(7.0)) < 1e-12

print("ALL OK")
