"""M3.1 向量范数：复算讲解中的具体向量、矩阵及数值结果。"""
import math

import numpy as np


# 三种单位球的几何例：菱形面积 2、单位圆面积 pi、边长 2 的正方形面积 4。
vertices_l1 = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
assert np.allclose(np.sum(np.abs(vertices_l1), axis=1), 1.0)
assert np.allclose(np.linalg.norm(vertices_l1, axis=1), 1.0)
twice_area_l1 = abs(np.dot(vertices_l1[:, 0], np.roll(vertices_l1[:, 1], 1)) - np.dot(vertices_l1[:, 1], np.roll(vertices_l1[:, 0], 1)))
assert math.isclose(twice_area_l1 / 2.0, 2.0)
assert math.isclose(math.pi * 1.0**2, math.pi)
vertices_inf = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=float)
assert np.allclose(np.max(np.abs(vertices_inf), axis=1), 1.0)
assert np.allclose(np.linalg.norm(vertices_inf, axis=1), math.sqrt(2.0))
twice_area_inf = abs(np.dot(vertices_inf[:, 0], np.roll(vertices_inf[:, 1], 1)) - np.dot(vertices_inf[:, 1], np.roll(vertices_inf[:, 0], 1)))
assert math.isclose(twice_area_inf / 2.0, 4.0)

# M3.1.K1 的复向量例 g(z)=|z1|+2|z2|，z=(1-i,2)^T。
z = np.array([1.0 - 1.0j, 2.0 + 0.0j])
g_z = abs(z[0]) + 2.0 * abs(z[1])
assert math.isclose(g_z, math.sqrt(2.0) + 4.0, rel_tol=1e-12)

# 常用范数次序、二维例 x=(1/3,2/3)^T，以及 R^3 等价界。
x = np.array([1.0 / 3.0, 2.0 / 3.0])
assert math.isclose(np.linalg.norm(x, 1), 1.0, rel_tol=1e-12)
assert math.isclose(np.linalg.norm(x, 2), math.sqrt(5.0) / 3.0, rel_tol=1e-12)
assert math.isclose(np.linalg.norm(x, np.inf), 2.0 / 3.0, rel_tol=1e-12)
assert np.linalg.norm(x, np.inf) <= np.linalg.norm(x, 2) <= np.linalg.norm(x, 1)
x3 = np.array([1.0, -2.0, 3.0])
assert np.linalg.norm(x3, np.inf) <= np.linalg.norm(x3, 1) <= 3.0 * np.linalg.norm(x3, np.inf)
assert np.linalg.norm(x3, np.inf) <= np.linalg.norm(x3, 2) <= math.sqrt(3.0) * np.linalg.norm(x3, np.inf)
assert np.linalg.norm(x3, 1) / math.sqrt(3.0) <= np.linalg.norm(x3, 2) <= np.linalg.norm(x3, 1)
equal_components = np.array([1.0, 1.0, 1.0])
assert math.isclose(np.linalg.norm(equal_components, 1), 3.0)
assert math.isclose(np.linalg.norm(equal_components, 2), math.sqrt(3.0))
assert math.isclose(np.linalg.norm(equal_components, np.inf), 1.0)

# 等价范数例：x=(1,1)^T，以及三个分量相等时 K1=1、K2=3 的上界取等。
x_equal_2 = np.array([1.0, 1.0])
assert math.isclose(np.linalg.norm(x_equal_2, 1), 2.0)
assert math.isclose(np.linalg.norm(x_equal_2, 2), math.sqrt(2.0))
assert np.linalg.norm(x_equal_2, np.inf) <= np.linalg.norm(x_equal_2, 1) <= 2.0 * np.linalg.norm(x_equal_2, np.inf)
assert np.linalg.norm(equal_components, 1) == 3.0 * np.linalg.norm(equal_components, np.inf)
single_nonzero = np.array([1.0, 0.0, 0.0])
assert math.isclose(np.linalg.norm(single_nonzero, 1) / np.linalg.norm(single_nonzero, np.inf), 1.0)
assert math.isclose(np.linalg.norm(equal_components, 1) / np.linalg.norm(equal_components, np.inf), 3.0)

# 等价范数说明中的序列 x_k=(1/k,1/k,1/k)^T，在 k=1,2,3 时的范数比例。
for k in (1.0, 2.0, 3.0):
    xk = np.array([1.0 / k, 1.0 / k, 1.0 / k])
    assert math.isclose(np.linalg.norm(xk, 1), 3.0 / k, rel_tol=1e-12)
    assert math.isclose(np.linalg.norm(xk, 2), math.sqrt(3.0) / k, rel_tol=1e-12)
    assert math.isclose(np.linalg.norm(xk, np.inf), 1.0 / k, rel_tol=1e-12)

# 加权范数知识点中的 A=diag(2,1)、x=(1,1)^T。
Aw = np.diag([2.0, 1.0])
xw = np.array([1.0, 1.0])
assert np.allclose(np.linalg.eigvalsh(Aw), [1.0, 2.0])
assert np.allclose(Aw @ xw, [2.0, 1.0])
assert math.isclose(float(xw @ Aw @ xw), 3.0)
assert math.isclose(np.sqrt(float(xw @ Aw @ xw)), math.sqrt(3.0))

# 0<p<1 反例，p=1/2；幂平均在两个坐标单位向量之和处为 4>2。
p = 0.5
u = np.array([1.0, 0.0])
v = np.array([0.0, 1.0])
pnorm = lambda w: np.sum(np.abs(w) ** p) ** (1.0 / p)
assert math.isclose(pnorm(u), 1.0)
assert math.isclose(pnorm(v), 1.0)
assert math.isclose(pnorm(u + v), 4.0)
assert pnorm(u + v) > pnorm(u) + pnorm(v)

# Frobenius 范数的复矩阵例，以及逐元素和共轭转置迹两种算法。
B = np.array([[1.0 + 0.0j, 1.0j], [2.0 + 0.0j, -1.0j]])
assert math.isclose(np.linalg.norm(B, "fro"), math.sqrt(7.0), rel_tol=1e-12)
assert math.isclose(float(np.trace(B.conj().T @ B).real), 7.0, rel_tol=1e-12)
assert math.isclose(np.linalg.norm(B, "fro"), 2.6458, abs_tol=5e-5)

# 典型例题的加权范数 A=[[2,1],[1,2]]、x=(1,2)^T。
A = np.array([[2.0, 1.0], [1.0, 2.0]])
xt = np.array([1.0, 2.0])
assert np.allclose(np.linalg.eigvalsh(A), [1.0, 3.0])
assert np.allclose(A @ xt, [4.0, 5.0])
assert math.isclose(float(xt @ A @ xt), 14.0)
assert math.isclose(np.sqrt(float(xt @ A @ xt)), math.sqrt(14.0), rel_tol=1e-12)
assert math.isclose(np.sqrt(14.0), 3.7417, abs_tol=5e-5)

# 自测题 2、4 的结果。
xs = np.array([1.0, -2.0, 2.0])
assert math.isclose(np.linalg.norm(xs, 1), 5.0)
assert math.isclose(np.linalg.norm(xs, 2), 3.0)
assert math.isclose(np.linalg.norm(xs, np.inf), 2.0)
C = np.diag([1.0, 3.0])
zs = np.array([2.0, -1.0])
assert np.allclose(C @ zs, [2.0, -3.0])
assert math.isclose(float(zs @ C @ zs), 7.0)
assert math.isclose(np.sqrt(float(zs @ C @ zs)), math.sqrt(7.0), rel_tol=1e-12)
assert math.isclose(np.sqrt(7.0), 2.6458, abs_tol=5e-5)

print("ALL OK")
