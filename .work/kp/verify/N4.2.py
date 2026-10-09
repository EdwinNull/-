"""N4.2 Newton 迭代。运行：python3 .work/kp/verify/N4.2.py"""
import math
import numpy as np

# 例 1
x = 0.5
for k, expect in [(1, 0.571020439), (2, 0.567155568), (3, 0.56714329)]:
    ex = math.exp(x)
    x = x - (x * ex - 1) / ((1 + x) * ex)
    assert abs(x - expect) < 5e-9
assert abs(x - 0.5671432904) < 1e-9

# 例 2
def m1(x):
    return x - (x * x - 2) / (4 * x)
def m2(x):
    return x - (x * x - 2) / (2 * x)
def m3(x):
    return x - x * (x * x - 2) / (x * x + 2)
x = 1.5
assert abs(m1(x) - 1.45833333) < 1e-8
x = 1.5
assert abs(m2(x) - 1.41666667) < 1e-8
x = m2(m2(m2(1.5)))
assert abs(x - math.sqrt(2)) < 1e-9
x = m3(m3(m3(1.5)))
assert abs(x - math.sqrt(2)) < 1e-9
# 方法 1 三次后仍差约 0.01
x = m1(m1(m1(1.5)))
assert abs(x - 1.425497619) < 1e-8
assert abs(x - math.sqrt(2)) > 1e-2

# 例 3
x = np.array([1.5, 1.0])
for _ in range(4):
    J = np.array([[1.0, 2], [4 * x[0], 2 * x[1]]])
    f = np.array([x[0] + 2 * x[1] - 3, 2 * x[0] ** 2 + x[1] ** 2 - 5])
    x = x + np.linalg.solve(J, -f)
assert abs(x[0] - 1.4880) < 5e-5
assert abs(x[1] - 0.75598) < 5e-6

# 例 4
x0 = np.array([0.5, 1.0])
def F(z):
    return np.array([z[0] ** 3 - z[1] ** 2 - 1, z[0] * z[1] ** 3 - z[1] - 4])
f0 = F(x0)
assert abs(np.linalg.norm(f0, np.inf) - 4.5) < 1e-12
J = np.array([[3 * x0[0] ** 2, -2 * x0[1]], [x0[1] ** 3, 3 * x0[0] * x0[1] ** 2 - 1]])
dx = np.linalg.solve(J, -f0)
x1 = x0 + dx
assert abs(x1[0] - 4.68421053) < 1e-7
f1 = F(x1)
assert abs(f1[0] - 99.11809287) < 1e-5
assert abs(f1[1] - 14.71356) < 1e-4
assert abs(f1[1] + 1.288234453) > 1
assert np.linalg.norm(f1, np.inf) > np.linalg.norm(f0, np.inf)
x2 = x0 + 0.25 * dx
f2 = F(x2)
assert abs(f2[0] - 1.354776509) < 1e-6
assert abs(f2[1] + 2.757782707) < 1e-6
assert np.linalg.norm(f2, np.inf) < np.linalg.norm(f0, np.inf)

print("ALL OK")
