"""N7.5 边值问题的差分法。运行：python3 .work/kp/verify/N7.5.py"""
import math
import numpy as np
import sympy as sp


def near(a, b, tol):
    return abs(a - b) <= tol


def tri(diag, off, rhs):
    n = len(diag)
    A = np.diag(diag) + np.diag([off] * (n - 1), 1) + np.diag([off] * (n - 1), -1)
    return np.linalg.solve(A, rhs)


# -y''=2：二次解无截断误差
y = tri([2.0] * 3, -1.0, np.array([2 / 16] * 3))
assert np.allclose(y, [0.1875, 0.25, 0.1875])
# -y''=1, h=1/2
assert near(1 / 4 / 2, 0.125, 1e-15) and near(0.5 * 0.5 / 2, 0.125, 1e-15)

# -y''=pi^2 sin(pi x), h=1/4
h = 0.25
rhs = np.array([h * h * math.pi ** 2 * math.sin(math.pi * h * i) for i in (1, 2, 3)])
y = tri([2.0] * 3, -1.0, rhs)
assert near(y[0], 0.7446042, 5e-8) and near(y[1], 1.0530293, 5e-8) and near(y[2], y[0], 1e-12)
err = max(abs(y[i] - math.sin(math.pi * h * (i + 1))) for i in range(3))
assert near(err, 0.0530, 5e-5)
assert near(math.pi ** 4 / 96 / 16, 0.0634, 5e-5) and err <= math.pi ** 4 / 96 / 16

# 例 1：表 7.5-1
A = np.diag([-2.01] * 9) + np.diag([1.0] * 8, 1) + np.diag([1.0] * 8, -1)
b = np.array([0.001 * (i + 1) for i in range(9)]); b[8] -= 1
assert near(b[8], -0.991, 1e-12)
yy = np.linalg.solve(A, b)
num = [0.0704894, 0.1426836, 0.2183048, 0.2991089, 0.3869042, 0.4835684, 0.5910684, 0.7114791, 0.8470045]
exa = [0.0704673, 0.1426409, 0.2182436, 0.2990332, 0.3868189, 0.4834801, 0.5909852, 0.7114109, 0.8469633]
ex = [2 * math.sinh(0.1 * (i + 1)) / math.sinh(1) - 0.1 * (i + 1) for i in range(9)]
for v, w in zip(yy, num):
    assert near(v, w, 5e-8)
for v, w in zip(ex, exa):
    assert near(v, w, 1.1e-7)
assert near(ex[0], 0.07046741, 5e-9)
assert near(max(abs(yy - np.array(ex))), 8.83e-5, 5e-7)
assert all(yy[i] > ex[i] for i in range(9))
assert near(2 / 96 * 0.01, 2.08e-4, 5e-7)

# 第三边值例：y''=y, y'(0)=1, y(1)=sh1, h=0.25
h = 0.25; s1 = math.sinh(1)
M = np.zeros((4, 4)); r = np.zeros(4)
M[0, 0] = -2 - h * h; M[0, 1] = 2; r[0] = 2 * h
for i in (1, 2, 3):
    M[i, i] = -2 - h * h; M[i, i - 1] = 1
    if i < 3:
        M[i, i + 1] = 1
r[3] -= s1
yv = np.linalg.solve(M, r)
M2 = M.copy(); r2 = r.copy(); M2[0] = 0; M2[0, :3] = [-3, 4, -1]; r2[0] = 2 * h
yo = np.linalg.solve(M2, r2)
assert near(yo[0], -0.0158, 5e-5) and near(yo[1], 0.2417, 5e-5)
assert near(math.sinh(0.25), 0.2526, 5e-5)
assert near(yv[0], 0.0085, 5e-5) and near(yv[1], 0.2587, 5e-5)
assert near(abs(yv[0]), 8.5e-3, 5e-5) and near(abs(yv[1] - math.sinh(0.25)), 6.1e-3, 5e-5)
assert near(abs(yo[0]), 1.6e-2, 5e-4) and near(abs(yo[1] - math.sinh(0.25)), 1.1e-2, 5e-4)
assert near(-(2 + h * h), -2.0625, 1e-15)

# 例 2：方程组与准确解 1+x^3
A2 = np.array([[-2.52, 2, 0, 0, 0], [1.06, -2.20, 1.02, 0, 0], [0, 1.20, -2.44, 1.12, 0], [0, 0, 1.42, -2.84, 1.30], [0, 0, 0, 1.72, -3.40]])
b2 = np.array([-0.52, -0.072, -0.024, 0.024, -3.048])
y2 = np.linalg.solve(A2, b2)
for v, w in zip(y2, (1.0132, 1.0167, 1.0693, 1.2188, 1.5130)):
    assert near(v, w, 5e-5)
x = sp.symbols("x")
ye = 1 + x ** 3
assert sp.simplify((1 + x ** 2) * sp.diff(ye, x, 2) - x * sp.diff(ye, x) - 3 * ye - (6 * x - 3)) == 0
assert ye.subs(x, 0) - sp.diff(ye, x).subs(x, 0) == 1 and ye.subs(x, 1) == 2
errs = [abs(y2[i] - (1 + (0.2 * i) ** 3)) for i in range(5)]
assert near(errs[0], 1.3e-2, 5e-4) and near(errs[4], 1.0e-3, 5e-5)
# 由差分格式重建矩阵
def row(i):
    xi = 0.2 * i; c = 1 + xi * xi
    lo = c / 0.04 + xi / 0.4; di = -2 * c / 0.04 - 3; up = c / 0.04 - xi / 0.4
    return [lo * 0.04, di * 0.04, up * 0.04], (6 * xi - 3) * 0.04
for i, exp_row in ((1, (1.06, -2.20, 1.02)), (2, (1.20, -2.44, 1.12)), (3, (1.42, -2.84, 1.30)), (4, (1.72, -3.40, 1.56))):
    rr, rh = row(i)
    assert all(near(a, b, 1e-12) for a, b in zip(rr, exp_row))
assert near(row(4)[1] - 1.56 * 2, -3.048, 1e-12)
assert near(row(2)[1], -0.024, 1e-12) and near(row(3)[1], 0.024, 1e-12) and near(row(1)[1], -0.072, 1e-12)
# i=0 行：y_{-1}=y1-0.4(y0-1)
assert near(0.04 * (-2.4 / 0.04 - 3), -2.52, 1e-12) and near(0.04 * (-3) - 0.4, -0.52, 1e-12)

# 例题 1：-y''+y=x，h=1/4
h = 0.25
y3 = tri([2.0625] * 3, -1.0, np.array([h * h * 0.25, h * h * 0.5, h * h * 0.75]))
exv = [xx - math.sinh(xx) / math.sinh(1) for xx in (0.25, 0.5, 0.75)]
for v, w in zip(y3, (0.0348852, 0.0563258, 0.0500368)):
    assert near(v, w, 5e-8)
for v, w in zip(exv, (0.0350476, 0.0565906, 0.0502758)):
    assert near(v, w, 5e-8)
assert near(max(abs(y3 - np.array(exv))), 2.65e-4, 5e-7) and all(y3 < np.array(exv))
assert near(2.65e-4 / 6.5e-4, 0.4, 0.02)
assert near(1 / 96 / 16, 6.5e-4, 5e-6)

# 习题 9：对角占优判断
d = [-(199 - 0.01 * i * i) for i in range(1, 10)]
assert near(d[0], -198.99, 1e-12) and all(abs(v) < 200 for v in d)
# 非线性边值迭代
h = 0.25
A3 = np.diag([-2.0] * 3) + np.diag([1.0] * 2, 1) + np.diag([1.0] * 2, -1)
yk = np.array([3.25, 2.5, 1.75]); hist = []
for k in range(60):
    rr = h * h * 1.5 * yk ** 2; rr[0] -= 4; rr[2] -= 1
    yn = np.linalg.solve(A3, rr); hist.append((yn, max(abs(yn - yk)))); yk = yn
    if hist[-1][1] < 1e-7:
        break
assert len(hist) == 35
for (v, dd), (w, e) in zip(hist[:3], (((2.142578, 1.275391, 0.994141), 1.2246), ((2.827809, 2.085991, 1.496668), 0.8106), ((2.431276, 1.612223, 1.201111), 0.4738))):
    assert np.allclose(v, w, atol=5e-7) and near(dd, e, 5e-5)
assert near(hist[2][1] / hist[1][1], 0.58, 0.03)
assert np.allclose(yk, (2.586745, 1.800795, 1.318863), atol=5e-7)
exn = 4 / (1 + np.array([0.25, 0.5, 0.75])) ** 2
assert np.allclose(exn, (2.56, 1.777778, 1.306122), atol=5e-7)
assert near(max(abs(yk - exn)), 2.7e-2, 5e-4)
xs = sp.symbols("xs"); ys = 4 / (1 + xs) ** 2
assert sp.simplify(sp.diff(ys, xs, 2) - sp.Rational(3, 2) * ys ** 2) == 0
def solve_ex1(N):
    hh = 1 / N
    AA = np.diag([-(2 + hh * hh)] * (N - 1)) + np.diag([1.0] * (N - 2), 1) + np.diag([1.0] * (N - 2), -1)
    bb = np.array([hh ** 3 * (i + 1) for i in range(N - 1)]); bb[-1] -= 1
    yy_ = np.linalg.solve(AA, bb)
    ee = np.array([2 * math.sinh(hh * (i + 1)) / math.sinh(1) - hh * (i + 1) for i in range(N - 1)])
    return max(abs(yy_ - ee)), yy_[N // 2 - 1] - ee[N // 2 - 1]
e10, e20 = solve_ex1(10), solve_ex1(20)
assert near(e10[0], 8.83e-5, 5e-7) and near(e20[0], 2.21e-5, 5e-7) and near(e10[0] / e20[0], 4.00, 5e-3)
assert near(e10[1], 8.53e-5, 5e-7) and near(e20[1], 2.13e-5, 5e-7)
print("ALL OK")
