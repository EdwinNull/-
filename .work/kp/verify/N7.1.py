"""N7.1 初值问题数值解法的构造及其精度。运行：python3 .work/kp/verify/N7.1.py"""
import math
import sympy as sp


def near(a, b, tol):
    return abs(a - b) <= tol


r6 = lambda v: round(v, 6)

# 例 1：Euler，表 7.1-1
f1 = lambda x, y: y + (1 + x) * y * y
x, y = 1.0, -1.0
ys = [y]; fs = []
for i in range(5):
    fs.append(f1(x, y)); y = y + 0.1 * fs[-1]; x = round(x + 0.1, 10); ys.append(y)
for v, w in zip(ys, (-1, -0.9, -0.8199, -0.753998, -0.698640, -0.651361)):
    assert near(v, w, 1e-6)
for v, w in zip(fs, (1, 0.801, 0.659019, 0.553582, 0.472794)):
    assert near(v, w, 1.5e-6)
assert near(-0.9 + 2.1 * 0.81, 0.801, 1e-12) and near(-0.9 + 0.0801, -0.8199, 1e-12)
assert near(-0.651361 + 1 / 1.5, 0.0153, 5e-5)

# 例 2：改进 Euler，表 7.1-2（6 位舍入）
f2 = lambda x, y: (y - y * y) / x
x, y = 1.0, 0.5
rows = []
for i in range(5):
    fv = r6(f2(x, y)); yb = r6(y + 0.1 * fv); fb = r6(f2(round(x + 0.1, 1), yb)); dy = r6(0.05 * (fv + fb))
    rows.append((y, fv, yb, fb, dy)); y = r6(y + dy); x = round(x + 0.1, 1)
book = [(0.5, 0.25, 0.525, 0.226704, 0.023825), (0.523835, 0.226756, 0.546511, 0.206531, 0.021664),
        (0.545499, 0.206608, 0.566160, 0.188941, 0.019777), (0.565276, 0.189030, 0.584179, 0.173510, 0.018127),
        (0.583403, 0.173603, 0.600783, 0.159898, 0.016675)]
for (a, b, c, d, e), (A, B, C, D, E) in zip(rows, book):
    assert near(a, A, 1.5e-6) and near(b, B, 1.5e-6) and near(d, D, 1.5e-6)
assert near(rows[0][4], 0.023835, 1e-9) and not near(0.023825, 0.023835, 5e-6)
assert near(rows[4][2], 0.600763, 1e-9) and not near(0.600783, 0.600763, 5e-6)
assert near(rows[0][3], 0.226705, 1e-9)
assert near(y, 0.600078, 1.5e-6) and near(y - 0.6, 7.8e-5, 1.5e-6)
assert near(0.05 * (0.25 + 0.226705), 0.023835, 5e-7)
assert near(0.525 * 0.475 / 1.1, 0.226705, 5e-7)
assert near(rows[2][0] - 1.2 / 2.2, 4.4e-5, 5e-7)
assert near(1.1 / 2.1, 0.523810, 5e-7) and near(1.2 / 2.2, 0.545455, 5e-7)

# y'=-y 一步比较
e = math.exp(-0.1)
assert near(e, 0.9048374, 5e-8)
eul, bwd, trap, imp = 0.9, 1 / 1.1, 0.95 / 1.05, 1 - 0.05 * 1.9
assert near(bwd, 0.9090909, 5e-8) and near(trap, 0.9047619, 5e-8) and near(imp, 0.905, 1e-12)
assert near(e - eul, 4.84e-3, 5e-6) and near(e - eul, 0.0048374, 5e-8)
assert near(e - bwd, -4.25e-3, 5e-6) and near(e - bwd, -0.0042535, 5e-8)
assert near(e - trap, 7.55e-5, 5e-8) and near(e - imp, -1.63e-4, 5e-7)
assert near(0.001 / 12, 8.33e-5, 5e-8)
# 向后 Euler 迭代
it = [0.9]
for k in range(4):
    it.append(1 - 0.1 * it[-1])
assert [round(v, 6) for v in it[1:]] == [0.91, 0.909, 0.9091, 0.90909]

# 整体误差
E1 = math.exp(-1)
assert near(0.9 ** 10, 0.3486784, 5e-8) and near(E1 - 0.9 ** 10, 0.0192010, 5e-8)
assert near(0.95 ** 20, 0.3584859, 5e-8) and near(E1 - 0.95 ** 20, 0.0093935, 5e-8)
assert near((E1 - 0.9 ** 10) / (E1 - 0.95 ** 20), 2.04, 5e-3)
assert near(0.5 * (math.e - 1), 0.859, 5e-4)
assert near(0.859 * 0.1, 0.0859, 1e-9) and near(0.859 * 0.05, 0.0430, 5e-5)

# 梯形局部误差主项 -h^3/12 y'''
h, X = sp.symbols("h X")
yf = sp.Function("y")
Y = sp.exp(2 * X) + X ** 3  # 任取光滑函数检验
lhs = Y.subs(X, h) - Y.subs(X, 0) - h / 2 * (sp.diff(Y, X).subs(X, 0) + sp.diff(Y, X).subs(X, h))
ser = sp.series(lhs, h, 0, 4).removeO()
assert sp.simplify(ser - (-h ** 3 / 12 * sp.diff(Y, X, 3).subs(X, 0))) == 0
# Euler 局部 h^2/2 y''
lhs = Y.subs(X, h) - Y.subs(X, 0) - h * sp.diff(Y, X).subs(X, 0)
assert sp.simplify(sp.series(lhs, h, 0, 3).removeO() - h ** 2 / 2 * sp.diff(Y, X, 2).subs(X, 0)) == 0

# y''+y=0 化为方程组一步 Euler
assert near(math.sin(0.1), 0.0998334, 5e-8) and near(math.cos(0.1), 0.9950042, 5e-8)

# 自测
assert near(1 + 0.05 * (1 + 1.2), 1.11, 1e-12)
assert near(1 / 1.2, 0.833333, 5e-7)
# 习题 1 的正确准确解
xx = sp.symbols("xx")
sol = xx ** 2 - xx + 1 - sp.exp(-xx)
assert sp.simplify(sp.diff(sol, xx) - (xx ** 2 + xx - sol)) == 0 and sol.subs(xx, 0) == 0
# 例题 4：方程组改进 Euler
hh = 0.1; yv, zv = 0.0, 1.0
out = []
for i in range(2):
    fy, fz = zv, -yv; yb, zb = yv + hh * fy, zv + hh * fz
    yv, zv = yv + hh / 2 * (fy + zb), zv + hh / 2 * (fz - yb); out.append((yb, zb, yv, zv))
assert near(out[0][2], 0.1, 1e-12) and near(out[0][3], 0.995, 1e-12)
assert near(out[1][0], 0.1995, 1e-12) and near(out[1][1], 0.985, 1e-12)
assert near(out[1][2], 0.199, 1e-12) and near(out[1][3], 0.980025, 1e-12)
assert near(math.sin(0.2), 0.1986693, 5e-8) and near(math.cos(0.2), 0.9800666, 5e-8)
assert near(math.sin(0.2) - 0.199, -3.3e-4, 5e-6) and near(math.cos(0.2) - 0.980025, 4.2e-5, 5e-7)
assert near(math.exp(-1) / 2, 0.184, 5e-4) and near(math.exp(-1) / 2 * 0.1, 0.0184, 5e-5)
hs = sp.symbols("hs")
assert sp.simplify(sp.series(sp.log(1 - hs) / hs, hs, 0, 3).removeO() - (-1 - hs / 2 - hs ** 2 / 3)) == 0
print("ALL OK")
