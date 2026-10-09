"""N6.6 数值微分。运行：python3 .work/kp/verify/N6.6.py"""
import math
import sympy as sp
from scipy.special import j0, j1


def near(a, b, tol):
    return abs(a - b) <= tol


x, h, t = sp.symbols("x h t")
f = sp.Function("f")

# 式 (6.6-3)：差分与差商，x^3 与 x^4
v = [i ** 3 for i in range(4)]
d1 = [v[i + 1] - v[i] for i in range(3)]; d2 = [d1[i + 1] - d1[i] for i in range(2)]; d3 = d2[1] - d2[0]
assert d1 == [1, 7, 19] and d2 == [6, 12] and d3 == 6
assert d3 / math.factorial(3) == 1
v4 = [i ** 4 for i in range(5)]
assert v4[4] - 4 * v4[3] + 6 * v4[2] - 4 * v4[1] + v4[0] == 24
assert 24 / math.factorial(4) == 1

# 式 (6.6-6)：n=2 时由 C_t^i 推出三点公式
f0, f1, f2 = sp.symbols("f0 f1 f2")
D1 = f1 - f0; D2 = f2 - 2 * f1 + f0
p = f0 + t * D1 + t * (t - 1) / 2 * D2
dp = sp.diff(p, t)
assert sp.simplify(dp.subs(t, 0) - (-3 * f0 + 4 * f1 - f2) / 2) == 0
assert sp.simplify(dp.subs(t, 1) - (f2 - f0) / 2) == 0
assert sp.simplify(dp.subs(t, 2) - (f0 - 4 * f1 + 3 * f2) / 2) == 0
assert sp.simplify(sp.diff(p, t, 2) - D2) == 0

# 余项系数：用多项式检验 (6.6-7)(6.6-8)(6.6-9) 主项
X = sp.symbols("X")
for poly in (X ** 2, X ** 3, X ** 4):
    fp = sp.Lambda(X, poly)
    F = [fp(k * h) for k in range(3)]
    e0 = sp.expand(sp.diff(poly, X).subs(X, 0) - (-3 * F[0] + 4 * F[1] - F[2]) / (2 * h))
    e1 = sp.expand(sp.diff(poly, X).subs(X, h) - (F[2] - F[0]) / (2 * h))
    if poly == X ** 3:
        assert sp.simplify(e0 - h ** 2 / 3 * 6) == 0
        assert sp.simplify(e1 + h ** 2 / 6 * 6) == 0
    if poly == X ** 4:
        e11 = sp.expand(sp.diff(poly, X, 2).subs(X, h) - (F[0] - 2 * F[1] + F[2]) / h ** 2)
        assert sp.simplify(e11 + h ** 2 / 12 * 24) == 0
# 式 (6.6-9) 端点式：x^4、x0=0、h=1，-24ξ+4=-14 有 ξ=0.75∈(0,2)
assert 0 - (0 - 2 * 1 + 16) == -14
assert 0 < (4 + 14) / 24 < 2

# Taylor：中心一阶与二阶的升幂展开
ser1 = sp.series((sp.exp(h) - sp.exp(-h)) / (2 * h), h, 0, 7).removeO()
assert sp.simplify(1 - ser1 - (-h ** 2 / 6 - h ** 4 / 120 - h ** 6 / 5040)) == 0
ser2 = sp.series((sp.exp(h) - 2 + sp.exp(-h)) / h ** 2, h, 0, 7).removeO()
assert sp.simplify(1 - ser2 - (-h ** 2 / 12 - h ** 4 / 360 - h ** 6 / 20160)) == 0
# 外推五点公式（一阶）与（二阶）
g = lambda u: u ** 5
I1 = lambda hh: (g(1 + hh) - g(1 - hh)) / (2 * hh)
I2 = (4 * I1(sp.Rational(1, 2) * h) - I1(h)) / 3
five = (g(1 - h) - 8 * g(1 - h / 2) + 8 * g(1 + h / 2) - g(1 + h)) / (6 * h)
assert sp.simplify(I2 - five) == 0
assert sp.simplify(sp.Rational(1, 3) * (4 / (h) - 1 / (2 * h)) - sp.Rational(7, 6) / h) == 0
# f'' 五点公式及误差 -h^4/1440 f^(6)
fe = sp.exp
Dh = lambda hh: (fe(-hh) - 2 + fe(hh)) / hh ** 2
F5 = (4 * Dh(h / 2) - Dh(h)) / 3
ser = sp.series(1 - F5, h, 0, 6).removeO()
assert sp.simplify(ser - h ** 4 / 1440) == 0
F5b = (-fe(-h) + 16 * fe(-h / 2) - 30 + 16 * fe(h / 2) - fe(h)) / (3 * h ** 2)
assert sp.simplify(sp.series(F5 - F5b, h, 0, 6).removeO()) == 0

# 例 1：J0
xs = [0.96, 0.98, 1.00, 1.02, 1.04]
tab = [0.7825361, 0.7739332, 0.7651977, 0.7563321, 0.7473390]
for a, b in zip(xs, tab):
    assert near(j0(a), b, 5e-8)
y = dict(zip(xs, tab))
d1v = (y[1.02] - y[0.98]) / 0.04
d2v = (y[0.98] - 2 * y[1.00] + y[1.02]) / 0.0004
d5v = (y[0.96] - 8 * y[0.98] + 8 * y[1.02] - y[1.04]) / 0.24
assert near(d1v, -0.4400275, 5e-8)
assert near(d2v, -0.325250, 5e-7)
assert near(d5v, -0.4400488, 5e-8)
assert near(y[1.02] - y[0.98], -0.0176011, 1e-10)
assert near(y[0.98] - 2 * y[1.00] + y[1.02], -0.0001301, 1e-10)
ex1 = -j1(1); ex2 = -j0(1) + j1(1)
assert near(ex1, -0.4400506, 5e-8); assert near(ex2, -0.3251471, 5e-8)
assert near(abs(d1v - ex1), 2.3e-5, 5e-7)
assert near(abs(d5v - ex1), 1.8e-6, 5e-8)
assert near(abs(d2v - ex2), 1.0e-4, 5e-6)
yy = {a: j0(a) for a in xs}
exact2 = (yy[0.98] - 2 * yy[1.00] + yy[1.02]) / 0.0004
assert abs(exact2 - ex2) < 1e-5 < abs(d2v - ex2)
assert near(4 * 5e-8 / 0.0004, 5e-4, 1e-12)
assert near(0.24, 6 * 0.04, 1e-15) and near(0.24, 12 * 0.02, 1e-15)

# e^x 自编
E = math.exp
c1 = (E(0.1) - E(-0.1)) / 0.2
assert near(c1, 1.0016675, 5e-8); assert near(1 - c1, -1.6675e-3, 5e-8)
assert near(-0.01 / 6, -1.6667e-3, 5e-8)
assert near(-0.01 / 6 - 1e-4 / 120, 1 - c1, 1e-8)
c05 = (E(0.05) - E(-0.05)) / 0.1
assert near(c05, 1.0004167, 5e-8)
i2 = (4 * c05 - c1) / 3
assert near(i2, 0.9999998, 5e-8); assert near(1 - i2, 2e-7, 5e-8)
tp = (E(0.1) - 1) / 0.1; te = (-3 + 4 * E(0.1) - E(0.2)) / 0.2
assert near(tp, 1.0517092, 5e-8); assert near(1 - tp, -0.0517092, 5e-8)
assert near(te, 0.9964046, 5e-8); assert near(1 - te, 0.0035954, 5e-8)
assert near(0.01 / 3, 0.0033333, 5e-8)
assert near((1 - tp) / (1 - c1), 31, 1.5)

# 舍入实验：sin 在 1 处，四位小数
S = math.sin; cs = math.cos(1)
assert near(cs, 0.5403023, 5e-8)
r4 = lambda u: round(S(u), 4)
errs = {hh: (r4(1 + hh) - r4(1 - hh)) / (2 * hh) - cs for hh in (0.1, 0.05, 0.002, 0.001)}
assert near(errs[0.1], -8.0e-4, 5e-6)
assert near(errs[0.05], -3.0e-4, 5e-6)
assert near(errs[0.002], -1.53e-2, 5e-5)
assert near(errs[0.001], 9.7e-3, 5e-5)
assert near((3 * 5e-5 / math.sin(1)) ** (1 / 3), 0.056, 5e-4)

# 例题 2：sin 外推
a1 = (S(1.2) - S(0.8)) / 0.4; a2 = (S(1.1) - S(0.9)) / 0.2
assert near(a1, 0.5367075, 5e-8); assert near(a2, 0.5394023, 5e-8)
assert near((a2 - a1) / 3, 0.0008983, 5e-8)
ext = a2 + (a2 - a1) / 3
assert near(ext, 0.5403005, 5e-8)
assert near((S(0.8) - 8 * S(0.9) + 8 * S(1.1) - S(1.2)) / 1.2, ext, 1e-12)
assert near(cs - ext, 1.8e-6, 5e-8)
assert near(cs - a1, 3.6e-3, 5e-5); assert near(cs - a2, 9.0e-4, 5e-6)
assert near(5e-5 / 0.1, 5e-4, 1e-12)

# 例题 3：最优步长
hs = (3 * 5e-7) ** (1 / 3)
assert near(hs, 0.01145, 5e-6)
E1 = 5e-7 / hs; E2 = hs ** 2 / 6
assert near(E1, 4.37e-5, 5e-8); assert near(E2, 2.18e-5, 5e-8); assert near(E1 + E2, 6.55e-5, 5e-8)
assert near(E1 / E2, 2, 1e-9)

# K8：样条在节点处的一阶导数
M = [0, 6, 12]; fv = [0, 1, 8]; hh = 1
val = -M[1] * (2 - 1) ** 2 / (2 * hh) + M[2] * 0 + (fv[2] - fv[1]) / hh - (M[2] - M[1]) / 6 * hh
assert val == 3
assert (fv[2] - fv[1]) - (2 * M[1] + M[2]) / 6 == 3
print("ALL OK")
