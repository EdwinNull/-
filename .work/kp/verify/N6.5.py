"""N6.5 二重积分的计算方法。运行：python3 .work/kp/verify/N6.5.py"""
import math
import numpy as np
from numpy.polynomial import legendre as Lg
from scipy.integrate import dblquad


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)
    return True


def simp_w(n):
    return np.array([1 if i in (0, 2 * n) else (4 if i % 2 else 2) for i in range(2 * n + 1)], float)


def simp2(f, a, b, c, d, n, m):
    h = (b - a) / n; k = (d - c) / m
    xs = np.linspace(a, b, 2 * n + 1); ys = np.linspace(c, d, 2 * m + 1)
    W = np.outer(simp_w(n), simp_w(m)) * h * k / 36
    return sum(W[i, j] * f(xs[i], ys[j]) for i in range(2 * n + 1) for j in range(2 * m + 1)), W


# 式 (6.5-3)：单个小矩形的权表
_, W = simp2(lambda x, y: 1.0, 0, 1, 0, 1, 1, 1)
assert np.allclose(W * 36, [[1, 4, 1], [4, 16, 4], [1, 4, 1]])
assert near(W.sum(), 1, 1e-15)
_, W = simp2(lambda x, y: 1.0, 0, 3, 0, 2, 3, 2)
assert near(W.sum(), 6, 1e-12)
assert W.size == 35

# K1 曲边区域例
g05 = 0.5 / 6 * (1 + 4 * math.exp(0.25) + math.exp(0.5))
g1 = (1 + 4 * math.exp(0.5) + math.e) / 6
assert near(g05, 0.6487352, 5e-8)
assert near(g1, 1.7188612, 5e-8)
Ik = 4 / 6 * g05 + g1 / 6
assert near(Ik, 0.7189670, 5e-8)
assert near(math.e - 2, 0.7182818, 5e-8)
assert near(Ik - (math.e - 2), 6.9e-4, 5e-6)

# K2、K3 可分离例
S2, _ = simp2(lambda x, y: math.exp(x + y), 0, 1, 0, 1, 1, 1)
assert near(S2, g1 ** 2, 1e-12)
assert near(S2, 2.9544837, 5e-8)
ex = (math.e - 1) ** 2
assert near(ex, 2.9524924, 5e-8)
assert near(ex - S2, -1.99e-3, 5e-6)
x0 = 0.5 - 0.5 / math.sqrt(3); x1 = 0.5 + 0.5 / math.sqrt(3)
G1 = 0.5 * (math.exp(x0) + math.exp(x1))
assert near(G1, 1.7178964, 5e-8)
assert near(G1 ** 2, 2.9511680, 5e-8)
assert near(ex - G1 ** 2, 1.32e-3, 5e-6)

# 例 1
f = lambda x, y: math.log(x + 2 * y)
I = dblquad(lambda y, x: f(x, y), 1.4, 2, 1, 1.5, epsabs=1e-14)[0]
assert near(I, 0.4295545, 5e-8)
assert near(I, 0.429554526, 2e-9)
IS, W = simp2(f, 1.4, 2, 1, 1.5, 2, 1)
assert near(IS, 0.42955244, 5e-9)
assert near(I - IS, 2.09e-6, 5e-9)
assert W.size == 15
assert near(0.3 * 0.5 / 36, 0.15 / 36, 1e-15)
t, w = Lg.leggauss(3)
inner = sum(w[i] * w[j] * math.log(0.3 * t[i] + 0.5 * t[j] + 4.2) for i in range(3) for j in range(3))
assert near(inner, 5.7273937, 5e-8)
assert near(0.075 * inner, 0.42955453, 5e-9)
assert near(abs(0.075 * inner - I), 4e-9, 1e-9)
assert near(0.6 * 0.5 / 4, 0.075, 1e-15)
assert near(0.075, 3 / 40, 1e-15)
# 带 40/3 时的值
assert near(40 / 3 * I, 5.7273937, 5e-8)
assert near(math.sqrt(0.6), 0.7745967, 5e-8)
for u in (-1, 1):
    assert near(1.7 + 0.3 * u, 1.4 if u < 0 else 2.0, 1e-15)
    assert near(1.25 + 0.25 * u, 1.0 if u < 0 else 1.5, 1e-15)

# 例题 2：∫0^1∫0^{x^2}(x+y)
gx = lambda x: x * x / 2 * ((x + 0) + (x + x * x))
assert near(gx(0.5), 0.15625, 1e-15)
assert near(gx(1), 1.5, 1e-15)
I2 = 4 / 6 * gx(0.5) + gx(1) / 6
assert near(I2, 0.3541667, 5e-8)
assert near(4 / 6 * 0.15625, 0.1041667, 5e-8)
assert near(0.25 + 0.1, 0.35, 1e-15)
assert near(0.35 - I2, -4.17e-3, 5e-6)
assert near(-12 / 2880, -4.17e-3, 5e-6)
assert near(0.35 - I2, -12 / 2880, 1e-12)

# 自测 3
xs, ws = Lg.leggauss(2)
v = sum(ws[i] * ws[j] * (1 + xs[i]) * (0.5 + 0.5 * xs[j]) for i in range(2) for j in range(2)) * (2 * 1 / 4)
assert near(v, 1.0, 1e-14)
# 例题 3：两向复化梯形
T2 = 0.25 * (1 + 2 * math.exp(0.5) + math.e)
assert near(T2, 1.7539311, 5e-8)
assert near(T2 ** 2, 3.0762743, 5e-8)
pts = [(0, 0), (1, 0), (0, 1), (1, 1)]
edge = [(0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5)]
val = sum(math.exp(x + y) / 16 for x, y in pts) + sum(math.exp(x + y) / 8 for x, y in edge) + math.e / 4
assert near(val, T2 ** 2, 1e-12)
assert near(ex - T2 ** 2, -0.1237818, 5e-8)
assert near(2 * ex, 5.9049849, 5e-8)
assert near(-2 * ex / 48, -0.1230205, 5e-8)
assert abs((ex - T2 ** 2) / (-2 * ex / 48) - 1) < 0.01
assert near(abs(I - IS) / abs(0.075 * inner - I), 580, 30)
print("ALL OK")
