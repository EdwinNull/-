"""N6.4 Gauss 型求积公式。运行：python3 .work/kp/verify/N6.4.py"""
import math
import numpy as np
import sympy as sp
from numpy.polynomial import legendre as Lg, laguerre as La, hermite as He
from scipy.special import i0, j0
from scipy.integrate import quad


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


# 例 1：矩方程 (6.4-1)
x0 = 0.5 * (1 - 1 / math.sqrt(3)); x1 = 0.5 * (1 + 1 / math.sqrt(3))
for k in range(4):
    near(0.5 * (x0 ** k + x1 ** k), 1 / (k + 1), 1e-15)
assert abs(0.5 * (x0 ** 4 + x1 ** 4) - 0.2) > 1e-3
near(x0, 0.2113249, 5e-8); near(x1, 0.7886751, 5e-8)
x = sp.symbols("x")
assert sp.integrate((x - sp.Rational(1, 2)) ** 2 - sp.Rational(1, 12), (x, 0, 1)) == 0
r = sp.solve(x ** 2 - x + sp.Rational(1, 6), x)
assert sorted(float(v) for v in r) == sorted([x0, x1]) or all(abs(a - b) < 1e-14 for a, b in zip(sorted(float(v) for v in r), [x0, x1]))
for p in (1, x):
    assert sp.integrate(p * (x ** 2 - x + sp.Rational(1, 6)), (x, 0, 1)) == 0

# 两点公式与 e^x、例题 1
E = math.e - 1
g2 = 0.5 * (math.exp(x0) + math.exp(x1))
near(g2, 1.7178964, 5e-8); near(E - g2, 3.85e-4, 5e-7)
simp = (1 + 4 * math.exp(0.5) + math.e) / 6
near(simp, 1.7188612, 5e-8); near(E - simp, -5.79e-4, 5e-7)
t, w = Lg.leggauss(3)
g3 = 0.5 * np.sum(w * np.exp(0.5 * (1 + t)))
near(g3, 1.7182810, 5e-8); near(E - g3, 8.2e-7, 5e-9)
near(E, 1.7182818, 5e-8)
near(sorted(0.5 * (1 + t))[0], 0.1127017, 5e-8); near(sorted(0.5 * (1 + t))[2], 0.8872983, 5e-8)

# 带权 x 的一点公式及余项
near(1 / 3 - 0.5 * 2 / 3, 0, 1e-15)
near(0.25 - 0.5 * 4 / 9, 1 / 36, 1e-15)
assert sp.integrate(x * (x - sp.Rational(2, 3)) ** 2, (x, 0, 1)) == sp.Rational(1, 36)
err = 1 - 0.5 * math.exp(2 / 3)
near(err, 0.0261330, 5e-8)
assert 1 / 72 < err < math.e / 72
near(1 / 72, 0.0138889, 5e-8); near(math.e / 72, 0.0377539, 5e-8)

# 表 6.4-1（含 n=3 印误）
leg = {1: [(0.5773502692, 1)], 2: [(0.7745966692, 0.5555555556), (0, 0.8888888889)],
       3: [(0.8611363116, 0.3478548451), (0.3399810436, 0.6521451549)],
       7: [(0.9602898565, 0.1012285363), (0.1834346425, 0.3626837834)]}
for n, rows in leg.items():
    xs, ws = Lg.leggauss(n + 1)
    for xi, ai in rows:
        j = np.argmin(abs(xs - xi)); near(xs[j], xi, 6e-11); near(ws[j], ai, 6e-11)
    near(ws.sum(), 2, 1e-13)
assert abs(0.861136116 - 0.8611363116) > 1e-7
near(math.sqrt(0.6), 0.7745966692, 5e-11)

# 例 2
f2 = lambda u: np.sqrt(1 + u)
near(1 / 3 * (0 + 4 + math.sqrt(2)), 1.804738, 5e-7)
near(0.555556 * math.sqrt(1 - 0.774597) + 0.888889 + 0.555556 * math.sqrt(1.774597), 1.892727, 5e-7)
gl = np.sum(w * f2(t))
near(gl, 1.892726, 5e-7)
ex2 = 2 / 3 * 2 ** 1.5
near(ex2, 1.885618, 5e-7)
near(gl - ex2, 7.11e-3, 5e-6); near(1.804738 - 1.885618, -8.09e-2, 5e-5)

# 例 3
gfun = lambda s: -math.pi / 2 * math.exp(math.pi / 2) * np.exp(math.pi / 2 * s) * np.sin(math.pi / 2 * s)
for n, v in ((1, -12.3362105), (3, -12.0701895), (5, -12.0703463)):
    xs, ws = Lg.leggauss(n + 1); near(np.sum(ws * gfun(xs)), v, 5e-8)
near(-(1 + math.exp(math.pi)) / 2, -12.0703463, 5e-8)

# 表 6.4-2 与例 4
xs, ws = La.laggauss(6)
near(xs[4], 9.8374674184, 5e-10); near(ws[5], 0.0000008985, 5e-11)
assert abs(9.8374674148 - xs[4]) > 1e-9
xs, ws = La.laggauss(3)
near(np.sum(ws * xs ** 7), 4140.0, 0.1)
v4 = 0.711093 * 0.415775 ** 7 + 0.278518 * 2.294280 ** 7 + 0.010389 * 6.289945 ** 7
near(v4, 4139.9, 0.05)
near(0.711093 * 0.415775 ** 7, 0.0015, 5e-5)
near(0.278518 * 2.294280 ** 7, 93.19, 5e-3)
near(0.010389 * 6.289945 ** 7, 4046.71, 5e-3)
near(5040 - v4, 900, 0.5)
near(36 * 5040 / 720, 252, 1e-12)
U3 = sp.expand(sp.exp(x) * sp.diff(x ** 3 * sp.exp(-x), x, 3))
assert sp.Poly(U3, x).LC() == -1
assert sp.integrate(sp.exp(-x) * U3 ** 2, (x, 0, sp.oo)) == 36
v5 = 0.603154 * 0.3225477 ** 7 + 0.357419 * 1.745761 ** 7 + 0.0388879 * 4.536620 ** 7 + 0.00053929 * 9.395071 ** 7
near(v5, 5039.97, 0.005)
xs, ws = La.laggauss(4); near(np.sum(ws * xs ** 7), 5040, 1e-6)

# 表 6.4-3
xs, ws = He.hermgauss(1); near(ws[0], 1.7724538509, 5e-11)
xs, ws = He.hermgauss(8)
j = np.argmin(abs(xs - 1.1571937124)); near(ws[j], 0.2078023258, 5e-11)
printed = 2 * (0.00019960407 + 0.01707798301 + 0.078023258 + 0.6611470126)
fixed = 2 * (0.00019960407 + 0.01707798301 + 0.2078023258 + 0.6611470126)
near(printed, 1.5129, 5e-5); near(fixed, 1.7724539, 5e-7)
xs, ws = He.hermgauss(3)
near(max(xs), math.sqrt(1.5), 1e-12); near(max(xs), 1.2247448714, 5e-11)
near(ws[0], math.sqrt(math.pi) / 6, 1e-12); near(ws[1], 4 * math.sqrt(math.pi) / 6, 1e-12)
near(math.sqrt(math.pi) / 6, 0.2954089752, 5e-11); near(4 * math.sqrt(math.pi) / 6, 1.1816359006, 5e-11)
near(2 * 0.8862269255 * 0.5, 0.8862269, 5e-8); near(math.sqrt(math.pi) / 2, 0.8862269, 5e-8)
near(2 * 0.8862269255 * 0.25, 0.4431135, 5e-8); near(3 * math.sqrt(math.pi) / 4, 1.3293404, 5e-8)

# Чебышев：例 5 与式 (6.4-6)
bound = lambda n, M: 2 * math.pi * M / (2 ** (2 * n + 2) * math.factorial(2 * n + 2))
near(bound(3, math.e), 1.65e-6, 5e-9); near(bound(4, math.e), 4.60e-9, 5e-12)
assert bound(3, math.e) > 1e-6 >= bound(4, math.e)
I5 = math.pi / 5 * sum(math.exp(math.cos((2 * i + 1) * math.pi / 10)) for i in range(5))
near(I5, 3.977463, 5e-7); near(math.pi * i0(1), 3.9774633, 5e-8)
near(math.cos(math.pi / 10), 0.9510565, 5e-8); near(math.cos(3 * math.pi / 10), 0.5877853, 5e-8)
# 余项常数推导：∫T^2/√(1-x^2)=π/2
near(quad(lambda th: math.cos(5 * th) ** 2, 0, math.pi)[0], math.pi / 2, 1e-12)

# 例题 2：权 √x
mu = [1 / (k + 1.5) for k in range(5)]
for k, v in enumerate((2 / 3, 2 / 5, 2 / 7, 2 / 9, 2 / 11)):
    near(mu[k], v, 1e-15)
c1, c0 = np.linalg.solve(np.array([[mu[1], mu[0]], [mu[2], mu[1]]]), -np.array([mu[2], mu[3]]))
near(c1, -10 / 9, 1e-12); near(c0, 5 / 21, 1e-12)
xa = 5 / 9 - 2 / 9 * math.sqrt(10 / 7); xb = 5 / 9 + 2 / 9 * math.sqrt(10 / 7)
near(xa ** 2 + c1 * xa + c0, 0, 1e-12)
near(xa, 0.2899492, 5e-8); near(xb, 0.8211619, 5e-8)
A1 = (mu[1] - mu[0] * xa) / (xb - xa); A0 = mu[0] - A1
near(A0, 0.2775560, 5e-8); near(A1, 0.3891107, 5e-8)
near(A0 * xa ** 3 + A1 * xb ** 3, 2 / 9, 1e-12); near(2 / 9, 0.2222222, 5e-8)
near(A0 * xa ** 4 + A1 * xb ** 4, 0.1788864, 5e-8); near(2 / 11, 0.1818182, 5e-8)

# 例题 3：cos x / √(1-x^2)
near(bound(2, 1), 1.36e-4, 5e-7); near(bound(3, 1), 6.09e-7, 5e-10)
nodes = [math.cos((2 * i + 1) * math.pi / 8) for i in range(4)]
near(nodes[0], 0.9238795, 5e-8); near(nodes[1], 0.3826834, 5e-8)
near(math.cos(nodes[0]), 0.6027290, 5e-8); near(math.cos(nodes[1]), 0.9276660, 5e-8)
I3 = math.pi / 4 * sum(math.cos(v) for v in nodes)
near(I3, 2.4039388, 5e-8); near(math.pi * j0(1), 2.4039394, 5e-8)
near(math.pi * j0(1) - I3, 5.9e-7, 5e-9)
assert abs(math.pi * j0(1) - I3) <= bound(3, 1)

# 自测 3、5
near(2 / 9 - 2 / 5, -8 / 45, 1e-15)
assert sp.integrate((x ** 2 - sp.Rational(1, 3)) ** 2, (x, -1, 1)) == sp.Rational(8, 45)
near(math.pi / 2 * (0.5 + 0.5), math.pi / 2, 1e-15)
near(quad(lambda th: math.cos(th) ** 2, 0, math.pi)[0], math.pi / 2, 1e-12)

# 习题 11：变换后为一次多项式
near(quad(lambda u: 6 * u / math.sqrt(u * (1 - 3 * u)), 0, 1 / 3)[0], math.pi * math.sqrt(3) / 3, 1e-7)
print("ALL OK")
