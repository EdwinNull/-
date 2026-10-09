"""N7.3 线性多步法。运行：python3 .work/kp/verify/N7.3.py"""
import math
import sympy as sp
from scipy.integrate import solve_ivp

R = sp.Rational


def near(a, b, tol):
    return abs(a - b) <= tol


def rk4(f, x, y, h):
    k1 = f(x, y); k2 = f(x + h / 2, y + h / 2 * k1); k3 = f(x + h / 2, y + h / 2 * k2); k4 = f(x + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# Adams 系数（数值积分法）
t = sp.symbols("t")


def coeffs(nodes):
    out = []
    for n in nodes:
        L = 1
        for m in nodes:
            if m != n:
                L *= (t - m) / sp.Integer(n - m)
        out.append(sp.integrate(L, (t, 0, 1)) * 24)
    return out


assert coeffs([0, -1, -2, -3]) == [55, -59, 37, -9]
assert coeffs([1, 0, -1, -2]) == [9, 19, -5, 1]
assert sp.integrate((t + 1) * (t + 2) * (t + 3) / 6, (t, 0, 1)) == R(55, 24)


# 式 (7.3-8)、(7.3-9)
def cond(a, b, m):
    eqs = [sum(a.values())]
    for j in range(1, m + 1):
        eqs.append(sum((-k) ** j * ak for k, ak in a.items()) + j * sum((-k) ** (j - 1) * bk if not (k == 0 and j == 1) else bk for k, bk in b.items()))
    return eqs


def errc(a, b, m):
    s = 1 - sum((-k) ** (m + 1) * ak for k, ak in a.items() if k >= 1) - (m + 1) * sum((-k) ** m * bk for k, bk in b.items())
    return sp.nsimplify(s) / sp.factorial(m + 1)


AB4 = ({0: 1}, {0: R(55, 24), 1: R(-59, 24), 2: R(37, 24), 3: R(-9, 24)})
AM4 = ({0: 1}, {-1: R(9, 24), 0: R(19, 24), 1: R(-5, 24), 2: R(1, 24)})
MIL = ({3: 1}, {0: R(8, 3), 1: R(-4, 3), 2: R(8, 3)})
HAM = ({0: R(9, 8), 2: R(-1, 8)}, {-1: R(3, 8), 0: R(6, 8), 1: R(-3, 8)})
for a, b in (AB4, AM4, MIL, HAM):
    assert all(e == 1 for e in cond(a, b, 4))
assert errc(*AB4, 4) == R(251, 720) and errc(*AM4, 4) == R(-19, 720)
assert errc(*MIL, 4) == R(14, 45) and errc(*HAM, 4) == R(-1, 40)
assert R(112, 3) / 120 == R(14, 45)
assert -3 + R(8, 3) - R(4, 3) + R(8, 3) == 1 and 81 - 4 * R(-4, 3) - 32 * R(8, 3) == 1
# Milne 系数由五个方程解出
a0, a1, a2, a3, bm, b0, b1, b2, b3 = sp.symbols("a0 a1 a2 a3 bm b0 b1 b2 b3")
E = [a0 + a1 + a2 + a3 - 1,
     -a1 - 2 * a2 - 3 * a3 + bm + b0 + b1 + b2 + b3 - 1,
     a1 + 4 * a2 + 9 * a3 + 2 * bm - 2 * b1 - 4 * b2 - 6 * b3 - 1,
     -a1 - 8 * a2 - 27 * a3 + 3 * bm + 3 * b1 + 12 * b2 + 27 * b3 - 1,
     a1 + 16 * a2 + 81 * a3 + 4 * bm - 4 * b1 - 32 * b2 - 108 * b3 - 1]
sol = sp.solve([e.subs({a0: 0, a1: 0, a2: 0, bm: 0}) for e in E], [a3, b0, b1, b2, b3])
assert sol == {a3: 1, b0: R(8, 3), b1: R(-4, 3), b2: R(8, 3), b3: 0}
sol = sp.solve([e.subs({a1: 0, a3: 0, b2: 0, b3: 0}) for e in E], [a0, a2, bm, b0, b1])
assert sol == {a0: R(9, 8), a2: R(-1, 8), bm: R(3, 8), b0: R(3, 4), b1: R(-3, 8)}
# AB2、Euler 两步
assert errc({0: 1}, {0: R(3, 2), 1: R(-1, 2)}, 2) == R(5, 12)
assert errc({1: 1}, {0: 2}, 2) == R(1, 3)
assert cond({1: 1}, {0: 2}, 3) == [1, 1, 1, -1]
# 显式两步 3 阶（不稳定例）
assert all(e == 1 for e in cond({0: -4, 1: 5}, {0: 4, 1: 2}, 3))
hh = 0.1; yy = [1, math.exp(-0.1)]
for i in range(1, 8):
    yy.append(-4 * yy[i] + 5 * yy[i - 1] + hh * (4 * (-yy[i]) + 2 * (-yy[i - 1])))
errs = [v - math.exp(-hh * k) for k, v in enumerate(yy)]
assert abs(errs[2]) < 2e-5 and near(errs[8], -0.25, 0.01)
# 开型 Newton-Cotes 与 Milne
assert [4 * w for w in (R(2, 3), R(-1, 3), R(2, 3))] == [R(8, 3), R(-4, 3), R(8, 3)]

# y'=-y 的数值例
h = 0.1
ex = [math.exp(-h * k) for k in range(5)]
f = [-v for v in ex]
ab2 = ex[1] + h / 2 * (3 * f[1] - f[0])
assert near(ab2, 0.8191118, 5e-8) and near(math.exp(-0.2), 0.8187308, 5e-8)
assert near(math.exp(-0.2) - ab2, -3.81e-4, 5e-7) and near(5 / 12 * h ** 3 * (-math.exp(-0.1)), -3.77e-4, 5e-7)
p = ex[3] + h / 24 * (55 * f[3] - 59 * f[2] + 37 * f[1] - 9 * f[0])
c = ex[3] + h / 24 * (9 * (-p) + 19 * f[3] - 5 * f[2] + f[1])
assert near(p, 0.6703229, 5e-8) and near(c, 0.6703197, 5e-8) and near(ex[4], 0.6703200, 5e-8)
assert near(ex[4] - p, -2.87e-6, 5e-9) and near(ex[4] - c, 3.1e-7, 5e-9)
assert near(251 / 720 * h ** 5 * (-math.exp(-0.3)), -2.58e-6, 5e-9)
assert near(abs((ex[4] - p) / (ex[4] - c)), 9.3, 0.2)
assert near(f[2], -0.8187308, 5e-8) and near(f[3], -0.7408182, 5e-8)

# 例 2：表 7.3-2
g = lambda x, y: 0.25 * y * y + x * x
ys = [-1.0]; xs = [0.0]
for i in range(3):
    ys.append(rk4(g, xs[-1], ys[-1], 0.1)); xs.append(xs[-1] + 0.1)
for v, w in zip(ys, (-1, -0.97528, -0.94978, -0.92154)):
    assert near(v, w, 5e-6)
fs = [g(a, b) for a, b in zip(xs, ys)]
for v, w in zip(fs, (0.25, 0.24779, 0.26552, 0.30232)):
    assert near(v, w, 1.2e-5)
res = []
for i in (3, 4):
    db = 0.1 / 24 * (55 * fs[i] - 59 * fs[i - 1] + 37 * fs[i - 2] - 9 * fs[i - 3]); yb = ys[i] + db; fb = g(xs[i] + 0.1, yb)
    dy = 0.1 / 24 * (9 * fb + 19 * fs[i] - 5 * fs[i - 1] + fs[i - 2]); ys.append(ys[i] + dy); xs.append(xs[i] + 0.1); fs.append(g(xs[-1], ys[-1]))
    res.append((db, yb, fb, dy, ys[-1]))
assert near(res[0][0], 0.03283, 5e-6) and near(res[0][1], -0.88871, 5e-6) and near(res[0][2], 0.35745, 5e-6)
assert near(res[0][3], 0.03284, 5e-6) and near(res[0][4], -0.88870, 5e-6)
assert near(res[1][1], -0.84946, 5e-6) and near(res[1][4], -0.84945, 1e-5)
assert near(55 * 0.30232 - 59 * 0.26552 + 37 * 0.24779 - 9 * 0.25, 7.88015, 5e-6)
assert near(9 * 0.35745 + 19 * 0.30232 - 5 * 0.26552 + 0.24779, 7.88132, 5e-6)
assert near(0.1 / 24 * 7.88015, 0.03283, 5e-6) and near(0.1 / 24 * 7.88132, 0.03284, 5e-6)

# 例 1：表 7.3-1 复算与高精度解
F = lambda x, y: math.sinh(0.5 * y + x) / 1.5 + 0.5 * y
Y = [0.0]; X = [0.0]
for i in range(3):
    Y.append(rk4(F, X[-1], Y[-1], 0.05)); X.append(X[-1] + 0.05)
FF = [F(a, b) for a, b in zip(X, Y)]
for i in range(3, 10):
    Y.append(Y[i] + 0.05 / 24 * (55 * FF[i] - 59 * FF[i - 1] + 37 * FF[i - 2] - 9 * FF[i - 3])); X.append(X[i] + 0.05); FF.append(F(X[-1], Y[-1]))
for v, w in zip(Y[5:], (0.022484, 0.032933, 0.045624, 0.060694, 0.078294, 0.098593)):
    assert near(v, w, 6e-7)
printed = (0.022485, 0.032936, 0.045628, 0.060698, 0.078301, 0.098596)
diffs = [b - a for a, b in zip(Y[5:], printed)]
assert all(5e-7 < d < 7.6e-6 for d in diffs)
s = solve_ivp(lambda x, y: [F(x, y[0])], [0, 0.5], [0], rtol=1e-12, atol=1e-14)
assert near(s.y[0, -1], 0.0985969, 5e-8)
assert near(Y[4], 0.014156, 5e-7) and near(Y[2], 0.003431, 5e-7)
assert near(0.05 / 24 * (55 * 0.10694 - 59 * 0.06964 + 37 * 0.03404), 0.006318, 5e-7)
assert near(0.007838 + 0.006318, 0.014156, 1e-12)
print("ALL OK")
