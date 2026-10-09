"""N7.2 Runge-Kutta 方法。运行：python3 .work/kp/verify/N7.2.py"""
import math
import sympy as sp


def near(a, b, tol):
    return abs(a - b) <= tol


def rk4(f, x, y, h):
    k1 = f(x, y); k2 = f(x + h / 2, y + h / 2 * k1); k3 = f(x + h / 2, y + h / 2 * k2); k4 = f(x + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), (k1, k2, k3, k4)


# y'=y^2 一步比较
g = lambda x, y: y * y
ex = 1 / 0.9
assert near(ex, 1.1111111, 5e-8)
tay = 1 + 0.1 + 0.01 / 2 * 2
ie = 1 + 0.05 * (1 + g(0.1, 1.1)); mid = 1 + 0.1 * g(0.05, 1.05)
r4, K = rk4(g, 0, 1, 0.1)
assert near(tay, 1.11, 1e-12) and near(ex - tay, 1.11e-3, 5e-6)
assert near(g(0.1, 1.1), 1.21, 1e-12) and near(ie, 1.1105, 1e-12) and near(ex - ie, 6.11e-4, 5e-7)
assert near(g(0.05, 1.05), 1.1025, 1e-12) and near(mid, 1.11025, 1e-12) and near(ex - mid, 8.61e-4, 5e-7)
assert near(K[2], 1.1132888, 5e-8) and near(K[3], 1.2350519, 5e-8) and near(r4, 1.1111105, 5e-8)
assert near(ex - r4, 6.2e-7, 5e-9)
assert near(1.055125 ** 2, 1.1132888, 5e-8) and near(1 + 0.1 * 1.1132888, 1.1113289, 5e-8)
assert near(1.2345679, 1.1111111 ** 2, 5e-7)
assert near((1 + 1.2345679) / 2, 1.1172840, 5e-7)
# 习题 3 型三级公式
k2 = 1.05 ** 2; k3 = (1 + 0.075 * k2) ** 2; y3 = 1 + 0.1 / 9 * (2 + 3 * k2 + 4 * k3)
assert near(k3, 1.1722122, 5e-8) and near(y3, 1.1110705, 5e-8) and near(ex - y3, 4.06e-5, 5e-8)
# Ralston 型
k2r = (1 + 0.2 / 3) ** 2; yr = 1 + 0.1 * (0.25 + 0.75 * k2r)
assert near(k2r, 1.1377778, 5e-8) and near(yr, 1.1103333, 5e-8) and near(ex - yr, 7.78e-4, 5e-7)

# 二阶条件 (7.2-6)：符号推导
h, c1, c2, a2, b21 = sp.symbols("h c1 c2 a2 b21")
X, Y = sp.symbols("X Y")
F = sp.Function("F")
fx = lambda x, y: x * y + y ** 2 + sp.sin(x)  # 任取光滑 f
x0, y0 = sp.Rational(1, 3), sp.Rational(1, 2)
f0 = fx(x0, y0)
K2 = fx(x0 + a2 * h, y0 + b21 * h * f0)
step = y0 + h * (c1 * f0 + c2 * K2)
# 准确解展开 y(x0+h)=y0+h f + h^2/2 (f_x+f f_y)
fxp = sp.diff(fx(X, Y), X).subs({X: x0, Y: y0}); fyp = sp.diff(fx(X, Y), Y).subs({X: x0, Y: y0})
exact2 = y0 + h * f0 + h ** 2 / 2 * (fxp + f0 * fyp)
diff = sp.series(step - exact2, h, 0, 3).removeO()
for sol in ({c1: sp.Rational(1, 2), c2: sp.Rational(1, 2), a2: 1, b21: 1}, {c1: 0, c2: 1, a2: sp.Rational(1, 2), b21: sp.Rational(1, 2)},
            {c1: sp.Rational(1, 4), c2: sp.Rational(3, 4), a2: sp.Rational(2, 3), b21: sp.Rational(2, 3)}):
    assert sp.simplify(diff.subs(sol)) == 0
# 习题 3 的三级公式为 3 阶：对 y'=y^2 比较到 h^3
hh = sp.symbols("hh")
yy = 1
k1 = yy ** 2; kk2 = (yy + hh / 2 * k1) ** 2; kk3 = (yy + sp.Rational(3, 4) * hh * kk2) ** 2
step3 = yy + hh / 9 * (2 * k1 + 3 * kk2 + 4 * kk3)
exactser = sp.series(1 / (1 - hh), hh, 0, 5).removeO()
d3 = sp.expand(sp.series(step3 - exactser, hh, 0, 5).removeO())
assert d3.coeff(hh, 1) == 0 and d3.coeff(hh, 2) == 0 and d3.coeff(hh, 3) == 0 and d3.coeff(hh, 4) != 0

# 例 1：表 7.2-1
fl = lambda x, y: x + y
x, y = 0.0, 1.0
tab = [(1, 1.2, 1.22, 1.444, 0.2428), (1.4428, 1.687080, 1.711508, 1.985102, 0.340836),
       (1.983636, 2.282000, 2.311836, 2.646003, 0.460577), (2.644213, 3.008634, 3.045076, 3.453228, 0.606829),
       (3.451042, 3.896146, 3.940657, 4.439173, 0.785461)]
for i in range(5):
    yn, KK = rk4(fl, x, y, 0.2)
    for v, w in zip(list(KK) + [yn - y], tab[i]):
        assert near(v, w, 1.5e-6)
    y = yn; x = round(x + 0.2, 10)
assert near(y, 3.436503, 1.5e-6)
assert near(2 * math.e - 2, 3.436564, 5e-7) and near(3.436564 - 3.436503, 6.1e-5, 1e-9)
assert near(2 * math.exp(0.2) - 1.2, 1.242806, 5e-7)

# 例 2
fs = lambda x, y: math.sinh(0.5 * y + x) / 1.5 + 0.5 * y
y1, K = rk4(fs, 0, 0, 0.1)
assert near(K[0], 0, 1e-15) and near(K[1], 0.033347, 5e-7) and near(K[3], 0.069679, 5e-7)
assert near(fs(0.05, 0.001667), 0.034737, 5e-7)
assert near(y1, 0.003431, 5e-7)
a, _ = rk4(fs, 0, 0, 0.05); b, _ = rk4(fs, 0.05, a, 0.05)
assert near(b, 0.003431, 5e-7)
c, _ = rk4(fs, 0.1, y1, 0.1); d, _ = rk4(fs, 0, 0, 0.2)
assert near(c, 0.014156, 5e-7) and near(d, 0.014155, 5e-7)
assert abs(b - y1) / 15 < 0.5e-5 / 16 and abs(c - d) / 15 < 0.5e-5
assert near(abs(b - y1) / 15, 2e-9, 1e-9)

# K6：Euler 外推
e1 = math.exp(-1); u, v = 0.9 ** 10, 0.95 ** 20
ext = 2 * v - u
assert near(ext, 0.3682934, 5e-8) and near(e1 - ext, -4.14e-4, 5e-7)
assert near(v - u, 0.0098075, 5e-8) and near(e1 - v, 0.0093935, 5e-8)

# K7：y'=x+y 在 0.2
p, _ = rk4(fl, 0, 1, 0.2); q1, _ = rk4(fl, 0, 1, 0.1); q, _ = rk4(fl, 0.1, q1, 0.1)
assert near(p, 1.2428, 1e-12) and near(q, 1.2428051, 5e-8)
assert near(q - p, 5.14e-6, 5e-9) and near((q - p) / 15, 3.43e-7, 5e-10)
assert near(q + (q - p) / 15, 1.2428055, 5e-8)
exv = 2 * math.exp(0.2) - 1.2
assert near(exv, 1.2428055, 5e-8) and near(exv - q, 3.7e-7, 5e-9) and near(exv - p, 5.5e-6, 5e-8)
assert near(16 * (q - p) / 15, 5.5e-6, 5e-8)
assert (q - p) / 15 < 1e-5 / 16 and 1e-6 / 16 <= (q - p) / 15 <= 1e-6
assert near(1e-5 / 16, 6.25e-7, 1e-15)

# 自测
assert near(1 + 0.1 * (-2 * (1 - 0.1)), 0.82, 1e-12) and near(math.exp(-0.2), 0.8187, 5e-5)
assert near((4 * 1.2370 - 1.2340) / 3, 1.2380, 1e-12) and near(0.0030 / 3, 0.0010, 1e-15)
print("ALL OK")
