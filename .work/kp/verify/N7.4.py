"""N7.4 预估-校正公式。运行：python3 .work/kp/verify/N7.4.py"""
import math
from fractions import Fraction as Fr


def near(a, b, tol):
    return abs(a - b) <= tol


def rk4(f, x, y, h):
    k1 = f(x, y); k2 = f(x + h / 2, y + h / 2 * k1); k3 = f(x + h / 2, y + h / 2 * k2); k4 = f(x + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# 补偿系数
def comp(Cp, Cc):
    return Cp / (Cp - Cc), Cc / (Cp - Cc)


assert comp(Fr(251, 720), Fr(-19, 720)) == (Fr(251, 270), Fr(-19, 270))
assert Fr(251, 720) - Fr(-19, 720) == Fr(270, 720)
assert comp(Fr(14, 45), Fr(-1, 40)) == (Fr(112, 121), Fr(-9, 121))
assert Fr(14, 45) - Fr(-1, 40) == Fr(121, 360)
assert comp(Fr(1, 3), Fr(-1, 12)) == (Fr(4, 5), Fr(-1, 5))
assert comp(Fr(3, 8), Fr(-1, 24)) == (Fr(9, 10), Fr(-1, 10))
assert Fr(251, 270) + Fr(19, 270) == 1
assert near(251 / 720, 0.349, 5e-4) and near(14 / 45, 0.311, 5e-4) and near(19 / 720, 0.026, 5e-4)

# y'=-y：Adams 对，补偿与否
h = 0.1


def adams(compd, N=10):
    y = [math.exp(-h * k) for k in range(4)]; f = [-v for v in y]; cp = 0.0; rec = []
    for i in range(3, N):
        p = y[i] + h / 24 * (55 * f[i] - 59 * f[i - 1] + 37 * f[i - 2] - 9 * f[i - 3])
        m = p + 251 / 270 * cp if compd else p
        c = y[i] + h / 24 * (9 * (-m) + 19 * f[i] - 5 * f[i - 1] + f[i - 2])
        yn = c - 19 / 270 * (c - p) if compd else c
        cp = c - p; y.append(yn); f.append(-yn); rec.append((p, m, c, yn))
    return y, rec


yc, rc = adams(True); yp, rp = adams(False)
assert near(rc[0][0], 0.6703229, 5e-8) and near(rc[0][2], 0.6703197, 5e-8) and near(rc[0][3], 0.6703200, 5e-8)
assert near(math.exp(-0.4) - rc[0][3], 8.5e-8, 5e-9) and near(math.exp(-0.4) - rp[0][3], 3.1e-7, 5e-9)
assert near(math.exp(-1) - yc[10], -3.6e-8, 5e-9) and near(math.exp(-1) - yp[10], 1.17e-6, 5e-9)
assert near((math.exp(-1) - yp[10]) / abs(math.exp(-1) - yc[10]), 33, 3)

# Euler 两步 + 梯形
def pair(compd):
    y = [1, math.exp(-0.1)]; cp = 0.0; rec = []
    for i in range(1, 10):
        p = y[i - 1] + 2 * h * (-y[i]); m = p + 0.8 * cp if compd else p
        c = y[i] + h / 2 * (-m - y[i]); yn = c - 0.2 * (c - p) if compd else c
        cp = c - p; y.append(yn); rec.append((p, m, c, yn))
    return y, rec


y1, r1 = pair(False); y2, r2 = pair(True)
assert near(r1[0][0], 0.8190325, 5e-8) and near(r1[0][2], 0.8186439, 5e-8)
assert near(math.exp(-0.2) - r1[0][2], 8.7e-5, 5e-7) and near(r2[0][3], 0.8187216, 5e-8)
assert near(math.exp(-0.2) - r2[0][3], 9.1e-6, 5e-8)
assert near(y1[10], 0.3675083, 5e-8) and near(math.exp(-1) - y1[10], 3.71e-4, 5e-7)
assert near(y2[10], 0.3678884, 5e-8) and near(math.exp(-1) - y2[10], -9.0e-6, 5e-8)
assert near(math.exp(-1), 0.3678794, 5e-8)
assert near((math.exp(-1) - y1[10]) / abs(math.exp(-1) - y2[10]), 41, 2)

# 例 1：表 7.4-1（6 位舍入）
F = lambda x, y: y - 2 * x / y
r = lambda v: round(v, 6)
ys = [1.0]; xs = [0.0]
for i in range(3):
    ys.append(rk4(F, xs[-1], ys[-1], h)); xs.append(xs[-1] + h)
for v, w in zip(ys, (1, 1.095446, 1.183217, 1.264912)):
    assert near(v, w, 5e-7)
ys = [1, 1.095446, 1.183217, 1.264912]; xs = [0, 0.1, 0.2, 0.3]
fs = [r(F(x, y)) for x, y in zip(xs, ys)]
assert fs[1:] == [0.912872, 0.845156, 0.790571]
cp = 0.0; out = []
for i in (3, 4):
    p = r(ys[i] + h / 24 * (55 * fs[i] - 59 * fs[i - 1] + 37 * fs[i - 2] - 9 * fs[i - 3])); m = r(p + 251 / 270 * cp)
    c = r(ys[i] + h / 24 * (9 * F(xs[i] + h, m) + 19 * fs[i] - 5 * fs[i - 1] + fs[i - 2])); yn = r(c - 19 / 270 * (c - p))
    out.append((p, m, c, yn)); cp = c - p; ys.append(yn); xs.append(xs[i] + h); fs.append(r(F(xs[-1], yn)))
assert out[0] == (1.341551, 1.341551, 1.341641, 1.341635)
assert out[1] == (1.414157, 1.414241, 1.414211, 1.414207)
assert fs[4] == 0.745348
assert near(h / 24 * (55 * 0.745348 - 59 * 0.790571 + 37 * 0.845156 - 9 * 0.912872), 0.072522, 5e-7)
assert near(math.sqrt(2) - 1.414207, 7e-6, 5e-7) and near(math.sqrt(2) - 1.414192, 2.2e-5, 5e-7)
assert near(1.414123 - 1.414039, 251 / 270 * 0.000090, 2e-6)
assert near(1.341641 - 19 / 270 * 0.000090, 1.341635, 5e-7)
for k, v in enumerate((1, 1.095445, 1.183216, 1.264911)):
    assert near(math.sqrt(1 + 0.2 * k), v, 5e-7)

# 线性解检验 Hamming 校正式
xi, hh = 0.7, 0.1
good = (9 * xi - (xi - 2 * hh)) / 8 + 3 / 8 * hh * (1 + 2 - 1)
bad = (9 * xi - (xi - 3 * hh)) / 8 + 3 / 8 * (1 + 2 - 1)
assert near(good, xi + hh, 1e-12) and abs(bad - (xi + hh)) > 0.1
assert near((9 * xi - (xi - 2 * hh)) / 8 + 3 / 8 * (1 + 2 - 1), xi + hh / 4 + 0.75, 1e-12)

# 自测 3
assert near(2.000030 - 19 / 270 * (2.000030 - 2.000300), 2.000049, 1e-9)
print("ALL OK")
