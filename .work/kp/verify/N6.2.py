"""N6.2 复化求积公式及其余项表达式。运行：python3 .work/kp/verify/N6.2.py"""
import math
import numpy as np
from scipy.integrate import quad


def T(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    return h / 2 * (f(x[0]) + f(x[-1]) + 2 * sum(f(t) for t in x[1:-1]))


def S(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    mids = x[:-1] + h / 2
    return h / 6 * (f(a) + f(b) + 2 * sum(f(t) for t in x[1:-1]) + 4 * sum(f(t) for t in mids))


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


# K1、K3：x^2 的 T_2 与余项
sq = lambda x: x * x
near(T(sq, 0, 1, 2), 0.375, 1e-15)
near(1 / 3 - 0.375, -1 / 24, 1e-15)
near(-(1 / 12) * 0.25 * 2, -1 / 24, 1e-15)

# 例 1 函数表与递推
sinc = lambda x: 1.0 if x == 0 else math.sin(x) / x
xs = [i / 8 for i in range(9)]
tab = [1.0000000, 0.9973979, 0.9896158, 0.9767267, 0.9588511, 0.9361556, 0.9088517, 0.8771926, 0.8414710]
for x, v in zip(xs, tab):
    near(sinc(x), v, 5e-8)
near(sinc(0.125), 0.99739787, 5e-9)          # 印出 0.9973987 为数字颠倒
assert math.floor(sinc(0.125) * 1e7) / 1e7 == 0.9973978
I = quad(sinc, 0, 1, epsabs=1e-14)[0]
near(I, 0.9460831, 1e-7)
T1 = T(sinc, 0, 1, 1); T2 = T(sinc, 0, 1, 2); T4 = T(sinc, 0, 1, 4); T8 = T(sinc, 0, 1, 8)
near(T1, 0.9207355, 5e-8)
H1 = sinc(0.5); H2 = 0.5 * (sinc(0.25) + sinc(0.75))
H4 = 0.25 * sum(sinc(t) for t in (0.125, 0.375, 0.625, 0.875))
near(H1, 0.9588511, 5e-8); near(H2, 0.9492338, 5e-8); near(H4, 0.9468682, 5e-8)
near((T1 + H1) / 2, T2, 1e-15); near((T2 + H2) / 2, T4, 1e-15); near((T4 + H4) / 2, T8, 1e-15)
near(T2, 0.9397933, 5e-8); near(T4, 0.9445135, 5e-8); near(T8, 0.9456909, 5e-8)
S1 = S(sinc, 0, 1, 1); S2 = S(sinc, 0, 1, 2); S4 = S(sinc, 0, 1, 4)
near(S1, 0.9461459, 5e-8); near(S2, 0.94608693, 5e-9); near(S4, 0.94608331, 5e-9)
# 由舍入后的 T 算 S
r = lambda v: round(v, 7)
near((4 * r(T2) - r(T1)) / 3, 0.9461459, 5e-8)
near((4 * r(T4) - r(T2)) / 3, 0.9460869, 5e-8)
near((4 * r(T8) - r(T4)) / 3, 0.9460834, 5e-8)
for n in (1, 2, 4):
    near((4 * T(sinc, 0, 1, 2 * n) - T(sinc, 0, 1, n)) / 3, S(sinc, 0, 1, n), 1e-14)
near(r(T8) - r(T4), 0.0011774, 1e-10)
near((r(T8) - r(T4)) / 3, 0.0003925, 5e-8)
near(r(I) - r(T8), 0.0003922, 1e-10)
near((I - T8) / (I - T4), 0.250, 5e-4)
near(0.0003922 / 0.0015696, 0.250, 5e-4)
near(r(I) - r(T4), 0.0015696, 1e-10)
# 自测 3
near(0.9456909 + (0.9456909 - 0.9445135) / 3, 0.9460834, 5e-8)

# 例 2：导数上界与步长
near(math.sqrt(18e-3), 0.1342, 5e-5)
assert 0.125 <= math.sqrt(18e-3)
near(1 / 180 * 0.125 ** 4 / 5, 0.271e-6, 5e-10)
for k in range(1, 6):
    g = lambda x, k=k: quad(lambda t: t ** k * math.cos(t * x + k * math.pi / 2), 0, 1)[0]
    assert max(abs(g(x)) for x in np.linspace(0, 1, 21)) <= 1 / (k + 1) + 1e-12

# 例 3
f = lambda x: math.exp(1 / x)
f4max = 73 * math.e
near(f4max, 198.4346, 5e-5)
hmax = (0.5e-3 * 180 * 16 / 198.4346) ** 0.25
near(0.5e-3 * 180 * 16 / 198.4346, 0.0072568, 5e-8)
near(hmax, 0.2919, 5e-5)
near(1 / 180 * 0.25 ** 4 / 16 * 198.4346, 2.69e-4, 5e-7)
near(1 / 180 * (1 / 3) ** 4 / 16 * 198.4346, 8.5e-4, 5e-6)
assert 1 / 3 > hmax and 0.25 < hmax
S4e = S(f, 1, 2, 4); Ie = quad(f, 1, 2)[0]
near(S4e, 2.0201022, 5e-8); near(Ie, 2.0200586, 5e-8)
near(S4e - Ie, 4.36e-5, 5e-8)

# 例 4
near((0.9461459 - 0.9460868) / 15, 0.394e-5, 5e-9)
near((0.9460868 - 0.9460832) / 15, 0.24e-6, 5e-9)
assert (0.9461459 - 0.9460868) / 15 > 0.5e-6 > (0.9460868 - 0.9460832) / 15

# K5：e^x 的 (I-T_n)/h^2
E = math.e - 1
near((1 - math.e) / 12, -0.143190, 5e-7)
for n, v in zip((1, 2, 4, 8, 16), (-0.140859, -0.142597, -0.143041, -0.143153, -0.143181)):
    near((E - T(math.exp, 0, 1, n)) * n * n, v, 5e-7)

# K6 自编、例题 3
near(math.sqrt(12e-3 / math.e), 0.06644, 5e-6)
assert 1 / 16 <= math.sqrt(12e-3 / math.e) < 1 / 15
near(E - T(math.exp, 0, 1, 16), -5.59e-4, 5e-7)
Ts = {n: T(math.exp, 0, 1, n) for n in (1, 2, 4, 8, 16, 32)}
for n, v in zip((1, 2, 4, 8, 16, 32), (1.8591409, 1.7539311, 1.7272219, 1.7205186, 1.7188411, 1.7184217)):
    near(Ts[n], v, 5e-8)
for n, v in zip((2, 4, 8, 16, 32), (0.1052098, 0.0267092, 0.0067033, 0.0016775, 0.0004195)):
    near(abs(Ts[n] - Ts[n // 2]), v, 5e-8)
for n, v in zip((1, 2, 4, 8, 16, 32), (-0.1408591, -0.0356493, -0.0089401, -0.0022368, -0.0005593, -0.0001398)):
    near(E - Ts[n], v, 5e-8)
assert abs(Ts[16] - Ts[8]) > 1e-3 >= abs(Ts[32] - Ts[16])
near(E, 1.7182818, 5e-8)
near(Ts[32], 1.718422, 5e-7)
corr = Ts[32] + (Ts[32] - Ts[16]) / 3
near(corr, 1.71828184, 5e-9)
near(abs(corr - E), 9e-9, 5e-9)
# 题中 n=1,2,4,8,16 的单步差（K8 例）
near(abs(Ts[2] - Ts[1]), 0.105210, 5e-7); near(abs(Ts[4] - Ts[2]), 0.026709, 5e-7)
near(abs(Ts[8] - Ts[4]), 0.006703, 5e-7); near(abs(Ts[16] - Ts[8]), 0.001677, 5e-7)
near(abs(Ts[32] - Ts[16]), 0.000419, 5e-7)

# 易错 5：x^2(1-x)^2，f'(0)=f'(1)
q = lambda x: x * x * (1 - x) ** 2
Iq = 1 / 30
e2 = Iq - T(q, 0, 1, 2); e4 = Iq - T(q, 0, 1, 4)
near(e2, 2.08e-3, 5e-6); near(e4, 1.30e-4, 5e-7)
near(e4 / e2, 1 / 16, 1e-3)

# K9：x^3 的样条求积
xs3 = [0, 0.5, 1]; fs = [t ** 3 for t in xs3]; M = [6 * t for t in xs3]; h = 0.5
val = sum(h / 2 * (fs[i] + fs[i + 1]) - h ** 3 / 24 * (M[i] + M[i + 1]) for i in range(2))
near(val, 0.25, 1e-15)
near(sum(h / 2 * (fs[i] + fs[i + 1]) for i in range(2)), 0.3125, 1e-15)

# 自测 4
near(math.sqrt(0.06 / math.e ** 2), 0.0901, 5e-5)
# 系数和
n = 5
assert 2 * n + 1 == 11
print("ALL OK")
