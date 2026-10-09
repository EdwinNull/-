"""N6.3 Richardson 外推法和数值积分的 Romberg 算法。运行：python3 .work/kp/verify/N6.3.py"""
import math
from fractions import Fraction as Fr
import sympy as sp


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def T(f, a, b, n):
    h = (b - a) / n
    return h / 2 * (f(a) + f(b) + 2 * sum(f(a + i * h) for i in range(1, n)))


def romberg(f, a, b, K, cols=4):
    R = [[T(f, a, b, 2 ** k)] for k in range(K + 1)]
    for k in range(K + 1):
        for j in range(1, min(k, cols - 1) + 1):
            R[k].append(R[k][j - 1] + (R[k][j - 1] - R[k - 1][j - 1]) / (4 ** j - 1))
    return R


# 式 (6.3-2) 的系数：符号推导，a_k^(2) 含 a_k^(1)
h, p1, pk, a1, ak = sp.symbols("h p1 pk a1 ak", positive=True)
lhs_k = 2 ** p1 * ak * (h / 2) ** pk - ak * h ** pk
assert sp.simplify(lhs_k - ((sp.Rational(1, 2)) ** (pk - p1) - 1) * ak * h ** pk) == 0
lhs_1 = 2 ** p1 * a1 * (h / 2) ** p1 - a1 * h ** p1
assert sp.simplify(lhs_1) == 0

# 向前差商外推（p1=1）
D = lambda t: (math.exp(t) - 1) / t
near(D(0.1), 1.0517092, 5e-8)
near(D(0.05), 1.0254219, 5e-8)
ex = 2 * D(0.05) - D(0.1)
near(ex, 0.9991347, 5e-8)
near(abs(1 - D(0.05)), 2.54e-2, 5e-5)
near(abs(1 - ex), 8.65e-4, 5e-7)
# 展开系数 -h/2 - h^2/6
t = sp.symbols("t")
ser = sp.series(1 - (sp.exp(t) - 1) / t, t, 0, 3).removeO()
assert sp.simplify(ser - (-t / 2 - t ** 2 / 6)) == 0

# 例 1：π
S = lambda hh: math.sin(math.pi * hh) / hh
s = [S(1 / 4), S(1 / 8), S(1 / 16)]
near(s[0], 2 * math.sqrt(2), 1e-14)
for v, w in zip(s, (2.8284271, 3.0614675, 3.1214452)):
    near(v, w, 5e-8)
for v, w in zip(s, (0.313, 0.080, 0.020)):
    near(math.pi - v, w, 5e-4)
i2 = [s[1] + (s[1] - s[0]) / 3, s[2] + (s[2] - s[1]) / 3]
near(i2[0], 3.1391476, 5e-8); near(i2[1], 3.1414377, 5e-8)
i3 = i2[1] + (i2[1] - i2[0]) / 15
near(i3, 3.141590393, 5e-10)
near(math.pi - i3, 2.26e-6, 5e-9)
assert abs(i3 - 3.141591373) > 9e-7
# 先舍入七位再算
sr = [round(v, 7) for v in s]
i2r = [(4 * sr[1] - sr[0]) / 3, (4 * sr[2] - sr[1]) / 3]
near((16 * i2r[1] - i2r[0]) / 15, 3.1415904, 5e-8)
near(round(i2[1], 7) - round(i2[0], 7), 0.0022901, 1e-10)
near(3.1414377 + 0.0022901 / 15, 3.1415904, 5e-8)
# Taylor 展开
hh = sp.symbols("hh")
ser = sp.series(sp.sin(sp.pi * hh) / hh, hh, 0, 6).removeO()
assert sp.simplify(ser - (sp.pi - sp.pi ** 3 / 6 * hh ** 2 + sp.pi ** 5 / 120 * hh ** 4)) == 0

# 式 (6.3-5) 系数：用多项式核对 I-T_n 的 h^2、h^4 项
x = sp.symbols("x")
for poly in (x ** 4, x ** 5 + 2 * x ** 3, x ** 6):
    a_, b_ = 0, 1
    f = sp.lambdify(x, poly)
    I = float(sp.integrate(poly, (x, a_, b_)))
    d1 = sp.diff(poly, x); d3 = sp.diff(poly, x, 3); d5 = sp.diff(poly, x, 5)
    a2 = float((d1.subs(x, a_) - d1.subs(x, b_)) / 12)
    a4 = float(-(d3.subs(x, a_) - d3.subs(x, b_)) / 720)
    a6 = float((d5.subs(x, a_) - d5.subs(x, b_)) / 30240)
    for n in (2, 3, 5):
        hv = 1 / n
        near(I - T(f, 0, 1, n), a2 * hv ** 2 + a4 * hv ** 4 + a6 * hv ** 6, 1e-12)
# x^2 例
near(1 / 3 - T(lambda u: u * u, 0, 1, 2), -1 / 24, 1e-15)
near(-(1 / 6) * 0.25, -1 / 24, 1e-15)
near((4 * T(lambda u: u * u, 0, 1, 2) - T(lambda u: u * u, 0, 1, 1)) / 3, 1 / 3, 1e-15)

# 例 2：表 6.3-2
sinc = lambda u: 1.0 if u == 0 else math.sin(u) / u
R = romberg(sinc, 0, 1, 3)
table = [[0.9207355], [0.9397933, 0.9461459], [0.9445135, 0.9460869, 0.9460830],
         [0.9456909, 0.9460833, 0.9460831, 0.9460831]]
for row, trow in zip(R, table):
    for v, w in zip(row, trow):
        near(v, w, 5e-8)
near(0.9445135 + (0.9445135 - 0.9397933) / 3, 0.9460869, 5e-8)
near(0.9460869 + (0.9460869 - 0.9461459) / 15, 0.9460830, 5e-8)
# 自测 6
near(0.9460869 + (0.9460869 - 0.9461459) / 15, 0.9460830, 5e-8)

# 权的结构：I_2=S_n、I_3=C_n、I_4 权全正
def weights(K):
    N = 2 ** K
    def Tw(n):
        w = [Fr(0)] * (N + 1); step = N // n; hn = Fr(1, n)
        for i in range(n + 1):
            w[i * step] += hn / 2 * (1 if i in (0, n) else 2)
        return w
    W = [[Tw(2 ** k)] for k in range(K + 1)]
    for k in range(K + 1):
        for j in range(1, k + 1):
            W[k].append([W[k][j - 1][i] + (W[k][j - 1][i] - W[k - 1][j - 1][i]) / (4 ** j - 1) for i in range(N + 1)])
    return W
W1 = weights(1)[1][1]
assert W1 == [Fr(1, 6), Fr(4, 6), Fr(1, 6)]
W2 = weights(2)[2][2]
assert W2 == [Fr(7, 90), Fr(32, 90), Fr(12, 90), Fr(32, 90), Fr(7, 90)]
W3 = weights(3)[3][3]
assert all(w > 0 for w in W3) and sum(W3) == 1

# 自编：∫_1^2 dx/x
g = lambda u: 1 / u
L = math.log(2)
R2 = romberg(g, 1, 2, 4)
t2 = [[0.7500000], [0.7083333, 0.6944444], [0.6970238, 0.6932540, 0.6931746],
      [0.6941219, 0.6931545, 0.6931479, 0.6931475], [0.6933912, 0.6931477, 0.6931472, 0.6931472]]
for row, trow in zip(R2, t2):
    for v, w in zip(row, trow):
        near(v, w, 5e-8)
near(L, 0.69314718, 5e-9)
near(0.5 * (0.75 + 1 / 1.5), 0.7083333, 5e-8)
near(abs(L - R2[3][3]), 3.0e-7, 5e-9)
near(abs(L - R2[4][3]), 2.5e-9, 5e-11)
near(abs(R2[4][3] - R2[3][3]), 2.9e-7, 5e-9)
assert abs(R2[4][3] - R2[3][3]) <= 1e-6
near(abs(L - R2[4][0]), 2.4e-4, 5e-6)
near((R2[4][2] - R2[3][2]) / 63, -1.1e-8, 5e-10)
near(R2[4][2], 0.69314719, 5e-9); near(R2[3][2], 0.69314790, 5e-9)
err = [[abs(L - v) for v in row] for row in R2]
near(err[1][0] / err[2][0], 3.9, 0.05)
near(err[3][1] / err[4][1], 15.6, 0.1)
near(err[3][2] / err[4][2], 52.6, 0.5)
near(err[3][3] / err[4][3], 118, 1.5)
assert 2 ** 4 + 1 == 17

# 自测 3
near((4 * 0.7083333 - 0.75) / 3, 0.6944444, 5e-8)
print("ALL OK")
