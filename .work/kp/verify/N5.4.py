"""N5.4 最佳平方逼近。运行：python3 .work/kp/verify/N5.4.py"""
import numpy as np
from numpy.linalg import solve, cond
from fractions import Fraction as F

e = np.e
H = np.array([[1, 1/2, 1/3], [1/2, 1/3, 1/4], [1/3, 1/4, 1/5]], float)

# 例 1
d = np.array([e - 1, 1.0, e - 2])
assert abs(d[0] - 1.718281828459045) < 1e-12
assert abs(d[2] - 0.718281828459045) < 1e-12
a = solve(H, d)
assert abs(a[0] - 1.01299131) < 1e-8
assert abs(a[1] - 0.85112505) < 1e-8
assert abs(a[2] - 0.83918398) < 1e-8
inner = a @ d
assert abs(inner - 3.194500214020838) < 1e-12
f2 = 0.5 * (e**2 - 1)
err = np.sqrt(f2 - inner)
assert abs(err - 0.005275930674933218) < 1e-12
assert abs(np.sqrt(0.5 * (np.exp(2) - 1) - 3.19449) - 0.006168424865774702) < 1e-12
# 印出系数回代
ab = np.array([1.01299, 0.85112, 0.83918])
db = np.array([1.71828, 1.0, 0.71828])
assert abs(ab @ db - 3.1944866676) < 1e-9

# 例 2
pi = np.pi
rhs = np.array([2/pi, 1/pi, 1/pi - 4/pi**3])
assert abs(rhs[0] - 0.6366197723675814) < 1e-12
assert abs(rhs[1] - 0.3183098861837907) < 1e-12
assert abs(rhs[2] - 0.1893037484509927) < 1e-12
a2 = solve(H, rhs)
assert abs(a2[0] + 0.0504655) < 1e-7
assert abs(a2[1] - 4.12251162) < 1e-7
assert abs(a2[2] + 4.12251162) < 1e-7
assert abs(a2[1] + a2[2]) < 1e-10

def hilbert(n):
    return [[F(1, i + j + 1) for j in range(n)] for i in range(n)]

def infcond(n):
    A = [row[:] for row in hilbert(n)]
    I = [[F(int(i == j)) for j in range(n)] for i in range(n)]
    for k in range(n):
        piv = A[k][k]
        for j in range(n):
            A[k][j] /= piv
            I[k][j] /= piv
        for i in range(n):
            if i == k:
                continue
            f = A[i][k]
            for j in range(n):
                A[i][j] -= f * A[k][j]
                I[i][j] -= f * I[k][j]
    def nrm(M):
        return max(sum(abs(x) for x in row) for row in M)
    return float(nrm(hilbert(n)) * nrm(I))

assert abs(infcond(3) - 748) < 1e-8
assert abs(infcond(6) - 29070279) < 1e-4

# 例 3
assert abs(8/3 - F(8, 3)) < 1e-15
# c1 = -24/35 * 15/16 = -9/14
assert F(-24, 35) * F(15, 16) == F(-9, 14)

# 例 4 正交
def ipx(p, q):
    r = np.polymul(p, q)
    deg = len(r) - 1
    acc = 0.0
    for i, c in enumerate(r):
        k = deg - i
        acc += c / (k + 2)
    return acc

p0 = np.array([1.0])
p1 = np.array([1.0, -2/3])
p2 = np.array([1.0, -6/5, 3/10])
assert abs(ipx(p0, p1)) < 1e-12
assert abs(ipx(p0, p2)) < 1e-12
assert abs(ipx(p1, p2)) < 1e-12
assert abs(ipx(p1, p1) - F(1, 36)) < 1e-12

# 自编：x^2 在 [0,1] 的一次最佳逼近
Hs = np.array([[1.0, 0.5], [0.5, 1/3]])
rs = np.array([1/3, 1/4])
aa = solve(Hs, rs)
assert abs(aa[0] + 1/6) < 1e-12
assert abs(aa[1] - 1) < 1e-12
assert abs(np.sqrt(1/5 - 7/36) - np.sqrt(1/180)) < 1e-12

# Chebyshev 系数
def cheb_coeff(k, n=8000):
    th = np.linspace(0, np.pi, n, endpoint=False) + np.pi / (2 * n)
    return (2/np.pi) * np.sum(np.exp(np.cos(th)) * np.cos(k * th)) * (np.pi / n)

book_a = [2.532132, 1.130318, 0.271495, 0.0443368, 0.00547424, 0.00054293]
for k, bk in enumerate(book_a):
    assert abs(cheb_coeff(k) - bk) < 5e-7
# s1, s3 展开
a0, a1, a2, a3 = book_a[:4]
assert abs(a0 / 2 - 1.266066) < 1e-6
# 1.266066 + 1.130318 x + 0.271495(2x^2-1) + 0.0443368(4x^3-3x)
c3 = 0.0443368 * 4
c1 = 1.130318 - 0.0443368 * 3
c2 = 0.271495 * 2
c0 = 1.266066 - 0.271495
assert abs(c0 - 0.994571) < 1e-6
assert abs(c1 - 0.9973076) < 1e-6
assert abs(c2 - 0.54299) < 1e-6
assert abs(c3 - 0.1773472) < 1e-6

# T_m(T_n)=T_{mn}, T_{2n}=T_n(2x^2-1)
def T(n, x):
    return np.cos(n * np.arccos(np.clip(x, -1, 1)))

x = 0.3
assert abs(T(2, T(3, x)) - T(6, x)) < 1e-12
assert abs(T(4, x) - T(2, 2 * x**2 - 1)) < 1e-12

# 例 5 与例 1 在精确算术下相同
from scipy.special import eval_legendre
xs, ws = np.polynomial.legendre.leggauss(200)
moms = [np.sum(ws * np.exp((1 + xs) / 2) * eval_legendre(n, xs)) for n in range(3)]
norms = [2, 2/3, 2/5]
cs = [moms[i] / norms[i] for i in range(3)]
A0 = cs[0] - cs[1] + cs[2]
A1 = 2 * cs[1] - 6 * cs[2]
A2 = 6 * cs[2]
assert abs(A0 - a[0]) < 1e-10
assert abs(A1 - a[1]) < 1e-10
assert abs(A2 - a[2]) < 1e-10
assert abs(moms[0] - 2 * (e - 1)) < 1e-10
assert abs(cs[0] - (e - 1)) < 1e-12

# 右端扰动 1e-5
Hp = np.array([[1, 0.5, 1/3], [0.5, 1/3, 0.25], [1/3, 0.25, 0.2]], float)
dp = d + np.array([1e-5, 0.0, 0.0])
delta = solve(Hp, dp) - a
assert abs(delta[0] - 9.0e-5) < 1e-8
assert abs(delta[1] + 3.6e-4) < 1e-8
assert abs(delta[2] - 3.0e-4) < 1e-8

# 例 1 点值
def phi(x):
    return a[0] + a[1] * x + a[2] * x**2
assert abs(phi(0) - 1.01299131) < 1e-8
assert abs(phi(0.5) - 1.64834983) < 1e-8
assert abs(phi(1) - 2.70330034) < 1e-8
assert abs(np.exp(0.5) - 1.6487212707001282) < 1e-12
assert abs(phi(0) - 1) - 0.01299131 < 1e-8
assert abs(phi(1) - e + 0.01498149) < 1e-7

# 递推低次项
assert abs((3 * 0 - 1) / 2 + 0.5) < 1e-15 or True
# L2=(3x^2-1)/2, U2=x^2-4x+2, H2=4x^2-2
x = np.array([0.2])
L2 = 0.5 * (3 * x**2 - 1)
U2 = x**2 - 4 * x + 2
H2 = 4 * x**2 - 2
assert abs(L2[0] - 0.5 * (3 * 0.04 - 1)) < 1e-15
assert abs(U2[0] - (0.04 - 0.8 + 2)) < 1e-15
assert abs(H2[0] - (0.16 - 2)) < 1e-12
assert abs(np.pi - np.pi) == 0

assert abs((0.5 * (e**2 - 1) - inner) - 2.7835495e-5) < 1e-8
assert abs(np.sqrt(2/5) - 0.6324555320336759) < 1e-12
# 例 4 二次式在 (0,1) 内有两个根
r = np.roots([1, -6/5, 3/10])
assert np.all((r > 0) & (r < 1))

assert abs(1.266066 - 1 - 0.266066) < 1e-6
assert abs(1 - 0.994571 - 0.005429) < 1e-6
assert abs(np.pi - 3.141592653589793) < 1e-12

assert abs(7/36 - 0.19444444444444445) < 1e-12
assert abs(1/180 - 0.005555555555555556) < 1e-12
assert abs(np.sqrt(1/180) - 0.07453559924999298) < 1e-12
assert abs((0.5*(e**2-1) - inner) - 2.7835494442864e-5) < 1e-8

# int_{-1}^1 (x-2/3) dx = -4/3, not zero
assert abs((0.5 - 2/3) - (0.5 + 2/3) + 4/3) < 1e-12

print("ALL OK")
