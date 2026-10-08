"""N2.2 矩阵分解在解线性方程组中的应用。运行：python3 .work/kp/verify/N2.2.py"""
import sympy as sp

# 例 1：改进平方根法，六位对照
A = sp.Matrix([[5, -4, 1], [-4, 6, -4], [1, -4, 6]])
b = sp.Matrix([2, -1, -1])
assert A == A.T
assert all(A[:k, :k].det() > 0 for k in range(1, 4))

d1 = sp.Integer(5)
t21, t31 = -4, 1
l21, l31 = t21 / d1, t31 / d1
assert l21 == sp.Rational(-4, 5)
assert l31 == sp.Rational(1, 5)
d2 = 6 - t21 * l21
assert d2 == sp.Rational(14, 5)
t32 = -4 - t21 * l31
assert t32 == sp.Rational(-16, 5)
l32 = t32 / d2
assert sp.simplify(l32 - sp.Rational(-8, 7)) == 0
d3_exact = sp.simplify(6 - t31 * l31 - t32 * l32)
assert d3_exact == sp.Rational(15, 7)
# 书上先把 l32 收成六位 -1.14286，再代入 d3
l32_6 = -1.14286
d3_6 = 6 - 0.200000 - (-3.20000) * l32_6
assert abs(d3_6 - 2.14285) < 5e-6
assert abs(float(l32) - (-1.14286)) < 5e-6

y1 = sp.Integer(2)
y2 = -1 - l21 * y1
y3 = -1 - l31 * y1 - l32 * y2
assert y2 == sp.Rational(3, 5)
z1, z2, z3 = y1 / d1, y2 / d2, y3 / d3_exact
x3 = z3
x2 = z2 - l32 * x3
x1 = z1 - l21 * x2 - l31 * x3
# 准确解
x = A.solve(b)
assert sp.simplify(x - sp.Matrix([sp.Rational(1, 3), sp.Rational(-1, 6), sp.Rational(-1, 3)])) == sp.zeros(3, 1)
assert abs(float(x1) - float(x[0])) < 1e-9
assert abs(float(x2) - float(x[1])) < 1e-9
assert abs(float(x3) - float(x[2])) < 1e-9
# 书上六位
assert abs(float(y3) - (-0.714284)) < 5e-6
assert abs(float(z3) - (-0.333334)) < 5e-6
assert abs(float(x1) - 0.333333) < 5e-6
assert abs(float(x2) - (-0.166667)) < 5e-6

# 追赶法公式自洽：随机对角占优三对角
n = 5
bdiag = [4] * n
a = [0, 1, 1, 1, 1]
c = [1, 1, 1, 1, 0]
p = c[: n - 1]
r = [bdiag[0]]
ell = [0]
for i in range(1, n):
    ell.append(a[i] / r[i - 1])
    r.append(bdiag[i] - ell[i] * p[i - 1])
# 重构 A 并与 LR 比较
L = sp.eye(n)
R = sp.zeros(n)
for i in range(n):
    R[i, i] = r[i]
    if i < n - 1:
        R[i, i + 1] = p[i]
        L[i + 1, i] = ell[i + 1]
A3 = sp.zeros(n)
for i in range(n):
    A3[i, i] = bdiag[i]
    if i < n - 1:
        A3[i, i + 1] = c[i]
        A3[i + 1, i] = a[i + 1]
assert sp.simplify(L * R - A3) == sp.zeros(n)

# Doolittle 分解运算量与 Gauss 消元相同
nn, k = sp.symbols("n k", integer=True, positive=True)
mult = sp.summation((nn - k) * (nn - k + 2), (k, 1, nn - 1))
add = sp.summation((nn - k) * (nn - k + 1), (k, 1, nn - 1))
assert sp.simplify(mult - (nn**3 / 3 + nn**2 / 2 - sp.Rational(5, 6) * nn)) == 0
assert sp.simplify(add - (nn**3 / 3 - nn / 3)) == 0

# Householder 符号：d = -sign(a)*|s|，使 u 的对角分量不抵消
s0 = sp.Matrix([3, 4])
a11 = 3
sj = sp.sqrt(s0.dot(s0))
d = -sp.sign(a11) * sj
assert d == -5
u = s0 - d * sp.Matrix([1, 0])
assert u == sp.Matrix([8, 4])

print("ALL OK")
