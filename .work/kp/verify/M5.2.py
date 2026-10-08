"""M5.2 方阵的正交（酉）三角分解。运行：python3 .work/kp/verify/M5.2.py"""
import sympy as sp

# 书页 117 的三阶例子
A = sp.Matrix([[1, 2, 2], [2, 1, 2], [1, 2, 1]])
Q = sp.Matrix([
    [1 / sp.sqrt(6), 1 / sp.sqrt(3), 1 / sp.sqrt(2)],
    [2 / sp.sqrt(6), -1 / sp.sqrt(3), 0],
    [1 / sp.sqrt(6), 1 / sp.sqrt(3), -1 / sp.sqrt(2)],
])
R = sp.Matrix([
    [sp.sqrt(6), sp.sqrt(6), 7 / sp.sqrt(6)],
    [0, sp.sqrt(3), 1 / sp.sqrt(3)],
    [0, 0, 1 / sp.sqrt(2)],
])
assert sp.simplify(Q * R - A) == sp.zeros(3)
assert sp.simplify(Q.T * Q) == sp.eye(3)
assert sp.simplify(Q[:, 0].dot(A[:, 0]) - sp.sqrt(6)) == 0

# 例 1
A1 = sp.Matrix([
    [1, sp.Rational(1, 2), 5],
    [1, sp.Rational(-1, 2), 2],
    [-1, sp.Rational(1, 2), -2],
    [1, sp.Rational(-3, 2), 0],
])
Q1 = sp.Matrix([
    [sp.Rational(1, 2), sp.sqrt(2) / 2, sp.Rational(1, 2)],
    [sp.Rational(1, 2), 0, sp.Rational(-1, 2)],
    [sp.Rational(-1, 2), 0, sp.Rational(1, 2)],
    [sp.Rational(1, 2), -sp.sqrt(2) / 2, sp.Rational(1, 2)],
])
R1 = sp.Matrix([
    [2, -1, sp.Rational(9, 2)],
    [0, sp.sqrt(2), 5 * sp.sqrt(2) / 2],
    [0, 0, sp.Rational(1, 2)],
])
assert sp.simplify(Q1 * R1 - A1) == sp.zeros(4, 3)
assert sp.simplify(Q1.T * Q1) == sp.eye(3)
assert A1[:, 0].norm() == 2

# 例 2
x = sp.Matrix([0, 3, 0, 4])
y = sp.Matrix([5, 0, 0, 0])
d = x - y
H = sp.simplify(sp.eye(4) - 2 * d * d.T / d.dot(d))
assert H[0, 0] == 0 and H[0, 1] == sp.Rational(3, 5) and H[0, 3] == sp.Rational(4, 5)
assert H[1, 1] == sp.Rational(16, 25) and H[1, 3] == sp.Rational(-12, 25)
assert H[3, 3] == sp.Rational(9, 25)
assert sp.simplify(H * x - y) == sp.zeros(4, 1)
assert sp.simplify(H.T * H) == sp.eye(4)
assert H[1, 3] != sp.Rational(-12, 15)

# 例 3
A3 = sp.Matrix([[2, 2, 1], [1, 2, 2], [2, 1, 2]])
d1 = A3[:, 0] - sp.Matrix([-3, 0, 0])
H1 = sp.simplify(sp.eye(3) - 2 * d1 * d1.T / d1.dot(d1))
HA = sp.simplify(H1 * A3)
assert HA[:, 0] == sp.Matrix([-3, 0, 0])
assert HA[1, 1] == sp.Rational(16, 15) and HA[2, 1] == sp.Rational(-13, 15)
x2 = sp.Matrix([sp.Rational(16, 15), sp.Rational(-13, 15)])
y2 = sp.Matrix([-sp.sqrt(17) / 3, 0])
dd = x2 - y2
Q2 = sp.simplify(sp.eye(2) - 2 * dd * dd.T / dd.dot(dd))
H2 = sp.diag(1, Q2)
R3 = sp.simplify(H2 * HA)
Q3 = sp.simplify(H1 * H2)
assert sp.simplify(Q3 * R3 - A3) == sp.zeros(3)
assert sp.simplify(R3[1, 2] + 8 * sp.sqrt(17) / 51) == 0
assert sp.simplify(R3[2, 2] - 5 * sp.sqrt(17) / 17) == 0
assert sp.simplify(Q3[0, 0] + sp.Rational(2, 3)) == 0
assert sp.simplify(Q3[1, 1] + 10 * sp.sqrt(17) / 51) == 0

Dsign = sp.diag(1, 1, -1)
assert sp.simplify((Q * Dsign) * (Dsign * R) - A) == sp.zeros(3)
assert (Dsign * R)[2, 2] == -1 / sp.sqrt(2)

bad = sp.Matrix([2, 1, 2]) - sp.Matrix([3, 0, 0])
good = sp.Matrix([2, 1, 2]) - sp.Matrix([-3, 0, 0])
assert bad.dot(bad) == 6 and good.dot(good) == 30

print("ALL OK")
