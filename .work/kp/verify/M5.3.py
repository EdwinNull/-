"""M5.3 奇异值分解。运行：python3 .work/kp/verify/M5.3.py"""
import sympy as sp

# 书页 124 正文例
A = sp.Matrix([[1, 1], [1, -1], [2, 3]])
M = sp.simplify(A.T * A)
assert M == sp.Matrix([[6, 6], [6, 11]])
lam = sp.symbols("lam")
assert sp.expand((M - lam * sp.eye(2)).det()) == lam**2 - 17 * lam + 30
assert sp.roots((M - lam * sp.eye(2)).det(), lam) == {15: 1, 2: 1}
assert sp.simplify(sp.sqrt(15) ** 2 - 15) == 0

# 例 2
s = 1 / sp.sqrt(2)
B = sp.Matrix([[s, 0, s], [s, 1, -s], [-s, 1, s]])
assert sp.simplify(B.T * B) == sp.Matrix([[sp.Rational(3, 2), 0, sp.Rational(-1, 2)], [0, 2, 0], [sp.Rational(-1, 2), 0, sp.Rational(3, 2)]])
assert (B.T * B).eigenvals() == {2: 2, 1: 1}
q1 = sp.Matrix([-s, 0, s])
q2 = sp.Matrix([0, 1, 0])
q3 = sp.Matrix([s, 0, s])
Q2 = sp.Matrix.hstack(q1, q2, q3)
assert sp.simplify(Q2.T * Q2) == sp.eye(3)
Sigma = sp.diag(sp.sqrt(2), sp.sqrt(2), 1)
Q1 = sp.simplify(B * Q2 * Sigma.inv())
assert sp.simplify(Q1.T * Q1) == sp.eye(3)
assert sp.simplify(Q1.T * B * Q2 - Sigma) == sp.zeros(3)
assert Q1 == sp.Matrix([[0, 0, 1], [-s, s, 0], [s, s, 0]])

# 例 3
C = sp.Matrix([[1, 0, -1], [1, 0, -1]])
assert C.T * C == sp.Matrix([[2, 0, -2], [0, 0, 0], [-2, 0, 2]])
assert (C.T * C).eigenvals() == {4: 1, 0: 2}
v1 = sp.Matrix([1, 0, -1]) / sp.sqrt(2)
v2 = sp.Matrix([0, 1, 0])
v3 = sp.Matrix([1, 0, 1]) / sp.sqrt(2)
V = sp.Matrix.hstack(v1, v2, v3)
U1 = sp.simplify(C * v1 / 2)
assert U1 == sp.Matrix([s, s])
U = sp.Matrix.hstack(U1, sp.Matrix([s, -s]))
assert sp.simplify(U.T * U) == sp.eye(2)
S = sp.simplify(U.T * C * V)
assert S == sp.Matrix([[2, 0, 0], [0, 0, 0]])
assert sp.simplify(C.norm()) == 2  # Frobenius via entries: sqrt(4)=2
assert abs(float(sp.Matrix(C).norm()) - 2) < 1e-12

# 2-范数等于最大奇异值：例 3
assert sp.sqrt(max((C.T * C).eigenvals())) == 2

print("ALL OK")
