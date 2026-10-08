"""M4.2 方阵函数及其计算 —— 讲解中全部数值的复算脚本。
运行：python3 .work/kp/verify/M4.2.py
"""
import numpy as np
import sympy as sp

lam = sp.symbols("lam")
t = sp.symbols("t", real=True, positive=True)

# ---------- 4.2 节例 1 ----------
A = sp.Matrix([[0, 1], [0, -2]])
assert sp.factor(A.charpoly(lam).as_expr()) == lam * (lam + 2)
P = sp.Matrix([[1, 1], [0, -2]])
assert P.inv() == sp.Matrix([[1, sp.Rational(1, 2)], [0, -sp.Rational(1, 2)]])
assert A * P == P * sp.diag(0, -2)
eA = P * sp.diag(1, sp.exp(-2)) * P.inv()
assert sp.simplify(eA - sp.Matrix([[1, (1 - sp.exp(-2)) / 2], [0, sp.exp(-2)]])) == sp.zeros(2)
cA = P * sp.diag(1, sp.cos(2)) * P.inv()
assert sp.simplify(cA - sp.Matrix([[1, (1 - sp.cos(2)) / 2], [0, sp.cos(2)]])) == sp.zeros(2)
sA = P * sp.diag(0, -sp.sin(2)) * P.inv()
assert sp.simplify(sA - sp.Matrix([[0, sp.sin(2) / 2], [0, -sp.sin(2)]])) == sp.zeros(2)
printed = sp.Matrix([[1, sp.sin(2) / 2], [0, -sp.sin(2)]])
assert sp.simplify(sA - printed) != sp.zeros(2)

# ---------- 4.2 节例 2 ----------
A2 = sp.Matrix([[0, 1, 0], [0, 0, 1], [2, 3, 0]])
assert sp.factor(A2.charpoly(lam).as_expr()) == (lam - 2) * (lam + 1) ** 2
assert sp.expand((A2 - lam * sp.eye(3)).det() - (2 - lam) * (1 + lam) ** 2) == 0
x1 = sp.Matrix([1, 2, 4])
x2 = sp.Matrix([1, -1, 1])
x3 = sp.Matrix([1, 0, -1])
assert A2 * x1 == 2 * x1
assert (A2 + sp.eye(3)) * x2 == sp.zeros(3, 1)
assert (A2 + sp.eye(3)) * x3 == x2
assert (A2 + sp.eye(3)).rank() == 2
P2 = sp.Matrix([[1, 1, 1], [2, -1, 0], [4, 1, -1]])
assert P2.det() == 9
P2inv = sp.Rational(1, 9) * sp.Matrix([[1, 2, 1], [2, -5, 2], [6, 3, -3]])
assert sp.simplify(P2 * P2inv) == sp.eye(3)
J2 = sp.Matrix([[2, 0, 0], [0, -1, 1], [0, 0, -1]])
assert sp.simplify(A2 * P2 - P2 * J2) == sp.zeros(3)
eJ = sp.Matrix([[sp.exp(2), 0, 0], [0, sp.exp(-1), sp.exp(-1)], [0, 0, sp.exp(-1)]])
eA2 = sp.simplify(P2 * eJ * P2inv)
tb_e = sp.Rational(1, 9) * sp.Matrix([
    [sp.exp(2) + 14 * sp.exp(-1), 2 * sp.exp(2) + sp.exp(-1), sp.exp(2) - 4 * sp.exp(-1)],
    [2 * sp.exp(2) - 8 * sp.exp(-1), 4 * sp.exp(2) + 2 * sp.exp(-1), 2 * sp.exp(2) + sp.exp(-1)],
    [4 * sp.exp(2) + 2 * sp.exp(-1), 8 * sp.exp(2) - 5 * sp.exp(-1), 4 * sp.exp(2) + 2 * sp.exp(-1)],
])
assert sp.simplify(eA2 - tb_e) == sp.zeros(3)
eJt = sp.Matrix([
    [sp.exp(2 * t), 0, 0],
    [0, sp.exp(-t), t * sp.exp(-t)],
    [0, 0, sp.exp(-t)],
])
eAt2 = sp.simplify(P2 * eJt * P2inv)
tb_et = sp.Rational(1, 9) * sp.Matrix([
    [sp.exp(2 * t) + (8 + 6 * t) * sp.exp(-t),
     2 * sp.exp(2 * t) - (2 - 3 * t) * sp.exp(-t),
     sp.exp(2 * t) - (1 + 3 * t) * sp.exp(-t)],
    [2 * sp.exp(2 * t) - (2 + 6 * t) * sp.exp(-t),
     4 * sp.exp(2 * t) + (5 - 3 * t) * sp.exp(-t),
     2 * sp.exp(2 * t) - (2 - 3 * t) * sp.exp(-t)],
    [4 * sp.exp(2 * t) - (4 - 6 * t) * sp.exp(-t),
     8 * sp.exp(2 * t) - (8 - 3 * t) * sp.exp(-t),
     4 * sp.exp(2 * t) + (5 - 3 * t) * sp.exp(-t)],
])
assert sp.simplify(eAt2 - tb_et) == sp.zeros(3)
assert sp.simplify(tb_et.subs(t, 0)) == sp.eye(3)
assert (1 + 8) / 9 == 1 and (2 - 2) / 9 == 0 and (4 + 5) / 9 == 1 and (4 - 4) / 9 == 0

# ---------- 4.2 节例 3，两种方法 ----------
A3 = sp.Matrix([[2, -1], [1, 0]])
assert sp.factor(A3.charpoly(lam).as_expr()) == (lam - 1) ** 2
assert (A3 - sp.eye(2)) ** 2 == sp.zeros(2)
assert A3 - sp.eye(2) != sp.zeros(2)
v1 = sp.Matrix([1, 1])
v2 = sp.Matrix([1, 0])
assert (A3 - sp.eye(2)) * v1 == sp.zeros(2, 1)
assert (A3 - sp.eye(2)) * v2 == v1
P3 = sp.Matrix([[1, 1], [1, 0]])
assert P3.det() == -1
assert P3.inv() == sp.Matrix([[0, 1], [1, -1]])
J3 = sp.Matrix([[1, 1], [0, 1]])
fJ = sp.Matrix([[sp.E, sp.E], [0, sp.E]])
assert sp.simplify(P3 * fJ * P3.inv()) == sp.E * A3
assert sp.simplify(A3.exp() - sp.E * A3) == sp.zeros(2)
assert sp.simplify(sp.trace(sp.E * A3) - 2 * sp.E) == 0
b0 = t * sp.exp(t)
b1 = (1 - t) * sp.exp(t)
assert sp.simplify(b0 * A3 + b1 * sp.eye(2) - (t * A3).exp()) == sp.zeros(2)
tb3 = sp.Matrix([[(1 + t) * sp.exp(t), -t * sp.exp(t)], [t * sp.exp(t), (1 - t) * sp.exp(t)]])
assert sp.simplify((t * A3).exp() - tb3) == sp.zeros(2)

# ---------- 4.2 节例 4 ----------
A4 = sp.Matrix([[17, 0, -25], [0, 3, 0], [9, 0, -13]])
assert sp.factor(A4.charpoly(lam).as_expr()) == (lam - 3) * (lam - 2) ** 2
b0 = sp.exp(3 * t) - (1 + t) * sp.exp(2 * t)
b1 = -4 * sp.exp(3 * t) + (4 + 5 * t) * sp.exp(2 * t)
b2 = 4 * sp.exp(3 * t) - 3 * (1 + 2 * t) * sp.exp(2 * t)
g = sp.simplify(b0 * A4 ** 2 + b1 * A4 + b2 * sp.eye(3))
tb4 = sp.Matrix([
    [(1 + 15 * t) * sp.exp(2 * t), 0, -25 * t * sp.exp(2 * t)],
    [0, sp.exp(3 * t), 0],
    [9 * t * sp.exp(2 * t), 0, (1 - 15 * t) * sp.exp(2 * t)],
])
assert sp.simplify(g - tb4) == sp.zeros(3)
assert sp.simplify(g - (t * A4).exp()) == sp.zeros(3)

# ---------- 4.2 节例 5 ----------
A5 = sp.Matrix([[5, -4, 4], [1, 0, 5], [1, -1, 6]])
assert sp.factor(A5.charpoly(lam).as_expr()) == (lam - 1) * (lam - 5) ** 2
assert (A5 - 5 * sp.eye(3)).rank() == 2
assert (A5 - sp.eye(3)).rank() == 2
assert sp.simplify((A5 - sp.eye(3)) * (A5 - 5 * sp.eye(3)) ** 2) == sp.zeros(3)
assert (A5 - 5 * sp.eye(3)) ** 2 != sp.zeros(3)
assert (A5 - sp.eye(3)) * (A5 - 5 * sp.eye(3)) != sp.zeros(3)
P5 = sp.Matrix([[1, 0, 1], [1, 1, 0], [0, 1, 0]])
assert P5.det() == 1
assert P5.inv() == sp.Matrix([[0, 1, -1], [0, 0, 1], [1, -1, 1]])
J5 = sp.Matrix([[1, 0, 0], [0, 5, 1], [0, 0, 5]])
assert A5 * P5 == P5 * J5
lnJ = sp.Matrix([[0, 0, 0], [0, sp.log(5), sp.Rational(1, 5)], [0, 0, sp.log(5)]])
lnA = sp.simplify(P5 * lnJ * P5.inv())
tb5 = sp.Matrix([
    [sp.log(5), -sp.log(5), sp.log(5)],
    [sp.Rational(1, 5), -sp.Rational(1, 5), sp.Rational(1, 5) + sp.log(5)],
    [sp.Rational(1, 5), -sp.Rational(1, 5), sp.Rational(1, 5) + sp.log(5)],
])
assert sp.simplify(lnA - tb5) == sp.zeros(3)

# ---------- 不可交换反例与可交换正例 ----------
N = sp.Matrix([[0, 1], [0, 0]])
B = sp.Matrix([[0, 0], [1, 0]])
assert N * B != B * N
eN = sp.eye(2) + N
eB = sp.eye(2) + B
assert eN * eB == sp.Matrix([[2, 1], [1, 1]])
ApB = N + B
assert ApB ** 2 == sp.eye(2)
eSum = sp.cosh(1) * sp.eye(2) + sp.sinh(1) * ApB
assert sp.simplify(eN * eB - eSum) != sp.zeros(2)
assert abs(float(sp.cosh(1)) - 1.5431) < 5e-5
assert abs(float(sp.sinh(1)) - 1.1752) < 5e-5
assert sp.simplify((sp.eye(2) + N) * (sp.eye(2) + 2 * N) - (sp.eye(2) + 3 * N)) == sp.zeros(2)

print("ALL OK")
