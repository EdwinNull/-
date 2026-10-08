"""M4.1 矩阵序列与矩阵级数 —— 讲解中全部数值的复算脚本。
运行：python3 .work/kp/verify/M4.1.py
"""
import numpy as np
import sympy as sp

k = sp.symbols("k", integer=True, positive=True)
lam = sp.symbols("lam")

# ---------- 4.1 节例 1 ----------
assert sp.summation(sp.Rational(1, 2) ** k, (k, 1, sp.oo)) == 1
assert sp.simplify(sp.summation(sp.pi / (3 * 4 ** k), (k, 1, sp.oo)) - sp.pi / 9) == 0
assert sp.summation(1 / (k * (k + 1)), (k, 1, sp.oo)) == 1

def Ak(n):
    return sp.Matrix([[sp.Rational(1, 2 ** n), sp.pi / (3 * 4 ** n)],
                      [0, sp.Rational(1, n * (n + 1))]])

S1 = Ak(1)
assert S1 == sp.Matrix([[sp.Rational(1, 2), sp.pi / 12], [0, sp.Rational(1, 2)]])
S2 = Ak(1) + Ak(2)
assert S2 == sp.Matrix([[sp.Rational(3, 4), 5 * sp.pi / 48], [0, sp.Rational(2, 3)]])
S3 = S2 + Ak(3)
assert S3 == sp.Matrix([[sp.Rational(7, 8), 7 * sp.pi / 64], [0, sp.Rational(3, 4)]])

# K2 短例：调和级数部分和
assert sum(sp.Rational(1, n) for n in range(1, 5)) == sp.Rational(25, 12)

# ---------- 书页 93 Jordan 块反例 ----------
A = sp.Matrix([[sp.Rational(4, 5), 1], [0, sp.Rational(4, 5)]])
An = np.array([[0.8, 1.0], [0.0, 0.8]])
assert abs(np.linalg.norm(An, np.inf) - 1.8) < 1e-12
assert abs(np.linalg.norm(An, 1) - 1.8) < 1e-12
assert abs(max(abs(np.linalg.eigvals(An))) - 0.8) < 1e-12
for n in range(1, 8):
    Jk = sp.Matrix([[sp.Rational(4, 5) ** n, n * sp.Rational(4, 5) ** (n - 1)],
                    [0, sp.Rational(4, 5) ** n]])
    assert A ** n == Jk
assert A ** 2 == sp.Matrix([[sp.Rational(16, 25), sp.Rational(8, 5)], [0, sp.Rational(16, 25)]])
assert A ** 3 == sp.Matrix([[sp.Rational(64, 125), sp.Rational(48, 25)], [0, sp.Rational(64, 125)]])
assert float(30 * (sp.Rational(4, 5) ** 29)) < 0.05  # 正文不引用此具体值，只确认多项式因子仍被压住

# K1 反例：极限不可逆时逆矩阵不必收敛
for n in (2, 3, 5):
    Ak_inv = (sp.Rational(1, n) * sp.eye(2)).inv()
    assert Ak_inv == n * sp.eye(2)

# ---------- 4.1 节例 2 ----------
B = sp.Matrix([[1, 4], [-2, -3]])
assert sp.expand(B.charpoly(lam).as_expr()) == lam ** 2 + 2 * lam + 5
assert set(sp.solve(lam ** 2 + 2 * lam + 5, lam)) == {-1 + 2 * sp.I, -1 - 2 * sp.I}
assert set(B.eigenvals()) == {-1 + 2 * sp.I, -1 - 2 * sp.I}
for ev in B.eigenvals():
    assert sp.simplify(sp.Abs(ev) - sp.sqrt(5)) == 0
assert abs(float(sp.sqrt(5)) - 2.236067977) < 1e-8
# 比值：|c_k/c_{k+1}|=(k+1)/k → 1
assert sp.limit((k + 1) / k, k, sp.oo) == 1

# ---------- 4.1 节例 3 ----------
C = sp.Matrix([[1, 4], [-1, -3]])
assert sp.factor(C.charpoly(lam).as_expr()) == (lam + 1) ** 2
assert C.eigenvals() == {-1: 2}
P = sp.Matrix([[2, 1], [-1, 0]])
J = sp.Matrix([[-1, 1], [0, -1]])
assert P.det() == 1
assert P.inv() == sp.Matrix([[0, -1], [1, 2]])
assert P * J * P.inv() == C
assert P.inv() * C * P == J
assert C * P == P * J
assert J ** 3 == sp.Matrix([[-1, 3], [0, -1]])
# 部分和前 4 项
T4 = sp.zeros(2)
for n in range(1, 5):
    T4 += sp.Rational(1, n ** 2) * (J ** n)
assert T4 == sp.Matrix([[sp.Rational(-115, 144), sp.Rational(7, 12)],
                       [0, sp.Rational(-115, 144)]])
# 标量级数的和
assert sp.simplify(sp.summation(((-1) ** k) / k ** 2, (k, 1, sp.oo)) + sp.pi ** 2 / 12) == 0
assert sp.simplify(sp.summation(((-1) ** (k - 1)) / k, (k, 1, sp.oo)) - sp.log(2)) == 0
Msum = sp.Matrix([[-sp.pi ** 2 / 12, sp.log(2)], [0, -sp.pi ** 2 / 12]])
closed = sp.simplify(P * Msum * P.inv())
expected = sp.Matrix([
    [-sp.pi ** 2 / 12 + 2 * sp.log(2), 4 * sp.log(2)],
    [-sp.log(2), -2 * sp.log(2) - sp.pi ** 2 / 12],
])
assert sp.simplify(closed - expected) == sp.zeros(2)
# 数值部分和逼近（交错调和级数收敛慢，取 400 项，误差应小于 0.01）
Cn = np.array(C.tolist(), dtype=float)
partial = np.zeros((2, 2))
for n in range(1, 401):
    partial += np.linalg.matrix_power(Cn, n) / n ** 2
target = np.array([[float(x) for x in row] for row in expected.tolist()])
assert np.allclose(partial, target, atol=0.01)

# ---------- 自编例题 4 ----------
G = sp.Matrix([[sp.Rational(1, 2), sp.Rational(1, 4)], [0, sp.Rational(1, 3)]])
assert set(G.eigenvals()) == {sp.Rational(1, 2), sp.Rational(1, 3)}
Gn = np.array(G.tolist(), dtype=float)
assert abs(np.linalg.norm(Gn, np.inf) - 0.75) < 1e-12
assert abs(np.linalg.norm(Gn, 1) - 7 / 12) < 1e-12
assert G ** 2 == sp.Matrix([[sp.Rational(1, 4), sp.Rational(5, 24)], [0, sp.Rational(1, 9)]])
assert G ** 3 == sp.Matrix([[sp.Rational(1, 8), sp.Rational(19, 144)], [0, sp.Rational(1, 27)]])

def b_formula(n):
    return sp.Rational(3, 2) * (sp.Rational(1, 2) ** n - sp.Rational(1, 3) ** n)

for n in range(1, 7):
    assert (G ** n)[0, 1] == b_formula(n)
# 递推一步
n = sp.symbols("n", integer=True, positive=True)
bk = sp.Rational(3, 2) * (sp.Rational(1, 2) ** n - sp.Rational(1, 3) ** n)
bnext = sp.Rational(1, 2) * bk + sp.Rational(1, 4) * sp.Rational(1, 3) ** n
assert sp.simplify(bnext - b_formula(n + 1)) == 0
assert sp.summation(b_formula(k), (k, 1, sp.oo)) == sp.Rational(3, 4)
assert sp.summation(sp.Rational(1, 2) ** k, (k, 0, sp.oo)) == 2
assert sp.summation(sp.Rational(1, 3) ** k, (k, 0, sp.oo)) == sp.Rational(3, 2)
S3g = sum((G ** n for n in range(0, 4)), sp.zeros(2))
assert S3g == sp.Matrix([[sp.Rational(15, 8), sp.Rational(85, 144)], [0, sp.Rational(40, 27)]])
closed_g = (sp.eye(2) - G).inv()
assert closed_g == sp.Matrix([[2, sp.Rational(3, 4)], [0, sp.Rational(3, 2)]])
assert closed_g - S3g == sp.Matrix([[sp.Rational(1, 8), sp.Rational(23, 144)], [0, sp.Rational(1, 54)]])

# ---------- 自测题 ----------
H = sp.Matrix([[sp.Rational(1, 2), 1], [0, sp.Rational(1, 2)]])
assert float(np.linalg.norm(np.array([[0.5, 1.0], [0.0, 0.5]]), np.inf)) == 1.5
assert H ** 3 == sp.Matrix([[sp.Rational(1, 8), sp.Rational(3, 4)], [0, sp.Rational(1, 8)]])
assert max(abs(complex(ev)) for ev in H.eigenvals()) == 0.5
series_sum = sp.Matrix([[sp.summation(sp.Rational(1, 2) ** k, (k, 1, sp.oo)), 0],
                        [0, sp.summation(1 / (k * (k + 1)), (k, 1, sp.oo))]])
assert series_sum == sp.eye(2)

# ---------- 习题 4.1 第 3 题导航中的数值 ----------
E = sp.Matrix([[1, 2], [8, 1]])
assert sp.factor(E.charpoly(lam).as_expr()) == (lam - 5) * (lam + 3)
assert set(E.eigenvals()) == {5, -3}
assert max(abs(v) for v in E.eigenvals()) == 5
assert sp.limit(k ** 2 / (k + 1) ** 2, k, sp.oo) == 1
# 第 2 题系数 c_k=k 的收敛半径
assert sp.limit(k / (k + 1), k, sp.oo) == 1

print("ALL OK")
