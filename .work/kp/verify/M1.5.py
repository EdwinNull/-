"""M1.5 特征值与特征向量 —— 讲解中全部数值的复算脚本。
运行：python3 .work/kp/verify/M1.5.py   （全部 assert 通过即输出 ALL OK）
"""
import numpy as np
import sympy as sp

I = sp.I
lam = sp.symbols("lam")


def cp(M):
    """det(M - lam*I)，与教材约定一致。"""
    return sp.expand((M - lam * sp.eye(M.shape[0])).det())


def same(a, b):
    return sp.simplify(sp.expand(a - b)) == 0


# ---------------- K1 例 ----------------
A = sp.Matrix([[3, 1], [-2, 0]])
assert same(cp(A), (lam - 1) * (lam - 2))
x1, x2 = sp.Matrix([1, -2]), sp.Matrix([1, -1])
assert A * x1 == x1 and A * x2 == 2 * x2
assert (A - sp.eye(2)).nullspace()[0].T.tolist()[0] in ([sp.Rational(-1, 2), 1],)  # 与 [1,-2] 成比例
# K1 要点：旋转 90 度矩阵特征多项式 lam^2+1
R90 = sp.Matrix([[0, -1], [1, 0]])
assert same(cp(R90), lam ** 2 + 1) and set(R90.eigenvals()) == {I, -I}

# ---------------- K2 例（教材 1.5 节例 1）----------------
E1 = sp.Matrix([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
assert same(cp(E1), (1 - lam) * (2 + lam) ** 2)
# 教材所述两步行列式变形
d1 = sp.Matrix([[1, 1, 1], [1, -1 - lam, 1], [1, 1, -1 - lam]]).det()
assert same(cp(E1), (1 - lam) * d1) and same(d1, (-2 - lam) ** 2)
assert (E1 - sp.eye(3)).rank() == 2 and (E1 + 2 * sp.eye(3)).rank() == 1
assert E1 * sp.Matrix([1, 1, 1]) == sp.Matrix([1, 1, 1])
assert E1 * sp.Matrix([-1, 1, 0]) == -2 * sp.Matrix([-1, 1, 0])
assert E1 * sp.Matrix([-1, 0, 1]) == -2 * sp.Matrix([-1, 0, 1])
assert 3 - (E1 + 2 * sp.eye(3)).rank() == 2  # 几何重数 2

# ---------------- K3 例 ----------------
B3 = sp.Matrix([[I, 1], [0, 1 + I]])
assert same(cp(B3), (I - lam) * (1 + I - lam))
assert set(B3.eigenvals()) == {I, 1 + I}
assert set(B3.T.eigenvals()) == {I, 1 + I}
assert B3.H == sp.Matrix([[-I, 0], [1, 1 - I]])
assert set(B3.H.eigenvals()) == {-I, 1 - I}

# ---------------- K4 例 ----------------
assert A.trace() == 3 and A.det() == 2
assert E1.trace() == -3 and E1.det() == 4 and 1 * (-2) * (-2) == 4

# ---------------- K5 例 ----------------
assert sp.Matrix.hstack(x1, x2).det() == 1
s = x1 + x2
assert s == sp.Matrix([2, -3]) and A * s == sp.Matrix([3, -4])
assert 3 * (-3) - 2 * (-4) == -1

# ---------------- K6 例 ----------------
A2 = A * A
assert A2 == sp.Matrix([[7, 3], [-6, -2]])
assert A2 - 3 * A + 2 * sp.eye(2) == sp.zeros(2)
Ai = (3 * sp.eye(2) - A) / 2
assert Ai == sp.Matrix([[0, sp.Rational(-1, 2)], [1, sp.Rational(3, 2)]])
assert A * Ai == sp.eye(2) and Ai == A.inv()
assert set(Ai.eigenvals()) == {1, sp.Rational(1, 2)}
# 幂零矩阵
U4 = sp.Matrix([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
assert same(cp(U4), lam ** 4) and U4 ** 4 == sp.zeros(4)
N2 = sp.Matrix([[1, 1], [-1, -1]])
assert N2 ** 2 == sp.zeros(2) and same(cp(N2), lam ** 2)

# ---------------- K7 ----------------
# 例：秩 1
a = sp.Matrix([1, 2, 3]); b = sp.Matrix([1, 0, -1])
ab = a * b.T
assert ab == sp.Matrix([[1, 0, -1], [2, 0, -2], [3, 0, -3]])
assert (b.T * a)[0] == -2 and ab.trace() == -2
assert same(cp(ab), (-lam) ** 2 * ((b.T * a)[0] - lam))
assert same(cp(ab), (-1) ** 3 * lam ** 2 * (lam - (b.T * a)[0]))
assert same((lam * sp.eye(3) - ab).det(), lam ** 2 * (lam + 2))
assert set(ab.eigenvals()) == {0, -2} and ab.eigenvals()[0] == 2
# 反例：同阶但不相似
Ac = sp.Matrix([[0, 1], [0, 0]]); Bc = sp.Matrix([[0, 0], [0, 1]])
assert Ac * Bc == sp.Matrix([[0, 1], [0, 0]]) and Bc * Ac == sp.zeros(2)
assert same(cp(Ac * Bc), lam ** 2) and same(cp(Bc * Ac), lam ** 2)

# ---------------- K8 ----------------
# 例
Ak = sp.Matrix([[1, 2], [0, 3]])
Pk = sp.Matrix([[1, 1], [0, 1]])
assert Pk.inv() == sp.Matrix([[1, -1], [0, 1]])
assert Ak * Pk == sp.Matrix([[1, 3], [0, 3]])
assert Pk.inv() * Ak * Pk == sp.diag(1, 3)
# I 与 J 特征多项式相同
J2 = sp.Matrix([[1, 1], [0, 1]])
assert same(cp(sp.eye(2)), cp(J2)) and same(cp(J2), (1 - lam) ** 2)

# ---------------- K9 ----------------
Jn = sp.Matrix([[2, 1], [0, 2]])
assert (Jn - 2 * sp.eye(2)).rank() == 1 and 2 - 1 == 1

# ---------------- K10 例 5、例 6 ----------------
A5 = sp.Matrix([[1, 0, 0], [1, 2, -1], [-1, -1, 2]])
assert same(cp(A5), (3 - lam) * (1 - lam) ** 2)
assert (A5 - sp.eye(3)).rank() == 1
xs5 = [sp.Matrix([0, 1, -1]), sp.Matrix([-1, 1, 0]), sp.Matrix([1, 0, 1])]
assert A5 * xs5[0] == 3 * xs5[0] and A5 * xs5[1] == xs5[1] and A5 * xs5[2] == xs5[2]
P5 = sp.Matrix.hstack(*xs5)
assert P5 == sp.Matrix([[0, -1, 1], [1, 1, 0], [-1, 0, 1]])
assert P5 ** -1 * A5 * P5 == sp.diag(3, 1, 1)
# 书页 37 所印 P^{-1}
assert P5.inv() == sp.Matrix([[sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(-1, 2)],
                              [sp.Rational(-1, 2), sp.Rational(1, 2), sp.Rational(1, 2)],
                              [sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 2)]])
# 例 6
P6 = sp.Matrix([[1, 0, 2], [0, 1, 2], [1, -1, -1]])
assert P6.det() == -1
assert P6.inv() == sp.Matrix([[-1, 2, 2], [-2, 3, 2], [1, -1, -1]])
D6 = sp.diag(1, -1, -1)
A6 = P6 * D6 * P6.inv()
assert A6 == sp.Matrix([[-3, 4, 4], [0, -1, 0], [-2, 4, 3]])
assert A6 ** 2 == sp.eye(3) and A6 ** 8 == sp.eye(3) and D6 ** 2 == sp.eye(3)
for col, ev in zip(P6.T.tolist(), [1, -1, -1]):
    v = sp.Matrix(col)
    assert A6 * v == ev * v

# ---------------- K11 ----------------
th = sp.symbols("th", real=True)
Qr = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])
assert sp.simplify(Qr.T * Qr - sp.eye(2)) == sp.zeros(2)
assert sp.simplify(cp(Qr) - (lam ** 2 - 2 * sp.cos(th) * lam + 1)) == 0
ev_r = Qr.eigenvals()
assert all(sp.simplify(sp.Abs(e) ** 2 - 1) == 0 for e in ev_r)
Uu = sp.Matrix([[1, I], [I, 1]]) / sp.sqrt(2)
assert sp.simplify(Uu.H * Uu - sp.eye(2)) == sp.zeros(2)
u1 = Uu[:, 0]; u2 = Uu[:, 1]
inner = lambda p, q: (q.H * p)[0]  # <p,q> = sum p_i conj(q_i)
assert sp.simplify(inner(u1, u2)) == 0 and sp.simplify(inner(u1, u1)) == 1 and sp.simplify(inner(u2, u2)) == 1
for e in Uu.eigenvals():
    assert sp.simplify(sp.Abs(e) - 1) == 0
assert set(sp.simplify(e) for e in Uu.eigenvals()) == {sp.simplify((1 + I) / sp.sqrt(2)), sp.simplify((1 - I) / sp.sqrt(2))}
# 复向量 x=[1,i]：x^T x=0，x^H x=2
xx = sp.Matrix([1, I])
assert (xx.T * xx)[0] == 0 and (xx.H * xx)[0] == 2

# ---------------- K12 ----------------
Hh = sp.Matrix([[0, -I], [I, 0]])
assert Hh.H == Hh
assert same(cp(Hh), lam ** 2 - 1) and set(Hh.eigenvals()) == {1, -1}
hx, hy = sp.Matrix([-I, 1]), sp.Matrix([I, 1])
assert Hh * hx == hx and Hh * hy == -hy
assert sp.simplify(inner(hx, hy)) == 0
assert sp.simplify((hx.T * hy)[0]) == 2
# 复对称但非 Hermite
Cs = sp.Matrix([[0, I], [I, 0]])
assert Cs.T == Cs and Cs.H == -Cs and set(Cs.eigenvals()) == {I, -I}
# 特征值全实但非 Hermite
assert set(sp.Matrix([[1, 1], [0, 2]]).eigenvals()) == {1, 2}

# ---------------- K13 例 ----------------
Uh = sp.Matrix([[-I, I], [1, 1]]) / sp.sqrt(2)
assert sp.simplify(Uh.H * Uh - sp.eye(2)) == sp.zeros(2)
assert sp.simplify(Hh * Uh - Uh * sp.diag(1, -1)) == sp.zeros(2)
assert sp.simplify(Uh.H * Hh * Uh - sp.diag(1, -1)) == sp.zeros(2)
assert sp.simplify(Hh * Uh - sp.Matrix([[-I, -I], [1, -1]]) / sp.sqrt(2)) == sp.zeros(2)
assert sp.simplify(hx.H * hx)[0] == 2 and sp.simplify(hy.H * hy)[0] == 2

# ---------------- K14 例 8、例 9 ----------------
e1 = sp.Matrix([1, 1, 1 - I]) / 2
assert sp.simplify((e1.H * e1)[0]) == 1
e2 = sp.Matrix([-1, 1, 0]) / sp.sqrt(2)
e3 = sp.Matrix([-(1 + I) / 2, -(1 + I) / 2, 1]) / sp.sqrt(2)
assert sp.simplify(e3 - sp.Matrix([-(1 + I) / (2 * sp.sqrt(2)), -(1 + I) / (2 * sp.sqrt(2)), 1 / sp.sqrt(2)])) == sp.zeros(3, 1)
# 与 e1 正交的方程： x1+x2+(1+i)x3=0
for v in (sp.Matrix([-1, 1, 0]), sp.Matrix([-(1 + I) / 2, -(1 + I) / 2, 1])):
    assert sp.simplify(v[0] + v[1] + (1 + I) * v[2]) == 0
    assert sp.simplify(inner(v, e1)) == 0
assert sp.simplify(-(-(1 + I) / 2) + (-(1 + I) / 2)) == 0  # 与 ε2 正交：-x1+x2=0
assert sp.simplify(sp.Abs(-(1 + I) / 2) ** 2 * 2 + 1) == 2
Q8 = sp.Matrix.hstack(e1, e2, e3)
assert sp.simplify(Q8.H * Q8 - sp.eye(3)) == sp.zeros(3)
# 书页 41 所印 Q（第 2 列符号相反）仍是酉矩阵，但第 2 列 = -ε2
Q8p = sp.Matrix([[sp.Rational(1, 2), 1 / sp.sqrt(2), -(1 + I) / (2 * sp.sqrt(2))],
                 [sp.Rational(1, 2), -1 / sp.sqrt(2), -(1 + I) / (2 * sp.sqrt(2))],
                 [(1 - I) / 2, 0, 1 / sp.sqrt(2)]])
assert sp.simplify(Q8p.H * Q8p - sp.eye(3)) == sp.zeros(3)
assert sp.simplify(Q8p[:, 1] + e2) == sp.zeros(3, 1)
assert sp.simplify(Q8p[:, 1] - e2) != sp.zeros(3, 1)
# 例 9
A9 = sp.Matrix([[1, 2, 2], [2, 1, 2], [2, 2, 1]])
assert same(cp(A9), (5 - lam) * (1 + lam) ** 2)
q1 = sp.Matrix([1, 1, 1]) / sp.sqrt(3)
q2 = sp.Matrix([-1, 1, 0]) / sp.sqrt(2)
q3 = sp.Matrix([-1, -1, 2]) / sp.sqrt(6)
Q9 = sp.Matrix.hstack(q1, q2, q3)
assert sp.simplify(Q9.T * Q9 - sp.eye(3)) == sp.zeros(3)
assert sp.simplify(Q9.T * A9 * Q9 - sp.diag(5, -1, -1)) == sp.zeros(3)
assert sp.simplify(Q9 - sp.Matrix([[1 / sp.sqrt(3), -1 / sp.sqrt(2), -1 / sp.sqrt(6)],
                                   [1 / sp.sqrt(3), 1 / sp.sqrt(2), -1 / sp.sqrt(6)],
                                   [1 / sp.sqrt(3), 0, 2 / sp.sqrt(6)]])) == sp.zeros(3)
assert sp.Matrix([-1, 1, 0]).dot(sp.Matrix([-1, 0, 1])) == 1
assert sp.Matrix([-1, 1, 0]).dot(sp.Matrix([-sp.Rational(1, 2), -sp.Rational(1, 2), 1])) == 0

# ---------------- 典型例题 1 ----------------
a_ = sp.symbols("a_")
M = sp.Matrix([[1, 0, 0], [a_, 1, 0], [1, 1, 2]])
assert same(cp(M), (1 - lam) ** 2 * (2 - lam))
assert (M - sp.eye(3)).subs(a_, 3).rank() == 2 and (M - sp.eye(3)).subs(a_, -7).rank() == 2
assert (M - sp.eye(3)).subs(a_, 0).rank() == 1
M0 = M.subs(a_, 0)
assert M0 == sp.Matrix([[1, 0, 0], [0, 1, 0], [1, 1, 2]])
P1 = sp.Matrix([[-1, -1, 0], [1, 0, 0], [0, 1, 1]])
assert P1.det() == 1 and P1.inv() == sp.Matrix([[0, 1, 0], [-1, -1, 0], [1, 1, 1]])
assert P1.inv() * M0 * P1 == sp.diag(1, 1, 2)
k = sp.symbols("k", integer=True, positive=True)
Mk = sp.Matrix([[1, 0, 0], [0, 1, 0], [2 ** k - 1, 2 ** k - 1, 2 ** k]])
assert sp.simplify(P1 * sp.diag(1, 1, 2 ** k) * P1.inv() - Mk) == sp.zeros(3)
for kk in range(1, 8):
    assert M0 ** kk == Mk.subs(k, kk)
assert (M0 - 2 * sp.eye(3)) * sp.Matrix([0, 0, 1]) == sp.zeros(3, 1)

# ---------------- 典型例题 2 ----------------
Am = sp.Matrix([[1, 0], [0, 1], [1, 1]]); Bm = sp.Matrix([[1, 2, 0], [0, 1, 1]])
assert Bm * Am == sp.Matrix([[1, 2], [1, 2]])
assert same(cp(Bm * Am), lam * (lam - 3))
AB = Am * Bm
assert AB == sp.Matrix([[1, 2, 0], [0, 1, 1], [1, 3, 1]])
assert same(cp(AB), (-lam) * lam * (lam - 3)) and same(cp(AB), -lam ** 2 * (lam - 3))
assert AB.trace() == 3 and AB.rank() == 2
assert AB.row(2) == AB.row(0) + AB.row(1)
assert AB.eigenvals() == {0: 2, 3: 1}
assert not AB.is_diagonalizable()
assert sp.Matrix([[1, 2], [1, 2]]).is_diagonalizable()

# ---------------- 典型例题 3 ----------------
H = sp.Matrix([[2, 1 - I], [1 + I, 3]])
assert H.H == H
assert same(cp(H), (lam - 1) * (lam - 4))
hxx, hyy = sp.Matrix([-1 + I, 1]), sp.Matrix([1 - I, 2])
assert sp.expand(H * hxx - hxx) == sp.zeros(2, 1) and sp.expand(H * hyy - 4 * hyy) == sp.zeros(2, 1)
assert sp.simplify((H - sp.eye(2)) * hxx) == sp.zeros(2, 1)
assert sp.expand((-1 + I) * (1 + I) + 2) == 0
assert sp.simplify((hxx.H * hxx)[0]) == 3 and sp.simplify((hyy.H * hyy)[0]) == 6
UH = sp.Matrix.hstack(hxx / sp.sqrt(3), hyy / sp.sqrt(6))
assert sp.simplify(UH.H * UH - sp.eye(2)) == sp.zeros(2)
assert sp.simplify(UH.H * H * UH - sp.diag(1, 4)) == sp.zeros(2)
# 漏取共轭会得到非零
assert sp.expand((hxx.T * hyy)[0]) != 0

# ---------------- 典型例题 4 ----------------
Pq = sp.Matrix([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
assert Pq.det() != 0
A4 = Pq * sp.diag(1, 2, 3) * Pq.inv()
B4 = A4 ** 2 - 3 * A4 + 2 * sp.eye(3)
assert B4.trace() == 2 and B4.det() == 0 and B4.rank() == 1
assert set(B4.eigenvals()) == {0, 2} and B4.eigenvals()[0] == 2
assert (A4.inv() + 2 * sp.eye(3)).det() == sp.Rational(35, 2)
assert 3 * sp.Rational(5, 2) * sp.Rational(7, 3) == sp.Rational(35, 2)
assert sp.Rational(1, 1) / 1 + 2 == 3 and sp.Rational(1, 2) + 2 == sp.Rational(5, 2) and sp.Rational(1, 3) + 2 == sp.Rational(7, 3)

# ---------------- 易错点 ----------------
# 3：[[2,1],[0,2]] 已在 K9 验证
# 5：A,B 同阶不相似 已在 K7 验证
# 6：复对称 [[0,i],[i,0]] 已在 K12 验证；k8 无数值

# ---------------- 自测题 ----------------
# 2
g = lambda t: t ** 2 + 1
assert 1 - 1 + 3 == 3 and 1 * (-1) * 3 == -3
assert [g(1), g(-1), g(3)] == [2, 2, 10]
# 3
A3 = sp.Matrix([[4, -2], [1, 1]])
assert same(cp(A3), (lam - 2) * (lam - 3))
assert A3 * sp.Matrix([1, 1]) == 2 * sp.Matrix([1, 1]) and A3 * sp.Matrix([2, 1]) == 3 * sp.Matrix([2, 1])
P3 = sp.Matrix([[1, 2], [1, 1]])
assert P3.inv() == sp.Matrix([[-1, 2], [1, -1]])
assert P3.inv() * A3 * P3 == sp.diag(2, 3)
assert P3 * sp.diag(32, 243) * P3.inv() == sp.Matrix([[454, -422], [211, -179]]) == A3 ** 5
# 5
U5 = sp.Matrix([[1, 1], [I, -I]]) / sp.sqrt(2)
assert sp.simplify(U5.H * U5 - sp.eye(2)) == sp.zeros(2)
c1, c2 = U5[:, 0], U5[:, 1]
assert sp.simplify(inner(c1, c2)) == 0 and sp.simplify(inner(c1, c1)) == 1
assert sp.simplify((U5.T * U5)[0, 0]) == 0
# 6
S = sp.Matrix([[2, 0, 0], [0, 2, 0], [0, 0, 5]])
assert (S - 2 * sp.eye(3)).rank() == 1 and (S - 5 * sp.eye(3)).rank() == 2
# 7：4×2 与 2×4
A7 = sp.Matrix([[1, 0], [0, 1], [0, 0], [0, 0]])
B7 = sp.Matrix([[3, 0, 1, 2], [0, 5, 3, 4]])
assert set((B7 * A7).eigenvals()) == {3, 5}
AB7 = A7 * B7
assert AB7.eigenvals() == {3: 1, 5: 1, 0: 2}
assert same(cp(AB7), lam ** 2 * (lam - 3) * (lam - 5))

# ---------------- 习题导航中的说明性数值 ----------------
# 习题 1.5 第 1 题：A 上三角 -2 三重，几何重数 1；B 二重根 1 几何重数 1；C 特征值 1±i
Ae = sp.Matrix([[-2, 1, 0], [0, -2, 1], [0, 0, -2]])
assert (Ae + 2 * sp.eye(3)).rank() == 2
Be = sp.Matrix([[1, 1, -1], [1, 1, 1], [0, -1, 2]])
assert same(cp(Be), -(lam - 1) ** 2 * (lam - 2)) and (Be - sp.eye(3)).rank() == 2
Ce = sp.Matrix([[1, 1], [-1, 1]])
assert set(Ce.eigenvals()) == {1 + I, 1 - I}
# 习题 1.5 第 2 题：P 与特征值
Pe = sp.Matrix([[1, 2, -2], [2, -2, -1], [2, 1, 2]])
Ae2 = Pe * sp.diag(1, 0, -1) * Pe.inv()
assert Ae2 ** 8 == Pe * sp.diag(1, 0, 1) * Pe.inv()
# 习题 1.5 第 4 题：二重特征值 1 的基础解系向量不正交
As = sp.Matrix([[5, -4, 2], [-4, 5, -2], [2, -2, 2]])
assert same(cp(As), -(lam - 1) ** 2 * (lam - 10))
assert sp.Matrix([1, 0, -2]).dot(sp.Matrix([0, 1, 2])) == -4
# 习题 1.5 第 5 题：x=1,y=-1
Ax5 = sp.Matrix([[-1, 2, 4], [2, 1, 2], [4, 2, -1]])
assert Ax5.trace() == -1 and Ax5.det() == 25
assert same(cp(Ax5), cp(sp.diag(5, -1, -5)))

print("ALL OK")
