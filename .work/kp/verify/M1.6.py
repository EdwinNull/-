"""M1.6 实二次型：讲解中全部数值与符号推导的验算脚本。

覆盖：教材 1.6 节例 1–5（含式 (1.6-7)），知识点中的小例，典型例题 1–3，易错点，自测题答案，习题导航中引用的数值。
"""
import numpy as np
import sympy as sp

x1, x2, x3, x4, t, eps = sp.symbols("x1 x2 x3 x4 t eps", real=True)


def quad(A, xs):
    X = sp.Matrix(xs)
    return sp.expand((X.T * A * X)[0])


def minors(A):
    return [A[:k, :k].det() for k in range(1, A.shape[0] + 1)]


def is_pd_numeric(A):
    return bool(np.all(np.linalg.eigvalsh(np.array(A, dtype=float)) > 1e-12))


# ---------- K1：教材 1.6 节例 1 ----------
f1 = 3*x1**2 + 2*x2**2 - x3**2 - 2*x1*x3 + 4*x2*x3
A1 = sp.Matrix([[3, 0, -1], [0, 2, 2], [-1, 2, -1]])
assert sp.expand(quad(A1, [x1, x2, x3]) - f1) == 0
assert A1 == A1.T

# K1 要点：非对称矩阵给出同一个二次型
fa = x1**2 + 4*x1*x2 + x2**2
S = sp.Matrix([[1, 2], [2, 1]])
U = sp.Matrix([[1, 4], [0, 1]])
assert sp.expand(quad(S, [x1, x2]) - fa) == 0
assert sp.expand(quad(U, [x1, x2]) - fa) == 0
assert sorted(S.eigenvals().keys()) == [-1, 3]
assert list(U.eigenvals().keys()) == [1] and U.eigenvals()[1] == 2

# ---------- K2/K4：f = 2 x1 x2 ----------
y1, y2 = sp.symbols("y1 y2", real=True)
C = sp.Matrix([[1, 1], [1, -1]])
B2 = sp.Matrix([[0, 1], [1, 0]])
assert sp.expand(quad(B2, [x1, x2]) - 2*x1*x2) == 0
assert C.det() == -2
assert C.T * B2 * C == sp.diag(2, -2)                       # f = 2 y1^2 - 2 y2^2
Q2 = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
assert sp.simplify(Q2.T * Q2) == sp.eye(2)
assert sp.simplify(Q2.T * B2 * Q2) == sp.diag(1, -1)         # f = y1^2 - y2^2
assert sorted(B2.eigenvals().keys()) == [-1, 1]

# ---------- K3：合同 ----------
Ai = sp.eye(2)
P3 = sp.Matrix([[1, 1], [0, 1]])
Bc = P3.T * Ai * P3
assert Bc == sp.Matrix([[1, 1], [1, 2]])
assert P3.det() == 1
assert sorted(Ai.eigenvals().keys()) == [1]
assert sorted(Bc.eigenvals().keys(), key=float) == [sp.Rational(3, 2) - sp.sqrt(5)/2, sp.Rational(3, 2) + sp.sqrt(5)/2]
assert sp.simplify(Bc.det() - 1) == 0

# ---------- K4：教材 1.6 节例 2 ----------
f2 = 2*x1*x2 + 2*x1*x3 - 2*x1*x4 - 2*x2*x3 + 2*x2*x4 + 2*x3*x4
A2 = sp.Matrix([[0, 1, 1, -1], [1, 0, -1, 1], [1, -1, 0, 1], [-1, 1, 1, 0]])
assert sp.expand(quad(A2, [x1, x2, x3, x4]) - f2) == 0
lam = sp.symbols("lam")
cp = sp.factor((A2 - lam*sp.eye(4)).det())
assert sp.expand(cp - (lam + 3)*(lam - 1)**3) == 0
assert A2.eigenvals() == {-3: 1, 1: 3}
v1 = sp.Matrix([1, -1, -1, 1])
assert (A2 + 3*sp.eye(4)) * v1 == sp.zeros(4, 1)
assert (A2 - sp.eye(4)).rank() == 1
w2, w3, w4 = sp.Matrix([1, 1, 0, 0]), sp.Matrix([0, 0, 1, 1]), sp.Matrix([1, -1, 1, -1])
for w in (w2, w3, w4):
    assert (A2 - sp.eye(4)) * w == sp.zeros(4, 1)
assert w2.dot(w3) == 0 and w2.dot(w4) == 0 and w3.dot(w4) == 0
assert [w.dot(w) for w in (v1, w2, w3, w4)] == [4, 2, 2, 4]
assert (-1)*1 + 1*(-1) + 1*1 + (-1)*(-1) == 0
Qe = sp.Matrix.hstack(v1/2, w2/sp.sqrt(2), w3/sp.sqrt(2), w4/2)
assert sp.simplify(Qe.T * Qe) == sp.eye(4)
assert sp.simplify(Qe.T * A2 * Qe) == sp.diag(-3, 1, 1, 1)
assert Qe[0, 0] == sp.Rational(1, 2) and Qe[1, 3] == -sp.Rational(1, 2)

# 教材例 2 的第一个行列式变形：各行相加后第一行为 (1-lam) 倍的全 1 行
Ml = A2 - lam*sp.eye(4)
rowsum = [sp.simplify(sum(Ml[:, j])) for j in range(4)]
assert all(sp.simplify(r - (1 - lam)) == 0 for r in rowsum[:1])
assert all(sp.simplify(sum(Ml[i, :]) - (1 - lam)) == 0 for i in range(4))
inner = sp.Matrix([[-lam - 1, -2], [-2, -lam - 1]]).det()
assert sp.expand((1 - lam)**2 * inner - (lam + 3)*(lam - 1)**3) == 0

# ---------- K5：定理 1.6-2 的小例 ----------
A5 = sp.Matrix([[2, 1], [1, 2]])
assert A5.eigenvals() == {1: 1, 3: 1}
assert sp.expand(quad(A5, [x1, x2]) - (2*x1**2 + 2*x1*x2 + 2*x2**2)) == 0
assert sp.expand(2*x1**2 + 2*x1*x2 + 2*x2**2 - ((x1 + x2)**2 + x1**2 + x2**2)) == 0

# ---------- K6：A = P^T P ----------
s3 = sp.sqrt(3)
P6 = sp.Matrix([[(s3 + 1)/2, (s3 - 1)/2], [(s3 - 1)/2, (s3 + 1)/2]])
assert sp.simplify(P6.T * P6 - A5) == sp.zeros(2, 2)
assert sp.simplify(P6.det() - s3) == 0
assert A5.det() == 3

# ---------- K7：式 (1.6-7) 的数值例 ----------
A7 = sp.Matrix([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
An1 = A7[:2, :2]
beta = A7[:2, 2]
ann = A7[2, 2]
b = ann - (beta.T * An1.inv() * beta)[0]
assert b == sp.Rational(4, 3)
L = sp.Matrix.vstack(sp.Matrix.hstack(sp.eye(2), sp.zeros(2, 1)), sp.Matrix.hstack(-beta.T*An1.inv(), sp.Matrix([[1]])))
R = sp.Matrix.vstack(sp.Matrix.hstack(sp.eye(2), -An1.inv()*beta), sp.Matrix.hstack(sp.zeros(1, 2), sp.Matrix([[1]])))
assert L * A7 * R == sp.Matrix.vstack(sp.Matrix.hstack(An1, sp.zeros(2, 1)), sp.Matrix.hstack(sp.zeros(1, 2), sp.Matrix([[b]])))
assert L.T == R
assert A7.det() == 4 and An1.det() == 3 and sp.Rational(4, 3) * 3 == 4
assert minors(A7) == [2, 3, 4]

# 教材 1.6 节例 5
f5 = x1**2 + x2**2 + 5*x3**2 + 2*t*x1*x2 - 2*x1*x3 + 4*x2*x3
A5b = sp.Matrix([[1, t, -1], [t, 1, 2], [-1, 2, 5]])
assert sp.expand(quad(A5b, [x1, x2, x3]) - f5) == 0
m5 = minors(A5b)
assert sp.expand(m5[1] - (1 - t**2)) == 0
assert sp.expand(m5[2] - (-5*t**2 - 4*t)) == 0
assert sp.solveset(t**2 - 1 < 0, t, sp.S.Reals) == sp.Interval.open(-1, 1)
assert sp.solveset(t*(5*t + 4) < 0, t, sp.S.Reals) == sp.Interval.open(-sp.Rational(4, 5), 0)
for tv in (-0.5, -0.79, -0.81, 0.1, -0.1):
    pd = is_pd_numeric(A5b.subs(t, tv))
    assert pd == (-0.8 < tv < 0), tv
assert is_pd_numeric(A5b.subs(t, sp.Rational(-2, 5)))
assert A5b.subs(t, sp.Rational(-2, 5)).det() == -5*sp.Rational(4, 25) + sp.Rational(8, 5)  # = 4/5

# ---------- K8：半正定 ----------
A8 = sp.Matrix([[1, 1], [1, 1]])
assert A8.eigenvals() == {0: 1, 2: 1}
P8 = sp.Matrix([[1, 1], [0, 0]])
assert P8.T * P8 == A8
assert A8.det() == 0
assert sp.expand(quad(A8, [x1, x2]) - (x1 + x2)**2) == 0
A8c = sp.diag(0, -1)
assert minors(A8c) == [0, 0]
assert A8c.eigenvals() == {0: 1, -1: 1}
# 例：f=(x1-x2)^2
A8b = sp.Matrix([[1, -1], [-1, 1]])
assert sp.expand(quad(A8b, [x1, x2]) - (x1 - x2)**2) == 0
assert A8b.eigenvals() == {0: 1, 2: 1}

# ---------- K9：教材例 3、例 4 + 数值例 ----------
A9 = sp.Matrix([[2, 1], [1, 1]])
assert A9.det() == 1 and is_pd_numeric(A9)
assert A9.inv() == sp.Matrix([[1, -1], [-1, 2]])
assert A9.adjugate() == sp.Matrix([[1, -1], [-1, 2]])
assert A9.adjugate() == A9.inv() * A9.det()
assert is_pd_numeric(A9.inv())
assert sorted(A9.eigenvals().keys(), key=float) == [sp.Rational(3, 2) - sp.sqrt(5)/2, sp.Rational(3, 2) + sp.sqrt(5)/2]
B9 = sp.diag(1, 2)
AB = A9 * B9
assert AB == sp.Matrix([[2, 2], [1, 2]])
assert B9 * A9 == sp.Matrix([[2, 1], [2, 2]])
assert AB != AB.T
assert AB.trace() == 4 and AB.det() == 2
assert sorted(AB.eigenvals().keys(), key=float) == [2 - sp.sqrt(2), 2 + sp.sqrt(2)]
assert all(float(v) > 0 for v in AB.eigenvals())
# AB = BA 时：A=diag(1,2) 与 B=diag(3,1)
Ad, Bd = sp.diag(1, 2), sp.diag(3, 1)
assert Ad*Bd == Bd*Ad == sp.diag(3, 2) and is_pd_numeric(Ad*Bd)

# ---------- 易错点 ----------
E1 = sp.Matrix([[2, -1], [-1, 2]])
assert is_pd_numeric(E1)                       # 正定但有负的非对角元
E2 = sp.Matrix([[1, 2, 2], [2, 1, 2], [2, 2, 1]])
assert E2.eigenvals() == {5: 1, -1: 2}
assert E2.det() == 5 and all(E2[i, i] > 0 for i in range(3))
assert not is_pd_numeric(E2)
assert minors(E2) == [1, -3, 5]
E3 = sp.diag(-1, -1)
assert E3.det() == 1 and not is_pd_numeric(E3)
# 负定：奇数阶顺序主子式 <0，偶数阶 >0
E4 = sp.Matrix([[-2, 1], [1, -2]])
assert minors(E4) == [-2, 3]
assert sorted(E4.eigenvals().keys()) == [-3, -1]

# ---------- 典型例题 1 ----------
TA = sp.Matrix([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
fT = 2*x1**2 + 2*x2**2 + 2*x3**2 + 2*x1*x2 + 2*x1*x3 + 2*x2*x3
assert sp.expand(quad(TA, [x1, x2, x3]) - fT) == 0
assert sp.expand((TA - lam*sp.eye(3)).det() - (4 - lam)*(1 - lam)**2) == 0
assert TA.eigenvals() == {4: 1, 1: 2}
for i in range(3):
    assert sum((TA - lam*sp.eye(3))[i, :]).subs(lam, 0) == 4
va, vb, vc = sp.Matrix([1, 1, 1]), sp.Matrix([1, -1, 0]), sp.Matrix([1, 1, -2])
assert TA*va == 4*va and TA*vb == vb and TA*vc == vc
assert va.dot(vb) == 0 and va.dot(vc) == 0 and vb.dot(vc) == 0
assert [va.dot(va), vb.dot(vb), vc.dot(vc)] == [3, 2, 6]
QT = sp.Matrix.hstack(va/sp.sqrt(3), vb/sp.sqrt(2), vc/sp.sqrt(6))
assert sp.simplify(QT.T*QT) == sp.eye(3)
assert sp.simplify(QT.T*TA*QT) == sp.diag(4, 1, 1)
assert sp.Matrix([1, 1, 1]).dot(sp.Matrix([1, 1, -2])) == 0
# 非同号：用另一对基 [1,-1,0],[0,1,-1] 不正交（说明必须正交化）
assert sp.Matrix([1, -1, 0]).dot(sp.Matrix([0, 1, -1])) == -1
# 另一组也可：施密特正交化 [0,1,-1] 对 [1,-1,0]
u = sp.Matrix([0, 1, -1]) - (sp.Matrix([0, 1, -1]).dot(vb)/vb.dot(vb))*vb
assert u == sp.Matrix([sp.Rational(1, 2), sp.Rational(1, 2), -1])
assert u*2 == vc

# ---------- 典型例题 2 ----------
tri = sp.Matrix([[2, t, 0], [t, 2, t], [0, t, 2]])
mt = minors(tri)
assert mt[0] == 2 and sp.expand(mt[1] - (4 - t**2)) == 0
assert sp.expand(mt[2] - (8 - 4*t**2)) == 0
assert sp.solveset(4 - t**2 > 0, t, sp.S.Reals) == sp.Interval.open(-2, 2)
assert sp.solveset(8 - 4*t**2 > 0, t, sp.S.Reals) == sp.Interval.open(-sp.sqrt(2), sp.sqrt(2))
ev = tri.eigenvals()
assert set(ev.keys()) == {sp.Integer(2), 2 + sp.sqrt(2)*t, 2 - sp.sqrt(2)*t}
for tv in (-1.4, -1.0, 0.0, 0.7, 1.41):
    assert is_pd_numeric(tri.subs(t, tv))
for tv in (-1.5, 1.42, 2.0):
    assert not is_pd_numeric(tri.subs(t, tv))
# 临界 t = sqrt(2)：半正定且奇异
tc = tri.subs(t, sp.sqrt(2))
assert sp.simplify(tc.det()) == 0
evc = np.linalg.eigvalsh(np.array(tc.evalf(), dtype=float))
assert abs(evc[0]) < 1e-12 and evc[1] > 0
assert sp.expand(quad(tri, [x1, x2, x3]) - (2*x1**2 + 2*x2**2 + 2*x3**2 + 2*t*x1*x2 + 2*t*x2*x3)) == 0

# ---------- 典型例题 3 ----------
A3 = sp.Matrix([[1, 1], [1, 1]])
e3 = sp.symbols("e3", positive=True)
assert (A3 + e3*sp.eye(2)).eigenvals() == {e3: 1, e3 + 2: 1}
assert [m for m in minors(A3 + e3*sp.eye(2))] == [1 + e3, sp.expand((1 + e3)**2 - 1)]
assert sp.expand((1 + e3)**2 - 1 - e3*(e3 + 2)) == 0
# A 有负特征值时：A = diag(2,-1)，A + eps I 在 eps = 1 时奇异
assert (sp.diag(2, -1) + sp.eye(2)).det() == 0

# ---------- 自测题 ----------
# 1
fs = x1**2 - 2*x2**2 + 4*x1*x2 - 6*x2*x3
As = sp.Matrix([[1, 2, 0], [2, -2, -3], [0, -3, 0]])
assert sp.expand(quad(As, [x1, x2, x3]) - fs) == 0
# 2
assert E2.det() == 5
# 3
A3t = sp.Matrix([[3, 1, 0], [1, 2, 1], [0, 1, 1]])
assert minors(A3t) == [3, 5, 2] and is_pd_numeric(A3t)
# 4
A4t = sp.Matrix([[-1, t, 0], [t, -2, 0], [0, 0, -3]])
m4 = minors(A4t)
assert m4[0] == -1 and sp.expand(m4[1] - (2 - t**2)) == 0 and sp.expand(m4[2] - (-3)*(2 - t**2)) == 0
for tv in (-1.4, 0, 1.4):
    ev4 = np.linalg.eigvalsh(np.array(A4t.subs(t, tv), dtype=float))
    assert np.all(ev4 < 0)
for tv in (-1.5, 1.5, 2.0):
    ev4 = np.linalg.eigvalsh(np.array(A4t.subs(t, tv), dtype=float))
    assert np.any(ev4 > 0)
# 5
assert E3.det() == 1
# 6：A 正定 => A^2 = A^T A，A 可逆
A6t = sp.Matrix([[2, 1], [1, 2]])
assert A6t**2 == A6t.T * A6t == sp.Matrix([[5, 4], [4, 5]])
assert (A6t**2).eigenvals() == {1: 1, 9: 1}
# 7
A7t = sp.Matrix([[3, 1], [1, 3]])
f7 = 3*x1**2 + 2*x1*x2 + 3*x2**2
assert sp.expand(quad(A7t, [x1, x2]) - f7) == 0
assert A7t.eigenvals() == {4: 1, 2: 1}
Q7 = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
assert sp.simplify(Q7.T * A7t * Q7) == sp.diag(4, 2)

# ---------- 习题导航中引用的数值 ----------
Ex1 = sp.Matrix([[1, 2, 0], [2, 2, -2], [0, -2, 3]])
assert sp.expand(quad(Ex1, [x1, x2, x3]) - (x1**2 + 2*x2**2 + 3*x3**2 + 4*x1*x2 - 4*x2*x3)) == 0
assert Ex1.eigenvals() == {-1: 1, 2: 1, 5: 1}
Ex2 = sp.Matrix([[1, t, 1], [t, 2, 0], [1, 0, 1 - t]])
m2 = minors(Ex2)
assert sp.expand(m2[1] - (2 - t**2)) == 0
assert sp.expand(m2[2] - t*(t - 2)*(t + 1)) == 0
assert sp.solveset(t*(t - 2)*(t + 1) > 0, t, sp.S.Reals) == sp.Union(sp.Interval.open(-1, 0), sp.Interval.open(2, sp.oo))
assert sp.solveset(2 - t**2 > 0, t, sp.S.Reals) == sp.Interval.open(-sp.sqrt(2), sp.sqrt(2))
assert sp.Interval.open(-sp.sqrt(2), sp.sqrt(2)).intersect(sp.Union(sp.Interval.open(-1, 0), sp.Interval.open(2, sp.oo))) == sp.Interval.open(-1, 0)

print("ALL OK")
