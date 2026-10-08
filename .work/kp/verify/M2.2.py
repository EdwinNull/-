"""M2.2 Cayley-Hamilton 定理 —— 讲解中全部数值的复算脚本。
运行：python3 .work/kp/verify/M2.2.py   （全部 assert 通过即输出 ALL OK）
"""
import sympy as sp

lam = sp.symbols("lam")
I3 = sp.eye(3)


def mat_poly(coeffs_high_to_low, A):
    """按降幂系数计算方阵多项式（Horner）。"""
    n = A.shape[0]
    R = sp.zeros(n, n)
    for c in coeffs_high_to_low:
        R = R * A + c * sp.eye(n)
    return R


def is_zero(M):
    return M == sp.zeros(*M.shape)


def minpoly_by_test(A):
    """按 (2.2-8) 的形式逐个试指数，返回 {特征值: 最小指数}。"""
    n = A.shape[0]
    ev = A.eigenvals()
    res = {}
    for l0 in ev:
        for r in range(1, ev[l0] + 1):
            M = sp.eye(n)
            for l1 in ev:
                e = r if l1 == l0 else ev[l1]
                M = M * (A - l1 * sp.eye(n)) ** e
            if is_zero(M):
                res[l0] = r
                break
    return res


# ---------- K1 方阵多项式的小例 ----------
B = sp.Matrix([[1, 2], [0, -1]])
assert B ** 2 == sp.eye(2)
assert B ** 2 + sp.eye(2) == 2 * sp.eye(2)

# ---------- K2 Jordan 块的多项式：J=J_3(2)，g(λ)=λ^4 ----------
J3 = sp.Matrix([[2, 1, 0], [0, 2, 1], [0, 0, 2]])
g = lam ** 4
gJ = sp.Matrix([[16, 32, 24], [0, 16, 32], [0, 0, 16]])
assert g.subs(lam, 2) == 16 and sp.diff(g, lam).subs(lam, 2) == 32
assert sp.diff(g, lam, 2).subs(lam, 2) / 2 == 24
assert J3 ** 4 == gJ

# ---------- 教材 2.2 节例 1（Jordan 标准形法） ----------
A = sp.Matrix([[1, 1, -1], [1, 1, 1], [0, -1, 2]])
fA = (A - lam * I3).det()
assert sp.expand(fA - (2 - lam) * (1 - lam) ** 2) == 0
assert (A - 2 * I3) * sp.Matrix([1, 0, -1]) == sp.zeros(3, 1)
x2 = sp.Matrix([-1, 1, 1]); x3 = sp.Matrix([0, 0, 1])
assert (A - I3) * x2 == sp.zeros(3, 1)
assert (A - I3) * x3 == x2
P = sp.Matrix([[1, -1, 0], [0, 1, 0], [-1, 1, 1]])
assert P.inv() == sp.Matrix([[1, 1, 0], [0, 1, 0], [1, 0, 1]])
assert P.inv() * A * P == sp.Matrix([[2, 0, 0], [0, 1, 1], [0, 0, 1]])
gpoly = 2 * lam ** 5 - 3 * lam ** 4 - lam ** 3 + 2 * lam - 1
dg = sp.diff(gpoly, lam)
assert gpoly.subs(lam, 2) == 11 and gpoly.subs(lam, 1) == -1 and dg.subs(lam, 1) == -3
gA = sp.Matrix([[14, 12, 3], [-3, -1, -3], [-15, -12, -4]])
assert P * sp.Matrix([[11, 0, 0], [0, -1, -3], [0, 0, -1]]) * P.inv() == gA
assert 2 * A ** 5 - 3 * A ** 4 - A ** 3 + 2 * A - I3 == gA

# ---------- 零化多项式正文例 ----------
A2 = sp.Matrix([[1, 1], [0, 2]])
assert A2 ** 2 == sp.Matrix([[1, 3], [0, 4]])
assert is_zero(A2 ** 2 - 3 * A2 + 2 * sp.eye(2))

# ---------- 教材 2.2 节例 2（带余除法） ----------
q, r = sp.div(sp.Poly(gpoly, lam), sp.Poly(sp.expand((2 - lam) * (1 - lam) ** 2), lam))
assert r.as_expr() == 15 * lam ** 2 - 33 * lam + 17
a0, a1, a2 = sp.symbols("a0 a1 a2")
sol = sp.solve([4 * a0 + 2 * a1 + a2 - 11, a0 + a1 + a2 + 1, 2 * a0 + a1 + 3], [a0, a1, a2])
assert sol == {a0: 15, a1: -33, a2: 17}
assert A ** 2 == sp.Matrix([[2, 3, -2], [2, 1, 2], [-1, -3, 3]])
assert 15 * A ** 2 - 33 * A + 17 * I3 == gA
# 商式（讲解中写出）
assert q.as_expr() == sp.expand(-2 * lam ** 2 - 5 * lam - 9)

# ---------- 教材 2.2 节例 3（求逆） ----------
assert sp.expand(fA) == -lam ** 3 + 4 * lam ** 2 - 5 * lam + 2
assert is_zero(-A ** 3 + 4 * A ** 2 - 5 * A + 2 * I3)
assert A.det() == 2
Ainv = sp.Rational(1, 2) * sp.Matrix([[3, -1, 2], [-2, 2, -2], [-1, 1, 0]])
assert (A ** 2 - 4 * A + 5 * I3) / 2 == Ainv and A * Ainv == I3

# ---------- K9 小例：最小多项式次数低于特征多项式 ----------
C = sp.Matrix([[2, 1, 0], [0, 2, 0], [0, 0, 2]])
assert not is_zero(C - 2 * I3) and is_zero((C - 2 * I3) ** 2)
assert minpoly_by_test(C) == {2: 2}

# ---------- 教材 2.2 节例 4 ----------
Jb = sp.diag(sp.Matrix([[-2, 1], [0, -2]]),
             sp.Matrix([[-2, 1, 0], [0, -2, 1], [0, 0, -2]]),
             sp.Matrix([[3]]), sp.Matrix([[3, 1], [0, 3]]))
assert sp.factor((Jb - lam * sp.eye(8)).det()) == sp.factor((-2 - lam) ** 5 * (3 - lam) ** 3)
assert minpoly_by_test(Jb) == {-2: 3, 3: 2}
E8 = sp.eye(8)
assert not is_zero((Jb + 2 * E8) ** 2 * (Jb - 3 * E8) ** 2)
assert not is_zero((Jb + 2 * E8) ** 3 * (Jb - 3 * E8))
assert is_zero((Jb + 2 * E8) ** 3 * (Jb - 3 * E8) ** 2)

# ---------- 分块对角矩阵正文例 ----------
A1 = sp.Matrix([[2, 1], [0, 2]]); A2b = sp.Matrix([[1, 1], [-2, 4]])
assert sp.factor(A2b.charpoly(lam).as_expr()) == sp.factor((lam - 3) * (lam - 2))
Ab = sp.diag(A1, A2b); E4 = sp.eye(4)
assert is_zero((Ab - 3 * E4) * (Ab - 2 * E4) ** 2)
assert not is_zero((Ab - 3 * E4) * (Ab - 2 * E4))
assert minpoly_by_test(Ab) == {2: 2, 3: 1}

# ---------- 教材 2.2 节例 5 ----------
assert sp.factor(lam ** 3 - 3 * lam ** 2 - lam + 3) == sp.factor((lam + 1) * (lam - 1) * (lam - 3))

# ---------- K12 小例 ----------
N = sp.Matrix([[1, 1], [0, 1]])
assert not is_zero(N - sp.eye(2)) and is_zero((N - sp.eye(2)) ** 2)

# ---------- 典型例题 1：A^100 ----------
T = sp.Matrix([[3, 1], [-1, 1]])
assert sp.expand(T.charpoly(lam).as_expr()) == lam ** 2 - 4 * lam + 4
a, b = sp.symbols("a b")
s = sp.solve([2 * a + b - 2 ** 100, a - 100 * 2 ** 99], [a, b])
assert s[a] == 100 * 2 ** 99 and s[b] == -198 * 2 ** 99
T100 = 2 ** 99 * sp.Matrix([[102, 100], [-100, -98]])
assert T ** 100 == T100
assert 2 ** 99 * (100 * T - 198 * sp.eye(2)) == T100

# ---------- 典型例题 2：最小多项式、求逆、可对角化 ----------
S = sp.Matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
assert sp.factor((S - lam * I3).det()) == sp.factor(-(lam - 2) * (lam + 1) ** 2)
assert S ** 2 == sp.Matrix([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
assert S ** 2 == S + 2 * I3
assert is_zero((S - 2 * I3) * (S + I3))
assert not is_zero(S - 2 * I3) and not is_zero(S + I3)
Sinv = sp.Rational(1, 2) * sp.Matrix([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
assert (S - I3) / 2 == Sinv and S * Sinv == I3
assert is_zero(S ** 3 - 3 * S - 2 * I3)
assert (S ** 2 - 3 * I3) / 2 == Sinv
assert S.is_diagonalizable()

# ---------- 典型例题 3：分块对角矩阵的最小多项式 ----------
M1 = sp.Matrix([[1, 1], [-1, 3]]); M2 = sp.Matrix([[2, 0], [0, 5]])
assert sp.expand(M1.charpoly(lam).as_expr()) == lam ** 2 - 4 * lam + 4
assert M1 - 2 * sp.eye(2) == sp.Matrix([[-1, 1], [-1, 1]])
assert is_zero((M1 - 2 * sp.eye(2)) ** 2)
M = sp.diag(M1, M2)
assert sp.factor(M.charpoly(lam).as_expr()) == sp.factor((lam - 2) ** 3 * (lam - 5))
assert minpoly_by_test(M) == {2: 2, 5: 1}
Pj, Jj = M.jordan_form()
assert sorted([Jj[i, i] for i in range(4)]) == [2, 2, 2, 5]
assert (M - 2 * sp.eye(4)).rank() == 2  # 几何重数 4-2=2：两个 λ=2 的块
assert not M.is_diagonalizable()

# ---------- 自测题 ----------
# 1. A=I 的零化多项式 (λ-1)(λ-2)，2 不是特征值
assert is_zero((sp.eye(2) - sp.eye(2)) * (sp.eye(2) - 2 * sp.eye(2)))
# 3. 求逆
Q = sp.Matrix([[1, 2], [2, 1]])
assert sp.expand(Q.charpoly(lam).as_expr()) == lam ** 2 - 2 * lam - 3
assert (Q - 2 * sp.eye(2)) / 3 == Q.inv() == sp.Rational(1, 3) * sp.Matrix([[-1, 2], [2, -1]])
# 4. 两个 4 阶幂零 Jordan 矩阵：特征多项式、最小多项式相同但不相似
J20 = sp.Matrix([[0, 1], [0, 0]])
X = sp.diag(J20, J20); Y = sp.diag(J20, sp.zeros(2, 2))
assert X.charpoly(lam) == Y.charpoly(lam)
assert is_zero(X ** 2) and is_zero(Y ** 2) and not is_zero(X) and not is_zero(Y)
assert X.rank() == 2 and Y.rank() == 1
# 6. 例 4 的 J：最小多项式次数 5，特征多项式次数 8
assert sum(minpoly_by_test(Jb).values()) == 5

# ---------- 习题导航中引用的数值（与现有解答一致） ----------
H = sp.Matrix([[2, -1, -2], [-1, 2, 2], [0, 0, 1]])
gH = H ** 8 - 9 * H ** 6 + H ** 4 - 3 * H ** 3 + 4 * H ** 2 + I3
assert gH == sp.Matrix([[16, -21, -42], [-21, 16, 42], [0, 0, -5]])
rem = sp.rem(sp.Poly(lam ** 8 - 9 * lam ** 6 + lam ** 4 - 3 * lam ** 3 + 4 * lam ** 2 + 1, lam),
             sp.Poly(sp.expand((lam - 1) ** 2 * (lam - 3)), lam))
assert rem.as_expr() == 32 * lam ** 2 - 107 * lam + 70
B2 = sp.Matrix([[3, 1, 1, 1], [-4, -1, -1, 1], [0, 0, 2, 1], [0, 0, -1, 0]])
assert minpoly_by_test(B2) == {1: 4}
B1 = sp.Matrix([[3, 1, 0, 0], [-4, -1, 0, 0], [0, 0, 2, 1], [0, 0, -1, 0]])
assert minpoly_by_test(B1) == {1: 2}
assert sp.factor(lam ** 3 - 2 * lam ** 2 - 5 * lam + 6) == sp.factor((lam - 1) * (lam - 3) * (lam + 2))
Bm = sp.Matrix([[0, 1, -1], [-4, 4, -2], [-2, 1, 1]])
Cm = sp.Matrix([[0, -1, -1], [-3, -1, -2], [7, 5, 6]])
Dm = sp.Matrix([[-3, 3, -2], [-7, 6, -3], [1, -1, 2]])
for X3 in (Bm, Cm, Dm):
    assert sp.factor(X3.charpoly(lam).as_expr()) == sp.factor((lam - 1) * (lam - 2) ** 2)
assert minpoly_by_test(Bm) == {1: 1, 2: 1}
assert minpoly_by_test(Cm) == {1: 1, 2: 2} and minpoly_by_test(Dm) == {1: 1, 2: 2}
assert (Bm - 2 * I3).rank() == 1 and (Cm - 2 * I3).rank() == 2 and (Dm - 2 * I3).rank() == 2

print("ALL OK")
