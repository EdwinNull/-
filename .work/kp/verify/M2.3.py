"""M2.3 方阵的酉相似化简 —— 讲解中全部数值的复算脚本。
运行：python3 .work/kp/verify/M2.3.py   （全部 assert 通过即输出 ALL OK）
"""
import numpy as np
import sympy as sp

I = sp.I
lam = sp.symbols("lam")
s2, s3 = sp.sqrt(2), sp.sqrt(3)


def is_unitary(U):
    return sp.simplify(U.H * U - sp.eye(U.shape[0])) == sp.zeros(U.shape[0])


def sq_sum(M):
    return sp.nsimplify(sp.simplify(sum(sp.Abs(x) ** 2 for x in M)))


def normal(M):
    return sp.simplify(M.H * M - M * M.H) == sp.zeros(M.shape[0])


# K3：教材 2.3 节例 3
A = sp.Matrix([[3, 2], [-1, 1]])
assert sp.expand((A - lam * sp.eye(2)).det() - (lam ** 2 - 4 * lam + 5)) == 0
assert set(A.eigenvals()) == {2 + I, 2 - I}
x1 = sp.Matrix([1 + I, -1])
assert sp.simplify((A - (2 + I) * sp.eye(2)) * x1) == sp.zeros(2, 1)
assert sp.simplify(2 / (1 - I) - (1 + I)) == 0
e1 = x1 / s3
e2 = sp.Matrix([1, 1 - I]) / s3
assert sp.simplify((e1.H * e2)[0]) == 0
# 易错：漏共轭得到的 [1,1+i]ᵀ 与 ε1 方向的内积 = 1·(1-i)+(1+i)(-1) = -2i ≠ 0
assert sp.expand((x1.H * sp.Matrix([1, 1 + I]))[0]) == -2 * I
U = sp.Matrix.hstack(e1, e2)
assert is_unitary(U)
assert sp.simplify(A * e2 - sp.Matrix([5 - 2 * I, -I]) / s3) == sp.zeros(2, 1)
R = sp.simplify(U.H * A * U)
assert R == sp.Matrix([[2 + I, 1 - 2 * I], [0, 2 - I]])
assert sq_sum(A) == 15 and sq_sum(R) == 15
# K8 例：10 < 15，差值 5=|1-2i|^2
assert sp.simplify(sp.Abs(2 + I) ** 2 + sp.Abs(2 - I) ** 2) == 10 and sp.Abs(1 - 2 * I) ** 2 == 5
assert not normal(A)

# K4 要点：普通相似不保持模平方和
assert sq_sum(sp.diag(1, 2)) == 5 and sq_sum(sp.Matrix([[1, 1], [0, 2]])) == 6
# K5 例
assert sq_sum(sp.Matrix([[1, 4], [0, 1]])) == 18

# K6：反例与正规矩阵
S = sp.Matrix([[0, 1], [1, 0]]); K = sp.Matrix([[0, 1], [-1, 0]])
assert normal(S) and normal(K) and not normal(S + K) and S + K == sp.Matrix([[0, 2], [0, 0]])
Jl = sp.Matrix([[3, 1], [0, 3]])
assert (Jl.H * Jl)[0, 0] == 9 and (Jl * Jl.H)[0, 0] == 10
# 易错 5：复对称但不正规的幂零矩阵
Z = sp.Matrix([[1, I], [I, -1]])
assert Z.T == Z and Z ** 2 == sp.zeros(2)
assert sp.expand((Z.H * Z)[0, 1]) == 2 * I and sp.expand((Z * Z.H)[0, 1]) == -2 * I

# K7 例 / 自测题 4：[[1,i],[i,1]]
B = sp.Matrix([[1, I], [I, 1]])
assert sp.expand(B.H * B) == 2 * sp.eye(2) and sp.expand(B * B.H) == 2 * sp.eye(2)
UB = sp.Matrix([[1, 1], [1, -1]]) / s2
assert is_unitary(UB) and sp.simplify(UB.H * B * UB) == sp.diag(1 + I, 1 - I)
assert sq_sum(B) == 4
# K7 要点：[[1,1],[0,2]] 可对角化但不正规，特征向量内积 1
B2 = sp.Matrix([[1, 1], [0, 2]])
assert not normal(B2) and B2 * sp.Matrix([1, 1]) == 2 * sp.Matrix([1, 1])
assert (B2.H * B2)[0, 0] == 1 and (B2 * B2.H)[0, 0] == 2
# K9 理解：Aᴴ[1,0]ᵀ=[1,1]ᵀ
assert B2.H * sp.Matrix([1, 0]) == sp.Matrix([1, 1])

# 典型例题 1
A1 = sp.Matrix([[3, 1], [-1, 1]])
assert sp.factor(A1.charpoly(lam).as_expr()) == (lam - 2) ** 2
assert (A1 - 2 * sp.eye(2)).rank() == 1
U1 = sp.Matrix([[1, 1], [-1, 1]]) / s2
assert is_unitary(U1)
assert sp.simplify(U1.T * A1 * U1) == sp.Matrix([[2, 2], [0, 2]])
assert sq_sum(A1) == 12
assert A1.T * A1 == sp.Matrix([[10, 2], [2, 2]]) and A1 * A1.T == sp.Matrix([[10, -2], [-2, 2]])

# 典型例题 2
A2 = sp.Matrix([[1, -1, 1], [2, -1, 1], [2, -3, 3]])
assert sp.expand(A2.charpoly(lam).as_expr() - (lam ** 3 - 3 * lam ** 2 + 2 * lam)) == 0
assert A2.trace() == 3 and A2.det() == 0
assert [A2.extract(r, r).det() for r in ([0, 1], [0, 2], [1, 2])] == [1, 1, 0]
assert A2 * sp.Matrix([1, 2, 2]) == sp.Matrix([1, 2, 2])
Q = sp.Matrix([[1, 2, 2], [2, 1, -2], [2, -2, 1]]) / 3
assert Q.T == Q and Q.T * Q == sp.eye(3)
assert A2 * Q * 3 == sp.Matrix([[1, -1, 5], [2, 1, 7], [2, -5, 13]])
B3 = Q.T * A2 * Q
assert B3 == sp.Matrix([[1, -1, 5], [0, 1, -1], [0, -1, 1]])
Asub = B3[1:, 1:]
assert set(Asub.eigenvals()) == {0, 2}
V1 = sp.Matrix([[1, 1], [-1, 1]]) / s2
assert sp.simplify(V1.T * Asub * V1) == sp.diag(2, 0)
assert sp.simplify(sp.Matrix([[-1, 5]]) * V1) == sp.Matrix([[-3 * s2, 2 * s2]])
Uf = sp.simplify(Q * sp.diag(1, V1))
assert Uf == sp.Matrix([[sp.Rational(1, 3), 0, 2 * s2 / 3], [sp.Rational(2, 3), s2 / 2, -s2 / 6], [sp.Rational(2, 3), -s2 / 2, -s2 / 6]])
Rf = sp.simplify(Uf.T * A2 * Uf)
assert Rf == sp.Matrix([[1, -3 * s2, 2 * s2], [0, 2, 0], [0, 0, 0]])
assert sq_sum(A2) == 31 and sq_sum(Rf) == 31
# 点评：另一种剥离顺序
V2 = sp.Matrix([[1, -1], [1, 1]]) / s2
Rp = sp.simplify(sp.diag(1, V2).T * B3 * sp.diag(1, V2))
assert Rp == sp.Matrix([[1, 2 * s2, 3 * s2], [0, 0, 0], [0, 0, 2]])
assert sp.simplify(sum(sp.Abs(Rp[i, j]) ** 2 for i in range(3) for j in range(3) if i < j)) == 26
assert not normal(A2)

# 典型例题 3：循环矩阵
C = sp.Matrix([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
G = sp.Matrix([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
assert C.T * C == G and C * C.T == G
assert sp.expand((C - lam * sp.eye(3)).det() - ((1 - lam) ** 3 + 1)) == 0
w = sp.Rational(-1, 2) + s3 / 2 * I
assert sp.expand(w ** 3) == 1 and sp.expand(1 + w + w ** 2) == 0 and sp.expand(sp.conjugate(w) - w ** 2) == 0
Uc = sp.Matrix([[1, 1, 1], [1, w, w ** 2], [1, w ** 2, w ** 4]]) / s3
assert is_unitary(Uc)
D = sp.simplify(sp.expand(Uc.H * C * Uc))
assert sp.simplify(D - sp.diag(2, sp.Rational(1, 2) + s3 / 2 * I, sp.Rational(1, 2) - s3 / 2 * I)) == sp.zeros(3)
assert sp.simplify(4 + sp.Abs(1 + w) ** 2 + sp.Abs(1 + w ** 2) ** 2) == 6 and sq_sum(C) == 6

# 自测题
assert sq_sum(sp.Matrix([[1, 2], [0, 3]])) == 14
M6a = sp.Matrix([[0, 2, 0], [0, 0, 0], [0, 0, 0]]); M6b = sp.Matrix([[0, s2, 0], [0, 0, s2], [0, 0, 0]])
assert sq_sum(M6a) == 4 and sq_sum(M6b) == 4 and M6a.rank() == 1 and M6b.rank() == 2
assert set(sp.Matrix([[0, 1], [-1, 0]]).eigenvals()) == {I, -I}
# 自测题 2 的数值抽查
rng = np.random.default_rng(0)
X = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
Hn = X + X.conj().T
c = 0.7 - 1.3j
Y = Hn + c * np.eye(3)
assert np.allclose(Y.conj().T @ Y, Y @ Y.conj().T)

# 习题导航第 1 题：正规性、特征值、推论验算
E = sp.Matrix([[-s2 * I, -4], [4, s2 * I]])
assert sp.expand(E.H * E) == 18 * sp.eye(2) and sp.expand(E * E.H) == 18 * sp.eye(2)
assert set(E.eigenvals()) == {3 * s2 * I, -3 * s2 * I}
assert sq_sum(E) == 36
UE = sp.Matrix([[1 / s3, 2 / sp.sqrt(6)], [-s2 * I / s3, s2 * I / sp.sqrt(6)]])
assert is_unitary(UE) and sp.simplify(UE.H * E * UE) == sp.diag(3 * s2 * I, -3 * s2 * I)

print("ALL OK")
