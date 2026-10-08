"""M2.4 实方阵的正交相似化简 —— 讲解中全部数值的复算脚本。
运行：python3 .work/kp/verify/M2.4.py   （全部 assert 通过即输出 ALL OK）
"""
import sympy as sp

R = sp.Rational
s2 = sp.sqrt(2)
lam = sp.symbols("lam")


def is_orth(Q):
    return (Q.T * Q - sp.eye(Q.shape[0])).applyfunc(sp.simplify) == sp.zeros(*Q.shape)


def hh(x, y):
    """式 (2.4-4)：把 x 变为 y 的 Householder 矩阵"""
    v = x - y
    return sp.eye(len(x)) - 2 * v * v.T / (v.T * v)[0]


def simp(M):
    return M.applyfunc(lambda t: sp.nsimplify(sp.radsimp(sp.simplify(t))))


def is_hess(M):
    n = M.shape[0]
    return all(sp.simplify(M[i, j]) == 0 for i in range(n) for j in range(n) if i > j + 1)


# ---------- K1：教材 2.4 节例 1 ----------
A = sp.Matrix([[1, 1, -1], [1, 1, 1], [0, -1, 2]])
assert sp.factor((A - lam * sp.eye(3)).det()) == sp.factor((2 - lam) * (1 - lam) ** 2)
assert (A - sp.eye(3)).rank() == 2           # λ=1 的几何重数 1，A 不可对角化
assert (A - 2 * sp.eye(3)) * sp.Matrix([1, 0, -1]) == sp.zeros(3, 1)
h = s2 / 2
Q = sp.Matrix([[h, 0, h], [0, 1, 0], [-h, 0, h]])
assert is_orth(Q)
assert simp(A * Q) == sp.Matrix([[s2, 1, 0], [0, 1, s2], [-s2, -1, s2]])
Rm = sp.Matrix([[2, s2, -1], [0, 1, s2], [0, 0, 1]])
assert simp(Q.T * A * Q) == Rm
# 反例：旋转矩阵特征值 ±i
Rot = sp.Matrix([[0, -1], [1, 0]])
assert set(Rot.eigenvals()) == {sp.I, -sp.I}

# ---------- 典型例题 1：自编实 Schur 化简 ----------
A = sp.Matrix([[0, 2, 0], [0, 2, 2], [1, -1, 3]])
assert sp.factor(A.charpoly(lam).as_expr()) == sp.factor((lam - 2) ** 2 * (lam - 1))
assert sp.expand(A.charpoly(lam).as_expr()) == lam**3 - 5*lam**2 + 8*lam - 4
assert (A - 2 * sp.eye(3)).rank() == 2
assert A * sp.Matrix([1, 1, 0]) == 2 * sp.Matrix([1, 1, 0])
r = 1 / s2
Q2 = sp.Matrix([[r, r, 0], [r, -r, 0], [0, 0, 1]])
assert is_orth(Q2)
assert simp(A * Q2) == sp.Matrix([[s2, -s2, 0], [s2, -s2, 2], [0, s2, 3]])
B = simp(Q2.T * A * Q2)
assert B == sp.Matrix([[2, -2, s2], [0, 0, -s2], [0, s2, 3]])
A1 = B[1:, 1:]
assert sp.expand(A1.charpoly(lam).as_expr()) == sp.expand((lam - 1) * (lam - 2))
assert simp(A1 * sp.Matrix([s2, -1])) == sp.Matrix([s2, -1])     # λ=1 的特征向量
s3 = sp.sqrt(3)
U1 = sp.Matrix([[s2 / s3, 1 / s3], [-1 / s3, s2 / s3]])
assert is_orth(U1)
assert simp(U1.T * A1 * U1) == sp.Matrix([[1, -2 * s2], [0, 2]])
Qf = simp(Q2 * sp.diag(1, U1))
assert Qf == sp.Matrix([[s2 / 2, s3 / 3, sp.sqrt(6) / 6],
                        [s2 / 2, -s3 / 3, -sp.sqrt(6) / 6],
                        [0, -s3 / 3, sp.sqrt(6) / 3]])
assert is_orth(Qf)
Rf = simp(Qf.T * A * Qf)
assert Rf == sp.Matrix([[2, -sp.sqrt(6), 0], [0, 1, -2 * s2], [0, 0, 2]])
# 首行验算：-2·(√2/√3) + √2·(-1/√3) = -√6，-2·(1/√3)+√2·(√2/√3)=0
assert sp.simplify(-2 * s2 / s3 + s2 * (-1 / s3) + sp.sqrt(6)) == 0
assert sp.simplify(-2 / s3 + s2 * s2 / s3) == 0

# ---------- K2/K3：Hessenberg 例与分块 ----------
H5 = sp.Matrix([[-1, 2, 3, 1, -1], [1, -1, 4, -7, 0], [0, 1, 2, -1, 2],
                [0, 0, 0, 1, -1], [0, 0, 0, 2, 3]])
assert is_hess(H5) and H5[3, 2] == 0
assert sp.expand(H5.charpoly(lam).as_expr()) == sp.expand(
    H5[:3, :3].charpoly(lam).as_expr() * H5[3:, 3:].charpoly(lam).as_expr())
# 性质 1 的小例：Hessenberg × 上三角
Hs = sp.Matrix([[1, 2, -1], [3, -1, 4], [0, 1, 2]])
T = sp.Matrix([[1, 1, 1], [0, 2, 1], [0, 0, 3]])
assert is_hess(Hs * T) and is_hess(T * Hs)
assert Hs * T == sp.Matrix([[1, 5, 0], [3, 1, 14], [0, 2, 7]])
assert T * Hs == sp.Matrix([[4, 2, 5], [6, -1, 10], [0, 3, 6]])

# ---------- K4/K5：Householder 矩阵 ----------
# 教材 2.4 节例 2
x = sp.Matrix([-2, 2, 1]); y = sp.Matrix([3, 0, 0])
H = hh(x, y)
assert H == sp.Matrix([[-R(2, 3), R(2, 3), R(1, 3)], [R(2, 3), R(11, 15), -R(2, 15)], [R(1, 3), -R(2, 15), R(14, 15)]])
assert H * x == y and H.T == H and H * H == sp.eye(3)
assert ((x - y).T * (x - y))[0] == 30
# Householder 的特征结构：Hω=-ω，ω⊥z 时 Hz=z，det H=-1
w = (x - y) / sp.sqrt(30)
assert simp(H * w) == simp(-w) and H.det() == -1
# 中点 (x+y)/2 被保持
assert H * (x + y) / 2 == (x + y) / 2
# 平面 Householder 的小例：ω=[3/5,4/5]
w2 = sp.Matrix([R(3, 5), R(4, 5)])
H2 = sp.eye(2) - 2 * w2 * w2.T
assert H2 == sp.Matrix([[R(7, 25), -R(24, 25)], [-R(24, 25), -R(7, 25)]])
assert H2.det() == -1

# ---------- 典型例题 2：3 阶非对称矩阵化 Hessenberg，比较 σ 的两种取法 ----------
A = sp.Matrix([[1, 2, 3], [4, 1, 0], [3, 2, 1]])
beta = sp.Matrix([4, 3])
assert (beta.T * beta)[0] == 25
Hm = hh(beta, sp.Matrix([-5, 0]))            # σ=-1
assert Hm == sp.Matrix([[-R(4, 5), -R(3, 5)], [-R(3, 5), R(4, 5)]])
assert beta - sp.Matrix([-5, 0]) == sp.Matrix([9, 3])
Q1 = sp.diag(1, Hm)
assert Q1 * A == sp.Matrix([[1, 2, 3], [-5, -2, -R(3, 5)], [0, 1, R(4, 5)]])
Bm = Q1 * A * Q1
assert Bm == sp.Matrix([[1, -R(17, 5), R(6, 5)], [-5, R(49, 25), R(18, 25)], [0, -R(32, 25), R(1, 25)]])
assert is_hess(Bm) and Bm.trace() == A.trace() == 3 and Bm.det() == A.det()
Hp = hh(beta, sp.Matrix([5, 0]))             # σ=+1
assert Hp == -Hm and beta - sp.Matrix([5, 0]) == sp.Matrix([-1, 3])
Bp = sp.diag(1, Hp) * A * sp.diag(1, Hp)
assert Bp == sp.Matrix([[1, R(17, 5), -R(6, 5)], [5, R(49, 25), R(18, 25)], [0, -R(32, 25), R(1, 25)]])
D = sp.diag(1, -1, -1)
assert Bp == D * Bm * D
assert A.det() == 8

# ---------- 教材 2.4 节例 3 ----------
A = sp.Matrix([[1, 0, 1], [0, 2, -1], [1, -1, 1]])
x = sp.Matrix([1, 0, 1]); y = sp.Matrix([1, 1, 0])
H = hh(x, y)
assert H == sp.Matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
assert H * A * H.T == sp.Matrix([[1, 1, 0], [1, 1, -1], [0, -1, 2]])

# ---------- 典型例题 3：4 阶实对称矩阵三对角化 ----------
A = sp.Matrix([[1, 1, 2, 2], [1, 4, 4, 3], [2, 4, 1, 0], [2, 3, 0, 0]])
assert A.T == A
beta = sp.Matrix([1, 2, 2])
Q2 = hh(beta, sp.Matrix([-3, 0, 0]))
assert Q2 == sp.Matrix([[-1, -2, -2], [-2, 2, -1], [-2, -1, 2]]) / 3
assert beta - sp.Matrix([-3, 0, 0]) == sp.Matrix([4, 2, 2])
assert Q2 * beta == sp.Matrix([-3, 0, 0])
A1 = A[1:, 1:]
assert A1 * Q2 == sp.Matrix([[-6, -1, -2], [-2, -2, -3], [-1, -2, -2]])
A2 = Q2 * A1 * Q2
assert A2 == sp.Matrix([[4, 3, 4], [3, 0, 0], [4, 0, 1]])
P1 = sp.diag(1, Q2)
B1 = P1 * A * P1
assert B1 == sp.Matrix([[1, -3, 0, 0], [-3, 4, 3, 4], [0, 3, 0, 0], [0, 4, 0, 1]])
b2 = sp.Matrix([3, 4])
Q3 = hh(b2, sp.Matrix([-5, 0]))
assert Q3 == sp.Matrix([[-R(3, 5), -R(4, 5)], [-R(4, 5), R(3, 5)]])
assert b2 - sp.Matrix([-5, 0]) == sp.Matrix([8, 4])
A3 = A2[1:, 1:]
assert Q3 * A3 * Q3 == sp.Matrix([[R(16, 25), -R(12, 25)], [-R(12, 25), R(9, 25)]])
P2 = sp.diag(1, 1, Q3)
Tm = P2 * B1 * P2
assert Tm == sp.Matrix([[1, -3, 0, 0], [-3, 4, -5, 0], [0, -5, R(16, 25), -R(12, 25)], [0, 0, -R(12, 25), R(9, 25)]])
Qt = P2 * P1
assert Qt == sp.Matrix([[1, 0, 0, 0], [0, -R(1, 3), -R(2, 3), -R(2, 3)],
                        [0, R(14, 15), -R(2, 15), -R(1, 3)], [0, R(2, 15), -R(11, 15), R(2, 3)]])
assert Qt * Qt.T == sp.eye(4) and Qt.T != Qt and Qt * A * Qt.T == Tm
assert sp.expand(A.charpoly(lam).as_expr()) == sp.expand(Tm.charpoly(lam).as_expr())
assert A.trace() == Tm.trace() == 6
assert sp.expand(Tm.charpoly(lam).as_expr()) == lam**4 - 6*lam**3 - 25*lam**2 + 39*lam - 9
assert Q3[:, 1] * Q3[:, 1].T == sp.Matrix([[R(16, 25), -R(12, 25)], [-R(12, 25), R(9, 25)]])

# ---------- K8：Householder 矩阵之积 ----------
e1 = sp.Matrix([1, 0, 0]); e2 = sp.Matrix([0, 1, 0])
Ha = sp.eye(3) - 2 * e1 * e1.T; Hb = sp.eye(3) - 2 * e2 * e2.T
assert Ha * Hb == Hb * Ha == sp.diag(-1, -1, 1)
assert (sp.eye(3) - Ha * Hb).rank() == 2     # Householder 矩阵的 I-H 秩为 1
Hc = sp.eye(2) - 2 * sp.Matrix([1, 0]) * sp.Matrix([[1, 0]])
Hd = hh(sp.Matrix([1, 0]), sp.Matrix([0, 1]))
assert Hc * Hd != Hd * Hc and (Hc * Hd).T == Hd * Hc

# ---------- 习题导航（与现有解答一致） ----------
H = hh(sp.Matrix([1, 1, 1, 1]), sp.Matrix([-2, 0, 0, 0]))
assert H == -R(1, 6) * sp.Matrix([[3, 3, 3, 3], [3, -5, 1, 1], [3, 1, -5, 1], [3, 1, 1, -5]])
A = sp.Matrix([[2, 2, 1], [1, 2, 2], [2, 1, 2]])
Q = sp.Matrix([[1, 0, 0], [0, sp.sqrt(5) / 5, 2 * sp.sqrt(5) / 5], [0, 2 * sp.sqrt(5) / 5, -sp.sqrt(5) / 5]])
assert is_orth(Q)
assert simp(Q * A * Q.T) == sp.Matrix([[2, 4 * sp.sqrt(5) / 5, 3 * sp.sqrt(5) / 5],
                                       [sp.sqrt(5), R(16, 5), R(2, 5)], [0, R(7, 5), R(4, 5)]])
# 习题 4：P = I - ωωᵀ，H = 2P - I
w = sp.Matrix([R(2, 3), R(1, 3), R(2, 3)])
P = sp.eye(3) - w * w.T
assert P.T == P and P * P == P and sp.eye(3) - 2 * w * w.T == 2 * P - sp.eye(3)

# ---------- 自测题 ----------
x = sp.Matrix([0, 3, 4]); y = sp.Matrix([5, 0, 0])
H = hh(x, y)
assert ((x - y).T * (x - y))[0] == 50
assert H == sp.Matrix([[0, R(3, 5), R(4, 5)], [R(3, 5), R(16, 25), -R(12, 25)], [R(4, 5), -R(12, 25), R(9, 25)]])
assert H * x == y
M = sp.Matrix([[1, 2, 0, 0], [3, 4, 5, 0], [0, 0, 6, 7], [0, 0, 8, 9]])
assert is_hess(M) and M[2, 1] == 0
ev = set(M.eigenvals())
assert ev == {R(5, 2) + sp.sqrt(33) / 2, R(5, 2) - sp.sqrt(33) / 2,
              R(15, 2) + sp.sqrt(233) / 2, R(15, 2) - sp.sqrt(233) / 2}

print("ALL OK")
