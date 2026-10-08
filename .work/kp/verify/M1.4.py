"""M1.4 线性方程组 —— 讲解中全部数值的复算脚本。
运行：python3 .work/kp/verify/M1.4.py   （全部 assert 通过即输出 ALL OK）
"""
import sympy as sp
from sympy import Matrix, Rational as R, symbols, simplify, eye, zeros

lam, a, b, k1, k2, t = symbols("lam a b k1 k2 t")


def rk(M):
    return Matrix(M).rank()


def aug(A, bb):
    return Matrix(A).row_join(Matrix(bb))


# ---------------- K1：矩阵形式的小例（自编） ----------------
A = Matrix([[1, 2, -1], [2, -1, 1]])
bb = Matrix([3, 1])
x0 = Matrix([1, 1, 0])
assert A * x0 == bb
assert x0[0] * A[:, 0] + x0[1] * A[:, 1] + x0[2] * A[:, 2] == bb  # 列向量形式
assert aug(A, bb).shape == (2, 4)

# ---------------- K2：定理 1.4-1 的三种情形（自编） ----------------
A1 = Matrix([[1, 1], [2, 2]])
assert rk(A1) == 1
assert rk(aug(A1, [1, 2])) == 1                     # 有无穷多解
assert rk(aug(A1, [1, 3])) == 2                     # 无解
A2 = Matrix([[1, 1], [1, -1]])
assert rk(A2) == 2 and rk(aug(A2, [1, 3])) == 2
assert A2.solve(Matrix([1, 3])) == Matrix([2, -1])  # 唯一解 [2,-1]
assert A2 * Matrix([2, -1]) == Matrix([1, 3])
# 无穷多解情形的一般解 x1=1-x2：[1-s, s]
s = symbols("s")
assert (A1 * Matrix([1 - s, s]) - Matrix([1, 2])).expand() == zeros(2, 1)

# ---------------- K3：定理 1.4-2、推论（自编） ----------------
A3 = Matrix([[1, 2], [2, 4]])
assert A3.det() == 0 and A3 * Matrix([2, -1]) == zeros(2, 1)
A4 = Matrix([[1, 1, 1], [1, -1, 0]])
assert rk(A4) == 2 < 3
assert A4 * Matrix([1, 1, -2]) == zeros(2, 1)
assert A4.nullspace()[0].T.rank() == 1 and len(A4.nullspace()) == 1

# ---------------- K4：教材 1.4 节例 1 ----------------
A5 = Matrix([[1, 1, 1, 1], [0, 1, 2, 2], [0, -1, a - 3, -2], [3, 2, 1, a]])
b5 = Matrix([0, 1, b, -1])
B5 = A5.row_join(b5)
# 行变换：r4-3r1，再 r3+r2、r4+r2
E = B5.copy()
E[3, :] = E[3, :] - 3 * E[0, :]
assert E[3, :].applyfunc(sp.expand) == Matrix([[0, -1, -2, a - 3, -1]])
E[2, :] = E[2, :] + E[1, :]
E[3, :] = E[3, :] + E[1, :]
assert E.applyfunc(sp.expand) == Matrix([[1, 1, 1, 1, 0], [0, 1, 2, 2, 1], [0, 0, a - 1, 0, b + 1], [0, 0, 0, a - 1, 0]])
assert sp.factor(A5.det()) == (a - 1) ** 2        # |A|=(a-1)^2
assert rk(B5.subs({a: 2, b: 5})) == 4 and rk(A5.subs({a: 2})) == 4
assert rk(A5.subs(a, 1)) == 2 and rk(B5.subs({a: 1, b: 7})) == 3
assert rk(B5.subs({a: 1, b: -1})) == 2
sol = Matrix([-1, 1, 0, 0]) + k1 * Matrix([1, -2, 1, 0]) + k2 * Matrix([1, -2, 0, 1])
assert (A5.subs(a, 1) * sol - b5.subs(b, -1)).applyfunc(sp.expand) == zeros(4, 1)
# 同解方程组的解
assert (Matrix([[1, 1, 1, 1], [0, 1, 2, 2]]) * sol - Matrix([0, 1])).applyfunc(sp.expand) == zeros(2, 1)

# ---------------- K5：解空间 ----------------
Ah = Matrix([[1, 1, 1]])
for v in (Matrix([-1, 1, 0]), Matrix([-1, 0, 1]), Matrix([1, -1, 0]), Matrix([0, 1, -1])):
    assert Ah * v == zeros(1, 1)
assert Matrix.hstack(Matrix([-1, 1, 0]), Matrix([-1, 0, 1])).rank() == 2
assert Matrix.hstack(Matrix([1, -1, 0]), Matrix([0, 1, -1])).rank() == 2
assert Matrix.hstack(Matrix([1, -1, 0]), Matrix([2, -2, 0])).rank() == 1
assert Ah * Matrix([2, -2, 0]) == zeros(1, 1)
# 非齐次解集不是子空间：x1+x2=1
A6 = Matrix([[1, 1]])
assert A6 * Matrix([1, 0]) == Matrix([1]) and A6 * Matrix([0, 1]) == Matrix([1])
assert A6 * (Matrix([1, 0]) + Matrix([0, 1])) == Matrix([2])
# r(A)=n 时 N(A)={0}
assert Matrix([[1, 0], [0, 1]]).nullspace() == []

# ---------------- K6：教材 1.4 节例 2 ----------------
A7 = Matrix([[1, 2, 0, -1], [2, 1, 0, 1], [1, -1, 0, 2]])
assert A7.rref()[0] == Matrix([[1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 0, 0]])
assert A7.rank() == 2
al1, al2 = Matrix([0, 0, 1, 0]), Matrix([-1, 1, 0, 1])
assert A7 * al1 == zeros(3, 1) and A7 * al2 == zeros(3, 1)
assert Matrix.hstack(al1, al2).rank() == 2
# 过程：r2-2r1, r3-r1
M = A7.copy()
M[1, :] = M[1, :] - 2 * M[0, :]
M[2, :] = M[2, :] - M[0, :]
assert M == Matrix([[1, 2, 0, -1], [0, -3, 0, 3], [0, -3, 0, 3]])
# (II) 的一般解与公共解
be1, be2 = Matrix([0, 1, 1, 0]), Matrix([-1, 2, 2, 1])
xII = k1 * be1 + k2 * be2
assert xII == Matrix([-k2, k1 + 2 * k2, k1 + 2 * k2, k2])
res = (A7 * xII).applyfunc(sp.expand)
assert res[0] == sp.expand(-k2 + 2 * (k1 + 2 * k2) - k2) == 2 * k1 + 2 * k2
assert sp.expand(-2 * k2 + (k1 + 2 * k2) + k2) == k1 + k2
assert sp.expand(-k2 - (k1 + 2 * k2) + 2 * k2) == -k1 - k2
assert sp.solve([2 * k1 + 2 * k2, k1 + k2, -k1 - k2], [k1], dict=True) == [{k1: -k2}]
common = (k2 * (-be1 + be2)).applyfunc(sp.expand)
assert common == Matrix([-k2, k2, k2, k2])
assert A7 * Matrix([-1, 1, 1, 1]) == zeros(3, 1)
assert Matrix.hstack(be1, be2).rank() == 2  # (II) 的两个向量线性无关
# 另一种做法：a1*al1+a2*al2 = k1*be1+k2*be2
a1, a2 = symbols("a1 a2")
eqs = list((a1 * al1 + a2 * al2 - xII))
s_ = sp.solve(eqs, [a1, a2, k1], dict=True)
assert s_ == [{a1: k2, a2: k2, k1: -k2}]

# ---------------- K7：性质与定理 1.4-3（小例 x1+x2=1） ----------------
eta, alph = Matrix([1, 0]), Matrix([1, -1])
assert A6 * eta == Matrix([1]) and A6 * alph == zeros(1, 1)
assert A6 * (eta + k1 * alph) == Matrix([1])
assert Matrix.hstack(Matrix([1, 0]), Matrix([0, 1])).T * Matrix([0, 0]) == zeros(2, 1)
assert (Matrix([1, 0]) - Matrix([0, 1])).T * Matrix([1, 1]) == Matrix([0])  # η1-η2=[1,-1] 满足 x1+x2=0

# ---------------- K8：教材 1.4 节例 3 ----------------
s1 = Matrix([2, 4, 0, 8]); s2 = Matrix([3, 0, 3, 3]); s3 = Matrix([2, 1, 0, 1])
c1, c2, c3 = s1 / 2, s2 / 3, s3 / 4
assert c1 == Matrix([1, 2, 0, 4]) and c2 == Matrix([1, 0, 1, 1]) and c3 == Matrix([R(1, 2), R(1, 4), 0, R(1, 4)])
assert c1 - c2 == Matrix([0, 2, -1, 3])
assert c2 - c3 == Matrix([R(1, 2), -R(1, 4), 1, R(3, 4)])
assert Matrix.hstack(c1 - c2, c2 - c3).rank() == 2
# 系数和为 1 的组合：1/2+1/2=1 等
assert R(1, 2) + R(1, 2) == 1 and R(1, 3) * (2 + 1) == 1 and R(1, 4) * (3 + 1) == 1
assert 4 - 2 == 2

# ---------------- 典型例题 1：含参数方程组 ----------------
Aλ = Matrix([[lam, 1, 1], [1, lam, 1], [1, 1, lam]])
bλ = Matrix([1, lam, lam ** 2])
Bλ = Aλ.row_join(bλ)
assert sp.factor(Aλ.det()) == (lam + 2) * (lam - 1) ** 2
# 交换 r1<->r3，再 r2-r1，r3-lam*r1
F = Matrix([Bλ[2, :], Bλ[1, :], Bλ[0, :]])
F[1, :] = F[1, :] - F[0, :]
F[2, :] = F[2, :] - lam * F[0, :]
F = F.applyfunc(sp.expand)
assert F == Matrix([[1, 1, lam, lam ** 2],
                    [0, lam - 1, 1 - lam, lam - lam ** 2],
                    [0, 1 - lam, 1 - lam ** 2, 1 - lam ** 3]]).applyfunc(sp.expand)
assert sp.expand(1 - lam ** 2 - (1 - lam) * (1 + lam)) == 0
assert sp.expand(1 - lam ** 3 - (1 - lam) * (1 + lam + lam ** 2)) == 0
assert sp.expand((1 + lam + lam ** 2) - (-lam)) == sp.expand((lam + 1) ** 2)
assert sp.expand((1 + lam) + 1) == lam + 2
# 唯一解公式
xs = sp.simplify(Aλ.LUsolve(bλ))
assert simplify(xs[0] + (lam + 1) / (lam + 2)) == 0
assert simplify(xs[1] - 1 / (lam + 2)) == 0
assert simplify(xs[2] - (lam + 1) ** 2 / (lam + 2)) == 0
x_0 = Matrix([-R(1, 2), R(1, 2), R(1, 2)])
assert Aλ.subs(lam, 0) * x_0 == bλ.subs(lam, 0)
x_3 = xs.subs(lam, 3)
assert x_3 == Matrix([-R(4, 5), R(1, 5), R(16, 5)])
assert Aλ.subs(lam, 3) * x_3 == bλ.subs(lam, 3)
# λ=1
A_1 = Aλ.subs(lam, 1); b_1 = bλ.subs(lam, 1)
assert rk(A_1) == 1 and rk(A_1.row_join(b_1)) == 1
sol1 = Matrix([1, 0, 0]) + k1 * Matrix([-1, 1, 0]) + k2 * Matrix([-1, 0, 1])
assert (A_1 * sol1 - b_1).applyfunc(sp.expand) == zeros(3, 1)
# λ=-2
A_m = Aλ.subs(lam, -2); b_m = bλ.subs(lam, -2)
assert rk(A_m) == 2 and rk(A_m.row_join(b_m)) == 3
assert b_m == Matrix([1, -2, 4]) and sum(b_m) == 3
# (λ+2)(x1+x2+x3)=1+λ+λ² ：三式相加
assert sp.expand(lam + 1 + 1) == lam + 2 and sp.expand(1 + lam + lam ** 2) == lam ** 2 + lam + 1

# ---------------- 典型例题 2：非齐次通解 ----------------
A8 = Matrix([[1, 2, 0, 1], [2, 4, 1, 1], [3, 6, 1, 2]])
b8 = Matrix([1, 3, 4])
B8 = A8.row_join(b8)
G = B8.copy()
G[1, :] = G[1, :] - 2 * G[0, :]
G[2, :] = G[2, :] - 3 * G[0, :]
assert G == Matrix([[1, 2, 0, 1, 1], [0, 0, 1, -1, 1], [0, 0, 1, -1, 1]])
assert B8.rref()[0] == Matrix([[1, 2, 0, 1, 1], [0, 0, 1, -1, 1], [0, 0, 0, 0, 0]])
assert rk(A8) == 2 and rk(B8) == 2
eta8 = Matrix([1, 0, 1, 0]); a81 = Matrix([-2, 1, 0, 0]); a82 = Matrix([-1, 0, 1, 1])
assert A8 * eta8 == b8 and A8 * a81 == zeros(3, 1) and A8 * a82 == zeros(3, 1)
assert Matrix.hstack(a81, a82).rank() == 2
assert (A8 * (eta8 + k1 * a81 + k2 * a82) - b8).applyfunc(sp.expand) == zeros(3, 1)
# b3 改为 5 则无解
b8b = Matrix([1, 3, 5])
assert rk(A8.row_join(b8b)) == 3
assert A8[2, :] == A8[0, :] + A8[1, :] and b8[2] == b8[0] + b8[1]
# 另一个特解 η+α1 = [-1,1,1,0]
assert A8 * (eta8 + a81) == b8 and eta8 + a81 == Matrix([-1, 1, 1, 0])

# ---------------- 典型例题 3：由已知解求通解（改编） ----------------
A9 = Matrix([[1, 1, 1], [2, 2, 2], [1, 1, 1]])
e1, e2, e3 = Matrix([1, 1, 0]), Matrix([0, 1, 1]), Matrix([1, 0, 1])
b9 = Matrix([2, 4, 2])
assert rk(A9) == 1
for e in (e1, e2, e3):
    assert A9 * e == b9
d1, d2 = e1 - e2, e1 - e3
assert d1 == Matrix([1, 0, -1]) and d2 == Matrix([0, 1, -1])
assert A9 * d1 == zeros(3, 1) and A9 * d2 == zeros(3, 1)
assert Matrix.hstack(d1, d2).rank() == 2
gen9 = e1 + k1 * d1 + k2 * d2
assert gen9 == Matrix([1 + k1, 1 + k2, -k1 - k2])
assert (A9 * gen9 - b9).applyfunc(sp.expand) == zeros(3, 1)
assert A9 * (e1 + e2 + e3) == 3 * b9 and (e1 + e2 + e3) / 3 == Matrix([R(2, 3)] * 3)
assert A9 * ((e1 + e2 + e3) / 3) == b9
assert sum(e1 + e2 + e3) == 6

# ---------------- 典型例题 4：(A-λI)x=0 ----------------
J = Matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
assert sp.factor((J - lam * eye(3)).det()) == -lam ** 2 * (lam - 3)
assert sp.expand((J - lam * eye(3)).det()) == -lam ** 3 + 3 * lam ** 2
assert J.eigenvals() == {0: 2, 3: 1}
v1, v2 = Matrix([-1, 1, 0]), Matrix([-1, 0, 1])
assert J * v1 == zeros(3, 1) and J * v2 == zeros(3, 1) and rk(Matrix.hstack(v1, v2)) == 2
J3 = J - 3 * eye(3)
assert J3 == Matrix([[-2, 1, 1], [1, -2, 1], [1, 1, -2]])
assert J3.rank() == 2
assert J3.rref()[0] == Matrix([[1, 0, -1], [0, 1, -1], [0, 0, 0]])
assert J3 * Matrix([1, 1, 1]) == zeros(3, 1)
# 过程：r1<->r2；r2+2r1，r3-r1；r3+r2
P = Matrix([[1, -2, 1], [-2, 1, 1], [1, 1, -2]])
P[1, :] = P[1, :] + 2 * P[0, :]
P[2, :] = P[2, :] - P[0, :]
assert P == Matrix([[1, -2, 1], [0, -3, 3], [0, 3, -3]])

# ---------------- 易错点中的数值例 ----------------
# 1) r(A)<n 但无解：x1+x2=1, 2x1+2x2=3
A10 = Matrix([[1, 1], [2, 2]])
assert rk(A10) == 1 and rk(aug(A10, [1, 3])) == 2
# 2) 自由未知量选取：x1-x2=0, x3=0
A11 = Matrix([[1, -1, 0], [0, 0, 1]])
assert rk(A11) == 2 and len(A11.nullspace()) == 1
assert A11 * Matrix([t, t, 0]) == zeros(2, 1)
# 取 (x2,x3) 为自由未知量：x3 只能为 0，不是自由的
assert A11[:, [0]].rank() == 1 and Matrix.hstack(A11[:, 0], A11[:, 1]).rank() == 1  # x1,x2 的列线性相关
assert A11[:, [0, 2]].rank() == 2  # x1,x3 列无关 -> 以 x2 为自由未知量
# 5) η1+η2 不是解：x1+x2=1
assert A6 * (Matrix([1, 0]) + Matrix([0, 1])) == Matrix([2])
assert A6 * ((Matrix([1, 0]) + Matrix([0, 1])) / 2) == Matrix([1])

# ---------------- 自测题 ----------------
# 3) 5x7, r=3：dim N(A)=7-3=4，基础解系含 4 个向量
assert 7 - 3 == 4
# 4) 计算题：x1-x2+x3=2, x1+x2-x3=0
A12 = Matrix([[1, -1, 1], [1, 1, -1]]); b12 = Matrix([2, 0])
assert rk(A12) == 2 and rk(aug(A12, b12)) == 2
e12 = Matrix([1, -1, 0]); a12 = Matrix([0, 1, 1])
assert A12 * e12 == b12 and A12 * a12 == zeros(2, 1)
assert (A12 * (e12 + t * a12) - b12).applyfunc(sp.expand) == zeros(2, 1)
assert A12.row_join(b12).rref()[0] == Matrix([[1, 0, 0, 1], [0, 1, -1, -1]])
# 6) 齐次方程组 [[1,1,1],[1,2,k],[1,4,k^2]]x=0
kk = symbols("kk")
V = Matrix([[1, 1, 1], [1, 2, kk], [1, 4, kk ** 2]])
assert sp.factor(V.det()) == (kk - 1) * (kk - 2)
assert V.subs(kk, 1).rank() == 2 and V.subs(kk, 2).rank() == 2 and V.subs(kk, 3).rank() == 3
# 2) 判断：r(B) >= r(A)
assert aug(A10, [1, 3]).rank() >= A10.rank()

# 例题 4 点评：幂零矩阵，λ=0 二重根但零空间 1 维
Nn = Matrix([[0, 1], [0, 0]])
assert sp.expand((Nn - lam * eye(2)).det()) == lam ** 2
assert len(Nn.nullspace()) == 1 and Nn * Matrix([1, 0]) == zeros(2, 1)
# 自测题 2 的反例：x1+x2+x3=0, x1+x2+x3=1
A13 = Matrix([[1, 1, 1], [1, 1, 1]])
assert rk(A13) == 1 and rk(aug(A13, [0, 1])) == 2 and 2 < 3

print("ALL OK")
