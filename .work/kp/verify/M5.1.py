"""M5.1 方阵的三角分解 —— 讲解中全部数值的复算脚本。
运行：python3 .work/kp/verify/M5.1.py
"""
import sympy as sp

# 书页 108 的消元
A = sp.Matrix([[1, 2, 1], [2, 2, 3], [-1, -3, 0]])
b = sp.Matrix([0, 3, 2])
assert A.solve(b) == sp.Matrix([1, -1, 1])
L1 = sp.Matrix([[1, 0, 0], [-2, 1, 0], [1, 0, 1]])
L2 = sp.Matrix([[1, 0, 0], [0, 1, 0], [0, sp.Rational(-1, 2), 1]])
assert L1.inv() == sp.Matrix([[1, 0, 0], [2, 1, 0], [-1, 0, 1]])
assert L2.inv() == sp.Matrix([[1, 0, 0], [0, 1, 0], [0, sp.Rational(1, 2), 1]])
L = L1.inv() * L2.inv()
assert L == sp.Matrix([[1, 0, 0], [2, 1, 0], [-1, sp.Rational(1, 2), 1]])
R = sp.Matrix([[1, 2, 1], [0, -2, 1], [0, 0, sp.Rational(1, 2)]])
assert L * R == A
# 第二行减两倍第一行，右端必须是 3 而不是 0
aug = sp.Matrix([[1, 2, 1, 0], [2, 2, 3, 3], [-1, -3, 0, 2]])
aug[1, :] -= 2 * aug[0, :]
aug[2, :] += aug[0, :]
assert aug[1, :] == sp.Matrix([[0, -2, 1, 3]])
aug[2, :] -= sp.Rational(1, 2) * aug[1, :]
assert aug[2, :] == sp.Matrix([[0, 0, sp.Rational(1, 2), sp.Rational(1, 2)]])

# 例 1
A1 = sp.Matrix([[2, 2, -5], [1, -3, 1], [1, 5, 2]])
assert A1[0, 0] == 2
assert A1[:2, :2].det() == -8
L1e = sp.Matrix([[1, 0, 0], [sp.Rational(1, 2), 1, 0], [sp.Rational(1, 2), -1, 1]])
R1e = sp.Matrix([[2, 2, -5], [0, -4, sp.Rational(7, 2)], [0, 0, 8]])
assert L1e * R1e == A1

# 例 2
A2 = sp.Matrix([[0, 1, -1, 2], [1, 0, 0, 1], [2, -1, 0, 1], [1, 3, 0, 1]])
assert A2[0, 0] == 0
assert A2.det() == -3
P = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
L2 = sp.Matrix([
    [1, 0, 0, 0],
    [sp.Rational(1, 2), 1, 0, 0],
    [0, sp.Rational(2, 7), 1, 0],
    [sp.Rational(1, 2), sp.Rational(1, 7), 0, 1],
])
R2 = sp.Matrix([
    [2, -1, 0, 1],
    [0, sp.Rational(7, 2), 0, sp.Rational(1, 2)],
    [0, 0, -1, sp.Rational(13, 7)],
    [0, 0, 0, sp.Rational(3, 7)],
])
assert sp.simplify(L2 * R2 - P * A2) == sp.zeros(4)
# 列主元中间矩阵
# 第一次交换后是原第 3,2,1,4 行，不是最终的 P
A2s = sp.Matrix([[2, -1, 0, 1], [1, 0, 0, 1], [0, 1, -1, 2], [1, 3, 0, 1]])
assert A2s == sp.Matrix.vstack(A2.row(2), A2.row(1), A2.row(0), A2.row(3))
row = A2s.copy()
row[1, :] -= sp.Rational(1, 2) * row[0, :]
row[3, :] -= sp.Rational(1, 2) * row[0, :]
assert row[1, :] == sp.Matrix([[0, sp.Rational(1, 2), 0, sp.Rational(1, 2)]])
assert row[3, :] == sp.Matrix([[0, sp.Rational(7, 2), 0, sp.Rational(1, 2)]])

# 例 3
A3 = sp.Matrix([[5, -2, 0], [-2, 3, -1], [0, -1, 1]])
assert A3[:1, :1].det() == 5
assert A3[:2, :2].det() == 11
assert A3.det() == 6
L3 = sp.Matrix([[1, 0, 0], [sp.Rational(-2, 5), 1, 0], [0, sp.Rational(-5, 11), 1]])
D3 = sp.diag(5, sp.Rational(11, 5), sp.Rational(6, 11))
assert sp.simplify(L3 * D3 * L3.T - A3) == sp.zeros(3)
Lbad = sp.Matrix([[1, 0, 0], [sp.Rational(-2, 5), 1, 0], [0, sp.Rational(-11, 5), 1]])
assert sp.simplify(Lbad * D3 * Lbad.T - A3) != sp.zeros(3)
G = sp.simplify(L3 * sp.diag(sp.sqrt(5), sp.sqrt(sp.Rational(11, 5)), sp.sqrt(sp.Rational(6, 11))))
assert sp.simplify(G * G.T - A3) == sp.zeros(3)
assert sp.simplify(G[1, 0] + sp.Rational(2, 5) * sp.sqrt(5)) == 0
assert sp.simplify(G[2, 1] + sp.sqrt(55) / 11) == 0
assert sp.simplify(G[2, 2] - sp.sqrt(66) / 11) == 0
assert sp.simplify(G[1, 1] - sp.sqrt(55) / 5) == 0

# 例题 4：用例 1 的分解解 Ax=[-3,2,3]
b4 = sp.Matrix([-3, 2, 3])
y4 = L1e.solve(b4)
x4 = R1e.solve(y4)
assert y4 == sp.Matrix([-3, sp.Rational(7, 2), 8])
assert x4 == sp.Matrix([1, 0, 1])
assert A1 * x4 == b4
# Cholesky 直接递推与 L D^{1/2} 一致
g11 = sp.sqrt(5)
g21 = -2 / g11
g31 = 0
g22 = sp.sqrt(3 - g21 ** 2)
g32 = (-1) / g22
g33 = sp.sqrt(1 - g32 ** 2)
G2 = sp.Matrix([[g11, 0, 0], [g21, g22, 0], [g31, g32, g33]])
assert sp.simplify(G2 - G) == sp.zeros(3)

# 不唯一：D=diag(2,1,4)
Dmove = sp.diag(2, 1, 4)
assert sp.simplify((L * Dmove) * (Dmove.inv() * R) - A) == sp.zeros(3)
assert L * Dmove == sp.Matrix([[2, 0, 0], [4, 1, 0], [-2, sp.Rational(1, 2), 4]])
assert sp.simplify(Dmove.inv() * R) == sp.Matrix([[sp.Rational(1, 2), 1, sp.Rational(1, 2)], [0, -2, 1], [0, 0, sp.Rational(1, 8)]])

# Δn=0 仍可 Doolittle
S = sp.Matrix([[1, 1], [1, 1]])
Ls = sp.Matrix([[1, 0], [1, 1]])
Rs = sp.Matrix([[1, 1], [0, 0]])
assert Ls * Rs == S and S.det() == 0 and S[0, 0] == 1

# 三对角自编例
At = sp.Matrix([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
assert [At[:k, :k].det() for k in (1, 2, 3)] == [2, 3, 4]
Lt = sp.Matrix([[1, 0, 0], [sp.Rational(1, 2), 1, 0], [0, sp.Rational(2, 3), 1]])
Rt = sp.Matrix([[2, 1, 0], [0, sp.Rational(3, 2), 1], [0, 0, sp.Rational(4, 3)]])
assert Lt * Rt == At

print("ALL OK")
