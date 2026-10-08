"""M1.2 矩阵及其运算：验算讲解中出现的全部数值与恒等式（numpy / sympy 精确核对）。"""
import numpy as np
import sympy as sp
from sympy import Matrix, I, Rational, symbols, eye, zeros, diag, simplify, binomial, expand

def M(rows):
    return Matrix(rows)

# ---------------- K1 特殊矩阵 ----------------
H = M([[2, 1 - I], [1 + I, 3]])
assert H.H == H                              # Hermite
assert H[0, 0].is_real and H[1, 1].is_real   # 对角元为实数
S = M([[1, I], [I, 1]])
assert S.T == S and S.H != S                 # 复对称但不是 Hermite
assert S.H == M([[1, -I], [-I, 1]])
Z = M([[1, 2 - I], [3, 4 + 5 * I]])
assert Z.H == Z.conjugate().T

# ---------------- K2 加法与数乘 ----------------
A = M([[1, 2], [3, 4]]); B = M([[0, 1], [-1, 2]])
assert 3 * A - 2 * B == M([[3, 4], [11, 8]])
assert (A + B) == (B + A)

# ---------------- K3 乘法 / 教材 1.2 节例 1 ----------------
A1 = M([[1, 2, 3], [-1, 0, -2]]); B1 = M([[2, 1], [-1, 2], [0, -1]])
assert A1 * B1 == M([[0, 2], [-2, 1]])
assert B1 * A1 == M([[1, 4, 4], [-3, -2, -7], [1, 0, 2]])
assert (A1 * B1).shape == (2, 2) and (B1 * A1).shape == (3, 3)
# 逐项
assert 1*2 + 2*(-1) + 3*0 == 0 and 1*1 + 2*2 + 3*(-1) == 2
assert (-1)*2 + 0*(-1) + (-2)*0 == -2 and (-1)*1 + 0*2 + (-2)*(-1) == 1
# 同阶不可交换
P = M([[1, 1], [0, 1]]); Q = M([[1, 0], [1, 1]])
assert P * Q == M([[2, 1], [1, 1]]) and Q * P == M([[1, 1], [1, 2]])
# AB=O 与消去律失效
E11 = M([[1, 0], [0, 0]]); E22 = M([[0, 0], [0, 1]])
assert E11 * E22 == zeros(2, 2)
Bm = M([[1, 2], [3, 4]]); Cm = M([[1, 2], [5, 6]])
assert E11 * Bm == E11 * Cm == M([[1, 2], [0, 0]]) and Bm != Cm
# 列的观点
col = M([[1], [-1]])
assert A1 * M([[2], [-1], [0]]) == M([[0], [-2]])   # AB 的第 1 列
# 习题 1 的结构由习题解答给出，此处只核对一个数值实例（不进入讲解）

# ---------------- K4 可交换矩阵 / 教材 1.2 节例 2 ----------------
b11, b12, b21, b22 = symbols('b11 b12 b21 b22')
A2 = M([[2, 1], [0, 3]]); Bb = M([[b11, b12], [b21, b22]])
assert A2 * Bb == M([[2*b11 + b21, 2*b12 + b22], [3*b21, 3*b22]])
assert Bb * A2 == M([[2*b11, b11 + 3*b12], [2*b21, b21 + 3*b22]])
sol = sp.solve([2*b11 + b21 - 2*b11, 2*b12 + b22 - b11 - 3*b12, 3*b21 - 2*b21, 3*b22 - b21 - 3*b22],
               [b21, b22], dict=True)[0]
assert sol == {b21: 0, b22: b11 + b12}
Bgen = Bb.subs(sol)
assert Bgen == M([[b11, b12], [0, b11 + b12]])
assert expand((b11 - 2*b12) * eye(2) + b12 * A2) == Bgen        # B=(b11-2 b12)I + b12 A
assert A2 * Bgen == Bgen * A2
# (A+B)^2 与 (AB)^k
assert (P + Q) ** 2 == M([[5, 4], [4, 5]])
assert P**2 + 2*P*Q + Q**2 == M([[6, 4], [4, 4]])
assert P**2 + P*Q + Q*P + Q**2 == M([[5, 4], [4, 5]])
assert P**2 == M([[1, 2], [0, 1]]) and Q**2 == M([[1, 0], [2, 1]])
assert (P*Q)**2 == M([[5, 3], [3, 2]]) and P**2 * Q**2 == M([[5, 2], [2, 1]])
assert (P*Q)**2 != P**2 * Q**2
# 交换时两式成立：取 A=B 的多项式
X = M([[1, 2], [3, 4]]); Y = 2*X + eye(2)
assert X*Y == Y*X and (X+Y)**2 == X**2 + 2*X*Y + Y**2 and (X*Y)**3 == X**3 * Y**3

# ---------------- K5 幂 / 教材 1.2 节例 3 ----------------
U = M([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
J = -2 * eye(4) + U
assert J == M([[-2, 1, 0, 0], [0, -2, 1, 0], [0, 0, -2, 1], [0, 0, 0, -2]])
assert U**2 == M([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
assert U**3 == M([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
assert U**4 == zeros(4, 4)
assert J**2 == M([[4, -4, 1, 0], [0, 4, -4, 1], [0, 0, 4, -4], [0, 0, 0, 4]])
assert J**3 == M([[-8, 12, -6, 1], [0, -8, 12, -6], [0, 0, -8, 12], [0, 0, 0, -8]])
assert J**2 == 4*eye(4) - 4*U + U**2
assert J**3 == -8*eye(4) + 12*U - 6*U**2 + U**3
def Jk(k):
    return sum((binomial(k, j) * (-2)**(k - j) * U**j for j in range(4)), zeros(4, 4))
for k in range(1, 12):
    assert J**k == Jk(k), k
# k = 4, 5 的元素
assert [(-2)**4, 4*(-2)**3, binomial(4, 2)*(-2)**2, binomial(4, 3)*(-2)] == [16, -32, 24, -8]
assert (J**4)[0, :] == M([[16, -32, 24, -8]])
assert [(-2)**5, 5*(-2)**4, binomial(5, 2)*(-2)**3, binomial(5, 3)*(-2)**2] == [-32, 80, -80, 40]
assert (J**5)[0, :] == M([[-32, 80, -80, 40]])
# 一般 Jordan 块：(i,i+j) 元 = C(k,j) lam^(k-j)
lam = symbols('lam')
Jl = lam * eye(4) + U
assert expand(Jl**6 - sum((binomial(6, j) * lam**(6 - j) * U**j for j in range(4)), zeros(4, 4))) == zeros(4, 4)
# 教材书页 93 的 0.8 Jordan 块（K5 关联）
J08 = M([[Rational(4, 5), 1], [0, Rational(4, 5)]])
assert (J08**3)[0, 1] == 3 * Rational(4, 5)**2

# ---------------- K6 A^H A = O ----------------
Ac = M([[1, I], [0, 1]])
assert Ac.H * Ac == M([[1, I], [-I, 2]])
assert [sum(abs(Ac[i, j])**2 for i in range(2)) for j in range(2)] == [1, 2]
col_i = M([[1], [I]])
assert col_i.T * col_i == M([[0]]) and col_i.H * col_i == M([[2]])
# 例题 3
A3 = M([[1, I], [-I, 1]]); B3 = M([[I], [-1]])
assert A3.H == A3
assert A3 * B3 == zeros(2, 1)
assert A3.H * A3 == M([[2, 2*I], [-2*I, 2]])
assert A3.H * A3 * B3 == zeros(2, 1)

# ---------------- K7 伴随矩阵与逆矩阵 ----------------
a, b, c, d = symbols('a b c d')
A22 = M([[a, b], [c, d]])
assert A22.adjugate() == M([[d, -b], [-c, a]])
assert A22 * A22.adjugate() == (a*d - b*c) * eye(2)
A5 = M([[4, 3, 2], [3, 2, 1], [2, 1, 1]])
assert 4*2*1 + 3*1*2 + 2*3*1 - 2*2*2 - 4*1*1 - 3*3*1 == -1 and A5.det() == -1
cof = Matrix(3, 3, lambda i, j: A5.cofactor(i, j))
assert cof == M([[1, -1, -1], [-1, 0, 2], [-1, 2, -1]])
assert A5.adjugate() == cof.T == M([[1, -1, -1], [-1, 0, 2], [-1, 2, -1]])
assert A5**-1 == M([[-1, 1, 1], [1, 0, -2], [1, -2, 1]])
assert A5.adjugate() / A5.det() == A5**-1
assert A5 * A5.adjugate() == A5.det() * eye(3)
# 二阶余子式逐个（讲解中列出）
assert M([[2, 1], [1, 1]]).det() == 1 and M([[3, 1], [2, 1]]).det() == 1
assert M([[3, 2], [2, 1]]).det() == -1 and M([[3, 2], [1, 1]]).det() == 1
assert M([[4, 2], [2, 1]]).det() == 0 and M([[4, 3], [2, 1]]).det() == -2
assert M([[4, 2], [3, 1]]).det() == -2 and M([[4, 3], [3, 2]]).det() == -1
# 非对称例：伴随必须转置（自编，用于易错点）
A6 = M([[1, 2], [3, 4]])
assert A6.adjugate() == M([[4, -2], [-3, 1]])
assert Matrix(2, 2, lambda i, j: A6.cofactor(i, j)) == M([[4, -3], [-2, 1]])
# 3 阶非对称：代数余子式矩阵 != 伴随
A7 = M([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
assert A7.det() == 1
assert A7**-1 == M([[-24, 18, 5], [20, -15, -4], [-5, 4, 1]])
assert Matrix(3, 3, lambda i, j: A7.cofactor(i, j)) == M([[-24, 20, -5], [18, -15, 4], [5, -4, 1]])

# ---------------- K8 可逆矩阵的性质 ----------------
AB = P * Q
assert AB.det() == 1 and AB**-1 == M([[1, -1], [-1, 2]])
assert Q**-1 * P**-1 == M([[1, -1], [-1, 2]])
assert P**-1 * Q**-1 == M([[2, -1], [-1, 1]]) != AB**-1
assert (3 * A6)**-1 == Rational(1, 3) * A6**-1
assert (A3 * Ac).H == Ac.H * A3.H
Cm2 = M([[1 + I, 2], [0, 1 - I]])
assert (Cm2.H)**-1 == (Cm2**-1).H
assert (eye(2) + (-eye(2))).det() == 0         # I + (-I) 不可逆
# (A^k)^{-1} = (A^{-1})^k
assert (P**3)**-1 == (P**-1)**3

# ---------------- K9 分块矩阵 ----------------
I2 = eye(2); O2 = zeros(2, 2)
Xb = M([[1, 2], [3, 4]]); Yb = M([[0, 1], [1, 0]])
def blk(a_, b_, c_, d_):
    return sp.BlockMatrix([[a_, b_], [c_, d_]]).as_explicit()
Pb = blk(I2, Xb, O2, I2); Qb = blk(I2, Yb, O2, I2)
assert Pb * Qb == blk(I2, Xb + Yb, O2, I2) == Qb * Pb
assert Xb + Yb == M([[1, 3], [4, 4]])
assert Pb**-1 == blk(I2, -Xb, O2, I2)
assert Pb * blk(I2, -Xb, O2, I2) == eye(4)
assert Pb == M([[1, 0, 1, 2], [0, 1, 3, 4], [0, 0, 1, 0], [0, 0, 0, 1]])
# 分块对角矩阵的积、幂、行列式、逆
A1b = M([[1, 2], [3, 4]]); A2b = M([[5]])
Mb = sp.diag(A1b, A2b)
assert Mb == M([[1, 2, 0], [3, 4, 0], [0, 0, 5]])
assert Mb.det() == A1b.det() * A2b.det() == -10
assert Mb**-1 == sp.diag(A1b**-1, A2b**-1)
assert A1b**-1 == M([[-2, 1], [Rational(3, 2), -Rational(1, 2)]])
assert Mb**3 == sp.diag(A1b**3, A2b**3)
Bb2 = M([[0, 1], [1, 0]]); Bb3 = M([[2]])
assert sp.diag(A1b, A2b) * sp.diag(Bb2, Bb3) == sp.diag(A1b * Bb2, A2b * Bb3)
# 一般分块乘法（教材书页 14 规则）：随机整数 4x4 按 2+2 分块
rng = np.random.default_rng(7)
R1 = rng.integers(-3, 4, (4, 4)); R2 = rng.integers(-3, 4, (4, 4))
blocks = lambda Mx: [[Mx[:2, :2], Mx[:2, 2:]], [Mx[2:, :2], Mx[2:, 2:]]]
a_, b_ = blocks(R1), blocks(R2)
prod = np.block([[a_[i][0] @ b_[0][j] + a_[i][1] @ b_[1][j] for j in range(2)] for i in range(2)])
assert (prod == R1 @ R2).all()

# ---------------- 典型例题 ----------------
# 例题 1：A = I + N, N^2 = O
A_e1 = M([[2, 1], [-1, 0]]); N = A_e1 - eye(2)
assert N == M([[1, 1], [-1, -1]]) and N**2 == zeros(2, 2)
kk = symbols('kk')
for k in range(0, 15):
    assert A_e1**k == eye(2) + k * N == M([[1 + k, k], [-k, 1 - k]])
assert A_e1**2 == M([[3, 2], [-2, -1]])
assert A_e1**10 == M([[11, 10], [-10, -9]])
assert A_e1**-1 == eye(2) - N == M([[0, -1], [1, 2]])
assert A_e1 * M([[0, -1], [1, 2]]) == eye(2)
assert A_e1.det() == 1
# 例题 2：A(t)
t = symbols('t')
At = M([[1, t, 0], [t, 1, t], [0, t, 1]])
assert expand(At.det()) == 1 - 2*t**2
assert sp.solve(1 - 2*t**2, t) == [-sp.sqrt(2)/2, sp.sqrt(2)/2]
A_t1 = At.subs(t, 1)
assert A_t1 == M([[1, 1, 0], [1, 1, 1], [0, 1, 1]]) and A_t1.det() == -1
assert Matrix(3, 3, lambda i, j: A_t1.cofactor(i, j)) == M([[0, -1, 1], [-1, 1, -1], [1, -1, 0]])
assert A_t1**-1 == M([[0, 1, -1], [1, -1, 1], [-1, 1, 0]])
assert A_t1 * M([[0, 1, -1], [1, -1, 1], [-1, 1, 0]]) == eye(3)
assert (2 * A_t1)**-1 == Rational(1, 2) * M([[0, 1, -1], [1, -1, 1], [-1, 1, 0]])
# 例题 2 的各余子式（t=1，按讲解中的行列式逐个核对）
mn = lambda r, c: A_t1.minor(r, c)
assert [mn(0, 0), mn(0, 1), mn(0, 2)] == [0, 1, 1]      # A11=0, A12=-1, A13=1
assert [mn(1, 0), mn(1, 1), mn(1, 2)] == [1, 1, 1]      # A21=-1, A22=1, A23=-1
assert [mn(2, 0), mn(2, 1), mn(2, 2)] == [1, 1, 0]      # A31=1, A32=-1, A33=0
assert A_t1.row(0) * (A_t1**-1) == M([[1, 0, 0]])
# 例题 4：见 K9；X、Y 本身不可交换
assert Xb * Yb == M([[2, 1], [4, 3]]) and Yb * Xb == M([[3, 4], [1, 2]]) and Xb * Yb != Yb * Xb

# ---------------- 自测题 ----------------
# 1: AB=O 但 A,B 均非零（E11 E22）——已在 K3 核对
# 2:
assert P**10 == M([[1, 10], [0, 1]])
# 3:
A_s3 = M([[2, 1], [7, 4]])
assert A_s3.det() == 1 and A_s3.adjugate() == M([[4, -1], [-7, 2]]) == A_s3**-1
# 4: 复矩阵 A^T A = O 但 A != O
assert col_i.T * col_i == M([[0]]) and col_i != zeros(2, 1)
# 5:
J3 = M([[2, 1, 0], [0, 2, 1], [0, 0, 2]])
assert (J3**5)[0, 2] == 80 and (J3**5)[0, 1] == 80 and (J3**5)[0, 0] == 32
assert (J3**5) == M([[32, 80, 80], [0, 32, 80], [0, 0, 32]])
assert binomial(5, 2) * 2**3 == 80
# 6: I + (-I)
# 7: 见 K9
# 习题 6 的 A（用于习题导航提示）
A_ex6 = M([[1, 2, 0, 0], [2, 6, 0, 0], [0, 0, 3, 5], [0, 0, 1, 2]])
assert A_ex6.det() == 2 and A_ex6.det() == M([[1, 2], [2, 6]]).det() * M([[3, 5], [1, 2]]).det()
# 习题 4 的 B^8 = I
B_ex4 = sp.diag(1, 1, -1)
assert B_ex4**2 == eye(3) and B_ex4**8 == eye(3)

print("ALL OK")
