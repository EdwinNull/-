"""M2.1 Jordan 标准形 —— 讲解中全部数值的复算脚本。
运行：python3 .work/kp/verify/M2.1.py   （全部 assert 通过即输出 ALL OK）
"""
import sympy as sp

lam = sp.symbols("lam")
R = sp.Rational
k = sp.symbols("k", integer=True, positive=True)


def charpoly(M):
    """返回 det(M - lam I) 的展开式。"""
    return sp.expand((M - lam * sp.eye(M.shape[0])).det())


def solvable(N, y):
    """方程组 N x = y 有解 <=> r([N | y]) = r(N)（定理 1.4-1）。"""
    return sp.Matrix.hstack(N, y).rank() == N.rank()


def nullity(M):
    return M.shape[1] - M.rank()


# ---------------- K1：教材 2.1 节例 1 ----------------
A1 = sp.Matrix([[-3, 3, -2], [-7, 6, -3], [1, -1, 2]])
assert charpoly(A1) == sp.expand((1 - lam) * (2 - lam) ** 2)
# 教材的列变换：c1 + c2 - c3 得第一列 (2-λ, 2-λ, -(2-λ))
M = A1 - lam * sp.eye(3)
c = M[:, 0] + M[:, 1] - M[:, 2]
assert sp.simplify(c - sp.Matrix([2 - lam, 2 - lam, -(2 - lam)])) == sp.zeros(3, 1)
assert sp.expand(sp.Matrix([[1, 3, -2], [0, 3 - lam, -1], [0, 2, -lam]]).det() - (lam - 1) * (lam - 2)) == 0
assert (A1 - 2 * sp.eye(3)).rank() == 2 and (A1 - sp.eye(3)).rank() == 2

# ---------------- K4：教材 2.1 节例 2 ----------------
A2 = sp.Matrix([[3, 1, 1], [0, 2, 0], [-1, 0, 1]])
assert charpoly(A2) == sp.expand((2 - lam) ** 3)
N2 = A2 - 2 * sp.eye(3)
assert N2.rank() == 2
y1, y2, y3 = sp.symbols("y1 y2 y3")
# 可解条件：y2 = 0
assert solvable(N2, sp.Matrix([1, 0, 5])) and not solvable(N2, sp.Matrix([0, 1, 0]))
x1 = sp.Matrix([1, 0, -1]); x2 = sp.Matrix([0, 0, 1]); x3 = sp.Matrix([-1, 1, 0])
assert N2 * x1 == sp.zeros(3, 1) and N2 * x2 == x1 and N2 * x3 == x2
assert not solvable(N2, x3)  # x3 的第 2 分量为 1，链到此为止
P2 = sp.Matrix.hstack(x1, x2, x3)
assert P2.inv() == sp.Matrix([[1, 1, 0], [1, 1, 1], [0, 1, 0]])
assert P2.inv() * A2 * P2 == sp.Matrix([[2, 1, 0], [0, 2, 1], [0, 0, 2]])
# 要点：(A-2I)x = x1 的通解为 x2 + c x1；换一个解 x2' = x2 + x1 仍可得 Jordan 形
x2b = x2 + x1
assert x2b == sp.Matrix([1, 0, 0]) and solvable(N2, x2b)
# 记录讲解中给出的另一组：x2' = [1,0,0]，x3' 取 [0,1,0]
x3c = sp.Matrix([0, 1, 0])
assert N2 * x3c == x2b
Pc = sp.Matrix.hstack(x1, x2b, x3c)
assert Pc == sp.Matrix([[1, 1, 0], [0, 0, 1], [-1, 0, 0]])
assert Pc.inv() * A2 * Pc == sp.Matrix([[2, 1, 0], [0, 2, 1], [0, 0, 2]])

# ---------------- K5：教材 2.1 节例 3 ----------------
A3 = sp.Matrix([[4, 3, 0, 1], [0, 2, 0, 0], [1, 3, 2, 1], [0, 0, 0, 2]])
assert charpoly(A3) == sp.expand((4 - lam) * (2 - lam) ** 3)
v1 = sp.Matrix([2, 0, 1, 0])
assert A3 * v1 == 4 * v1
N3 = A3 - 2 * sp.eye(4)
assert N3.rank() == 2
# 行变换结果：第1行 [1,0,0,0 | y1-y3]，第3行 [0,1,0,1/3 | (-y1+2y3)/3]；可解 <=> y2 = y4 = 0
Y = sp.Matrix(sp.symbols("Y1:5"))
r1 = sp.Matrix([[1, 0, 0, 0]]); r3 = sp.Matrix([[0, 1, 0, R(1, 3)]])
# 验证：凡满足 y2=y4=0 的 y，x=[y1-y3, (-y1+2y3)/3, 0, 0] 是解
xs = sp.Matrix([Y[0] - Y[2], (-Y[0] + 2 * Y[2]) / 3, 0, 0])
assert sp.simplify(N3 * xs - sp.Matrix([Y[0], 0, Y[2], 0])) == sp.zeros(4, 1)
v2 = sp.Matrix([0, -1, 0, 3]); v3 = sp.Matrix([0, 0, 1, 0]); v4 = sp.Matrix([-1, 0, 0, 2])
assert N3 * v2 == sp.zeros(4, 1) and N3 * v3 == sp.zeros(4, 1) and N3 * v4 == v3
assert not solvable(N3, v2) and solvable(N3, v3)
# 特征向量 a v2 + b v3 作右端可解 <=> a = 0
a, b = sp.symbols("a b")
w = a * v2 + b * v3
assert (w[1], w[3]) == (-a, 3 * a)
P3 = sp.Matrix.hstack(v1, v2, v3, v4)
assert P3.det() == -4
assert P3.inv() == sp.Matrix([[2, 3, 0, 1], [0, -4, 0, 0], [-2, -3, 4, -1], [0, 6, 0, 2]]) / 4
assert P3.inv() * A3 * P3 == sp.diag(4, 2, sp.Matrix([[2, 1], [0, 2]]))

# ---------------- K9：例 3 的指标 ----------------
assert N3 ** 2 == sp.Matrix([[4, 6, 0, 2], [0, 0, 0, 0], [2, 3, 0, 1], [0, 0, 0, 0]])
assert N3 ** 3 == sp.Matrix([[8, 12, 0, 4], [0, 0, 0, 0], [4, 6, 0, 2], [0, 0, 0, 0]])
assert [nullity(N3 ** j) for j in (1, 2, 3)] == [2, 3, 3]
assert (N3 ** 2) * v4 == sp.zeros(4, 1) and N3 * v4 != sp.zeros(4, 1)

# ---------------- K10：教材 2.1 节例 4 ----------------
A4 = sp.Matrix([[3, -1, 1, 1, 0, 0], [1, 1, -1, -1, 0, 0], [0, 0, 2, 0, 1, 1],
                [0, 0, 0, 2, -1, -1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1]])
assert sp.factor(charpoly(A4)) == sp.factor(-lam * (2 - lam) ** 5)
u1 = sp.Matrix([0, 0, 0, 0, 1, -1])
assert A4 * u1 == sp.zeros(6, 1)
N4 = A4 - 2 * sp.eye(6)
# 可解条件：y3 + y4 = 0 且 y5 + y6 = 0
for yv, ok in [([1, 2, 1, -1, 3, -3], True), ([0, 0, 1, 0, 0, 0], False), ([0, 0, 0, 0, 1, 0], False)]:
    assert solvable(N4, sp.Matrix(yv)) == ok
assert N4.rank() == 4  # 列空间维数 4 = 6 - 2 个条件
e2 = sp.Matrix([0, 0, 1, -1, 0, 0]); e3 = sp.Matrix([1, 1, 0, 0, 0, 0])
f2 = sp.Matrix([0, 0, 0, 0, R(1, 2), R(1, 2)])
g2 = sp.Matrix([1, 0, 0, 0, 0, 0]); g3 = sp.Matrix([R(1, 2), 0, R(1, 2), 0, 0, 0])
assert N4 * e2 == sp.zeros(6, 1) and N4 * e3 == sp.zeros(6, 1)
assert N4 * f2 == e2 and N4 * g2 == e3 and N4 * g3 == g2
assert not solvable(N4, f2) and solvable(N4, g2)
P4 = sp.Matrix.hstack(u1, e2, f2, e3, g2, g3)
assert P4 == sp.Matrix([[0, 0, 0, 1, 1, R(1, 2)], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, R(1, 2)],
                        [0, -1, 0, 0, 0, 0], [1, 0, R(1, 2), 0, 0, 0], [-1, 0, R(1, 2), 0, 0, 0]])
J4 = sp.diag(0, sp.Matrix([[2, 1], [0, 2]]), sp.Matrix([[2, 1, 0], [0, 2, 1], [0, 0, 2]]))
assert P4.det() != 0 and P4.inv() * A4 * P4 == J4
# 秩序列：r(N^k) = 4, 2, 1, 1；关于 2 的零度 2, 4, 5, 5
assert [(N4 ** j).rank() for j in (1, 2, 3, 4)] == [4, 2, 1, 1]
nul = [6 - (N4 ** j).rank() for j in (0, 1, 2, 3, 4)]
assert nul == [0, 2, 4, 5, 5]
ge = [nul[j] - nul[j - 1] for j in (1, 2, 3, 4)]  # 阶数 >= j 的块数
assert ge == [2, 2, 1, 0]
# “坏基”：取 e2 与 e2+e3 作两条链的链首，二级根向量都不能再延长
for head in (e2, e2 + e3):
    sol, prm = N4.gauss_jordan_solve(head)
    # 通解第 5、6 分量之和恒为 1，故不满足下一步的可解条件
    assert sp.simplify(sol[4] + sol[5]) == 1
# N^2 作用于 N((A-2I)^3) 的像只含 x3^(1) 方向（长链链底的唯一方向）
ker3 = (N4 ** 3).nullspace()
imgs = sp.Matrix.hstack(*[(N4 ** 2) * v for v in ker3])
assert imgs.rank() == 1 and sp.Matrix.hstack(imgs, e3).rank() == 1
# 自上而下：链首应取 N^2 w（w 为 3 级根向量），例如 w = g3
assert (N4 ** 2) * g3 == e3 and (N4 ** 3) * g3 == sp.zeros(6, 1)

# ---------------- K13：教材 2.1 节例 5 ----------------
A5 = A1
p1 = sp.Matrix([1, 2, 1]); p2 = sp.Matrix([-1, -1, 1]); p3 = sp.Matrix([-1, -2, 0])
assert A5 * p1 == p1
N5 = A5 - 2 * sp.eye(3)
assert N5 * p2 == sp.zeros(3, 1) and N5 * p3 == p2
# 可解条件 3y1 - 2y2 + y3 = 0
assert 3 * p2[0] - 2 * p2[1] + p2[2] == 0
for yv in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 1]):
    yy = sp.Matrix(yv)
    assert solvable(N5, yy) == (3 * yy[0] - 2 * yy[1] + yy[2] == 0)
P5 = sp.Matrix.hstack(p1, p2, p3)
assert P5.inv() == sp.Matrix([[2, -1, 1], [-2, 1, 0], [3, -2, 1]])
J5 = sp.diag(1, sp.Matrix([[2, 1], [0, 2]]))
assert P5.inv() * A5 * P5 == J5
Jk = sp.diag(1, sp.Matrix([[2 ** k, k * 2 ** (k - 1)], [0, 2 ** k]]))
Ak = P5 * Jk * P5.inv()
book = sp.Matrix([
    [2 + 2 ** (k + 1) - 3 * (k + 2) * 2 ** (k - 1), -1 - 2 ** k + 2 * (k + 2) * 2 ** (k - 1), 1 - (k + 2) * 2 ** (k - 1)],
    [4 + 2 ** (k + 1) - 3 * (k + 4) * 2 ** (k - 1), -2 - 2 ** k + 2 * (k + 4) * 2 ** (k - 1), 2 - (k + 4) * 2 ** (k - 1)],
    [2 - 2 ** (k + 1) + 3 * k * 2 ** (k - 1), -1 + 2 ** k - 2 * k * 2 ** (k - 1), 1 + k * 2 ** (k - 1)]])
assert sp.simplify(Ak - book) == sp.zeros(3, 3)
for kk in range(1, 7):
    assert book.subs(k, kk) == A5 ** kk
# 讲解中的化简形式：第 1 行 = [2 - (3k+2)2^(k-1), -1 + (k+1)2^k, 1 - (k+2)2^(k-1)]
simp = sp.Matrix([
    [2 - (3 * k + 2) * 2 ** (k - 1), -1 + (k + 1) * 2 ** k, 1 - (k + 2) * 2 ** (k - 1)],
    [4 - (3 * k + 8) * 2 ** (k - 1), -2 + (k + 3) * 2 ** k, 2 - (k + 4) * 2 ** (k - 1)],
    [2 + (3 * k - 4) * 2 ** (k - 1), -1 - (k - 1) * 2 ** k, 1 + k * 2 ** (k - 1)]])
assert sp.simplify(simp - book) == sp.zeros(3, 3)
assert book.subs(k, 2) == sp.Matrix([[-14, 11, -7], [-24, 18, -10], [6, -5, 5]])

# ---------------- 典型例题 1（自编）：链首必须取特征向量的组合 ----------------
B = sp.Matrix([[3, -1, 0], [1, 1, 0], [1, -1, 2]])
assert charpoly(B) == sp.expand((2 - lam) ** 3)
NB = B - 2 * sp.eye(3)
assert NB == sp.Matrix([[1, -1, 0], [1, -1, 0], [1, -1, 0]]) and NB.rank() == 1
b1 = sp.Matrix([1, 1, 0]); b2 = sp.Matrix([0, 0, 1])
assert NB * b1 == sp.zeros(3, 1) and NB * b2 == sp.zeros(3, 1)
assert not solvable(NB, b1) and not solvable(NB, b2)
u = b1 + b2
assert u == sp.Matrix([1, 1, 1]) and solvable(NB, u)
xb = sp.Matrix([1, 0, 0])
assert NB * xb == u
PB = sp.Matrix.hstack(u, xb, b1)
assert PB.det() == 1
assert PB.inv() == sp.Matrix([[0, 0, 1], [1, -1, 0], [0, 1, -1]])
assert PB.inv() * B * PB == sp.Matrix([[2, 1, 0], [0, 2, 0], [0, 0, 2]])
assert NB ** 2 == sp.zeros(3, 3)

# ---------------- 典型例题 2（自编）：秩序列 + 自上而下取链 ----------------
C = sp.Matrix([[1, -1, 0, -1], [0, 1, 0, 0], [0, -1, 1, -1], [-1, -1, 1, 1]])
assert charpoly(C) == sp.expand((1 - lam) ** 4)
NC = C - sp.eye(4)
assert NC == sp.Matrix([[0, -1, 0, -1], [0, 0, 0, 0], [0, -1, 0, -1], [-1, -1, 1, 0]])
assert NC ** 2 == sp.Matrix([[1, 1, -1, 0], [0, 0, 0, 0], [1, 1, -1, 0], [0, 0, 0, 0]])
assert NC ** 3 == sp.zeros(4, 4)
assert [(NC ** j).rank() for j in (1, 2, 3)] == [2, 1, 0]
nulC = [4 - (NC ** j).rank() for j in (0, 1, 2, 3)]
assert nulC == [0, 2, 3, 4]
assert [nulC[j] - nulC[j - 1] for j in (1, 2, 3)] == [2, 1, 1]
w = sp.Matrix([1, 0, 0, 0])
assert NC * w == sp.Matrix([0, 0, 0, -1]) and NC ** 2 * w == sp.Matrix([1, 0, 1, 0])
# 特征子空间基
c1 = sp.Matrix([1, 0, 1, 0]); c2 = sp.Matrix([0, 1, 1, -1])
assert NC * c1 == sp.zeros(4, 1) and NC * c2 == sp.zeros(4, 1)
PC = sp.Matrix.hstack(NC ** 2 * w, NC * w, w, c2)
assert PC == sp.Matrix([[1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, -1, 0, -1]])
assert PC.det() == -1
assert PC.inv() == sp.Matrix([[0, -1, 1, 0], [0, -1, 0, -1], [1, 1, -1, 0], [0, 1, 0, 0]])
assert PC.inv() * C * PC == sp.diag(sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]]), 1)
# 对照：自下而上时，可解条件为 y2 = 0 且 y3 = y1；c1 满足、c2 不满足
assert solvable(NC, c1) and not solvable(NC, c2)
for yv in ([1, 0, 1, 5], [0, 1, 0, 0], [1, 0, 0, 0]):
    yy = sp.Matrix(yv)
    assert solvable(NC, yy) == (yy[1] == 0 and yy[2] == yy[0])

# ---------------- 典型例题 3（自编）：含参数 ----------------
pa, pb = sp.symbols("a b")
D = sp.Matrix([[1, pa, 1], [0, 1, pb], [0, 0, 1]])
ND = D - sp.eye(3)
assert ND ** 2 == sp.Matrix([[0, 0, pa * pb], [0, 0, 0], [0, 0, 0]])
cases = {(1, 1): [3, 1], (2, -3): [3, 1], (0, 1): [1, 2], (1, 0): [1, 2], (0, 0): [1, 2]}
for (va, vb), _ in cases.items():
    Ns = ND.subs({pa: va, pb: vb})
    r1_, r2_ = Ns.rank(), (Ns ** 2).rank()
    if va * vb != 0:
        assert (r1_, r2_) == (2, 1)
        Js = sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    else:
        assert (r1_, r2_) == (1, 0)
        Js = sp.Matrix([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    Pj, Jj = D.subs({pa: va, pb: vb}).jordan_form()
    assert sorted(Jj.diagonal()) == [1, 1, 1]
    assert sum(1 for i in range(2) if Jj[i, i + 1] == 1) == sum(1 for i in range(2) if Js[i, i + 1] == 1)
# a=b=1 时的链：w = e3，N w = [1,1,0]，N^2 w = [1,0,0]
Ds = D.subs({pa: 1, pb: 1}); NDs = Ds - sp.eye(3)
e_3 = sp.Matrix([0, 0, 1])
assert NDs * e_3 == sp.Matrix([1, 1, 0]) and NDs ** 2 * e_3 == sp.Matrix([1, 0, 0])
PD = sp.Matrix.hstack(NDs ** 2 * e_3, NDs * e_3, e_3)
assert PD.inv() * Ds * PD == sp.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
# a=0, b=1：N = [[0,0,1],[0,0,1],[0,0,0]]，链 w=e3 -> [1,1,0]，另一特征向量 [1,0,0]
Ds2 = D.subs({pa: 0, pb: 1}); NDs2 = Ds2 - sp.eye(3)
assert NDs2 * e_3 == sp.Matrix([1, 1, 0])
PD2 = sp.Matrix.hstack(sp.Matrix([1, 1, 0]), e_3, sp.Matrix([1, 0, 0]))
assert PD2.inv() * Ds2 * PD2 == sp.Matrix([[1, 1, 0], [0, 1, 0], [0, 0, 1]])

# ---------------- K11 小例：同特征多项式、同几何重数而不相似 ----------------
J22 = sp.diag(sp.Matrix([[0, 1], [0, 0]]), sp.Matrix([[0, 1], [0, 0]]))
J31 = sp.diag(sp.Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]]), 0)
assert J22.rank() == J31.rank() == 2
assert (J22 ** 2).rank() == 0 and (J31 ** 2).rank() == 1

# ---------------- 自测题 ----------------
# 3：5 阶，r(N)=3, r(N^2)=1, r(N^3)=0 -> 块 3、2
nul5 = [0, 2, 4, 5]
assert [nul5[j] - nul5[j - 1] for j in (1, 2, 3)] == [2, 2, 1]
J5t = sp.diag(sp.Matrix([[3, 1, 0], [0, 3, 1], [0, 0, 3]]), sp.Matrix([[3, 1], [0, 3]]))
N5t = J5t - 3 * sp.eye(5)
assert [(N5t ** j).rank() for j in (1, 2, 3)] == [3, 1, 0]
# 4：A=[[2,1],[-1,0]]
T = sp.Matrix([[2, 1], [-1, 0]])
assert charpoly(T) == sp.expand((lam - 1) ** 2)
NT = T - sp.eye(2)
t1 = sp.Matrix([1, -1]); t2 = sp.Matrix([1, 0])
assert NT * t1 == sp.zeros(2, 1) and NT * t2 == t1
PT = sp.Matrix.hstack(t1, t2)
assert PT.inv() == sp.Matrix([[0, -1], [1, 1]])
assert PT.inv() * T * PT == sp.Matrix([[1, 1], [0, 1]])
# 2：x 为 2 级根向量、e 为特征向量 => x+e 仍为 2 级（用例 2 数据核对）
xx = x2 + 7 * x1
assert N2 * xx != sp.zeros(3, 1) and N2 ** 2 * xx == sp.zeros(3, 1)

print("ALL OK")
