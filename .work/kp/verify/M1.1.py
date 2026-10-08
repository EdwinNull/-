"""M1.1 向量和向量空间 —— 讲解中全部数值的复算脚本。
运行：python3 .work/kp/verify/M1.1.py   （全部 assert 通过即输出 ALL OK）
"""
import itertools
import numpy as np
import sympy as sp

I = sp.I
R = sp.Rational


def ip(a, b):
    """本书内积：<a,b> = sum a_i * conj(b_i)（式 (1.1-1)）。"""
    return sp.simplify(sum(x * sp.conjugate(y) for x, y in zip(a, b)))


def norm(a):
    return sp.sqrt(ip(a, a))


def V(*xs):
    return sp.Matrix(xs)


# ---------- K1 例 ----------
a = V(1, 2, 3); b = V(0, -1, 4)
assert 2 * a - 3 * b == V(2, 7, -6)
assert 2 * a == V(2, 4, 6) and 3 * b == V(0, -3, 12)

# ---------- K2 例：alpha=[1,i] ----------
al = V(1, I)
assert ip(al, al) == 2 and norm(al) == sp.sqrt(2)
assert sp.simplify(al[0] * al[0] + al[1] * al[1]) == 0          # 漏共轭的错误结果 1+i^2=0
# 自述：alpha=[i] 时不加共轭 i^2=-1
assert I ** 2 == -1
# 内积规则 (2)(4) 与第二变元共轭齐次：随机复向量数值核对
rng = np.random.default_rng(0)
def cip(x, y): return np.sum(x * np.conj(y))
x = rng.normal(size=4) + 1j * rng.normal(size=4)
y = rng.normal(size=4) + 1j * rng.normal(size=4)
k = 0.7 - 1.3j
assert abs(cip(x, y) - np.conj(cip(y, x))) < 1e-12
assert abs(cip(k * x, y) - k * cip(x, y)) < 1e-12
assert abs(cip(x, k * y) - np.conj(k) * cip(x, y)) < 1e-12
assert abs(cip(x, y) - np.vdot(y, x)) < 1e-12                   # <a,b> = b^H a
assert abs(np.linalg.norm(k * x) - abs(k) * np.linalg.norm(x)) < 1e-12
# 向量空间 C^1：<[1],[i]>、<alpha,i beta>（自测题 1）
assert ip(V(1), V(I)) == -I and ip(V(1), I * V(1)) == -I and I * ip(V(1), V(1)) == I

# ---------- K3 ----------
# 勾股逆在复空间不成立：alpha=[1], beta=[i]
assert abs(1 + I) ** 2 == 2 and 1 + 1 == 2 and ip(V(1), V(I)) == -I
# 实例
al = V(1, 2, 2); be = V(2, -1, 0); ga = V(1, 1, 1)
assert ip(al, be) == 0
assert (al + be) == V(3, 1, 2) and ip(al + be, al + be) == 14 and ip(al, al) == 9 and ip(be, be) == 5
assert ip(al, ga) == 5 and abs(float(3 * sp.sqrt(3)) - 5.196) < 5e-4 and 5 < 3 * sp.sqrt(3)
# Cauchy-Schwarz 与三角不等式随机核对（复）
for _ in range(200):
    x = rng.normal(size=5) + 1j * rng.normal(size=5)
    y = rng.normal(size=5) + 1j * rng.normal(size=5)
    assert abs(cip(x, y)) <= np.linalg.norm(x) * np.linalg.norm(y) + 1e-12
    assert np.linalg.norm(x + y) <= np.linalg.norm(x) + np.linalg.norm(y) + 1e-12
# 取等条件：成比例
x = rng.normal(size=4) + 1j * rng.normal(size=4)
y = (0.3 + 2j) * x
assert abs(abs(cip(x, y)) - np.linalg.norm(x) * np.linalg.norm(y)) < 1e-10

# ---------- K4 例 1（n=3、n=4 上三角） ----------
for n in (3, 4, 6):
    M = sp.Matrix(n, n, lambda i, j: 1 if i <= j else 0)        # 列 j 为 [1,..,1,0..0]（前 j+1 个为 1）
    assert M.det() == 1 and M.rank() == n
# 要点中的反例
S = sp.Matrix([[1, 2, 0], [0, 0, 1]])
assert S.rank() == 2 and S.shape[1] == 3
assert V(1, 0) == R(1, 2) * V(2, 0)
assert sp.Matrix.hstack(V(1, 0), V(2, 0)).rank() == 1           # [0,1] 不属于它们的张成：第二分量均为 0

# ---------- K5 要点 ----------
assert sp.Matrix.hstack(V(1, 0), V(2, 0)).rank() == 1           # {[1,0]} 与 {[1,0],[2,0]} 秩相同但后者个数为 2

# ---------- K6 例 ----------
assert sp.Matrix.hstack(V(1, 0), V(2, 0), V(0, 1)).rank() == 2
assert sp.Matrix.hstack(V(1, 0), V(0, 1)).rank() == 2 and sp.Matrix.hstack(V(2, 0), V(0, 1)).rank() == 2

# ---------- K7 例 与 要点 ----------
assert V(0, 2, -1) + V(0, 5, 4) == V(0, 7, 3) and 3 * V(0, 2, -1) == V(0, 6, -3)
assert V(1, 0) + V(0, 1) == V(1, 1) and 1 * 1 != 0               # x*y=1 != 0 的反例（坐标轴并集）

# ---------- K8 例（书页 7–8 正文例）与 e_i ----------
a1 = V(1, 3, 4); a2 = V(1, 2, 0); a3 = V(1, 0, 0)
B = sp.Matrix.hstack(a1, a2, a3)
assert B.det() != 0
coef = B.solve(V(1, 1, 4))
assert coef == V(1, -1, 1)
assert 1 * a1 + (-1) * a2 + 1 * a3 == V(1, 1, 4)

# ---------- K9 例 与 拓展 ----------
e1 = V(R(3, 5), R(4, 5)); e2 = V(R(-4, 5), R(3, 5))
assert ip(e1, e1) == 1 and ip(e2, e2) == 1 and ip(e1, e2) == 0
bt = V(1, 2)
c1, c2 = ip(bt, e1), ip(bt, e2)
assert c1 == R(11, 5) and c2 == R(2, 5) and c1 * e1 + c2 * e2 == bt
# 例 3（内积证无关）：正交非零向量组的 Gram 矩阵为对角阵且可逆
G = sp.Matrix(3, 3, lambda i, j: ip([V(1, 1, 0), V(1, -1, 1), V(-1, 1, 2)][i], [V(1, 1, 0), V(1, -1, 1), V(-1, 1, 2)][j]))
assert G.is_diagonal() and G.det() != 0

# ---------- K10 教材 例 4 ----------
def schmidt(vs):
    es, bs = [], []
    for v in vs:
        b = v - sum((ip(v, e) * e for e in es), sp.zeros(len(v), 1))
        b = sp.simplify(b)
        bs.append(b)
        es.append(sp.simplify(b / norm(b)))
    return bs, es

bs, es = schmidt([V(1, 1, 0), V(2, 0, 1), V(2, 2, 1)])
assert bs[0] == V(1, 1, 0) and bs[1] == V(1, -1, 1)
assert sp.simplify(bs[2] - V(R(-1, 3), R(1, 3), R(2, 3))) == sp.zeros(3, 1)
assert sp.simplify(es[0] - V(1, 1, 0) / sp.sqrt(2)) == sp.zeros(3, 1)
assert sp.simplify(es[1] - V(1, -1, 1) / sp.sqrt(3)) == sp.zeros(3, 1)
assert sp.simplify(es[2] - V(-1, 1, 2) / sp.sqrt(6)) == sp.zeros(3, 1)
assert sp.simplify(ip(V(2, 0, 1), es[0]) - sp.sqrt(2)) == 0
assert sp.simplify(ip(V(2, 2, 1), es[0]) - 2 * sp.sqrt(2)) == 0
assert sp.simplify(ip(V(2, 2, 1), es[1]) - 1 / sp.sqrt(3)) == 0
assert sp.simplify(norm(bs[0]) - sp.sqrt(2)) == 0

# 教材 例 5（复）
bs, es = schmidt([V(I, -1, I), V(1, 0, I), V(1, 1, 1)])
assert sp.simplify(norm(bs[0]) - sp.sqrt(3)) == 0
assert sp.simplify(ip(V(1, 0, I), es[0]) - (1 - I) / sp.sqrt(3)) == 0
assert sp.simplify(bs[1] - V(2 - I, 1 - I, -1 + 2 * I) / 3) == sp.zeros(3, 1)
assert sp.simplify(norm(bs[1]) - 2 / sp.sqrt(3)) == 0
assert sp.simplify(es[1] - V(2 - I, 1 - I, -1 + 2 * I) / (2 * sp.sqrt(3))) == sp.zeros(3, 1)
assert sp.simplify(bs[2] - V(I, 1 - I, 1) / 2) == sp.zeros(3, 1)
assert sp.simplify(norm(bs[2]) - 1) == 0
assert sp.simplify(es[2] - V(I, 1 - I, 1) / 2) == sp.zeros(3, 1)
# 内积次序写反则不再正交（易错点 8）
wrong = V(1, 0, I) - ip(es[0], V(1, 0, I)) * es[0]
assert sp.simplify(ip(wrong, es[0])) != 0

# ---------- 典型例题 1 ----------
al = [V(1, 0, 1, 2), V(0, 1, 1, 1), V(1, 1, 2, 3), V(0, 0, 1, 1), V(1, 0, -1, 0)]
A = sp.Matrix.hstack(*al)
assert A.rank() == 3
assert sp.Matrix.hstack(al[0], al[1], al[3]).rank() == 3
assert sp.Matrix.hstack(al[0], al[1], al[4]).rank() == 3
assert al[2] == al[0] + al[1]
assert al[4] == al[0] - 2 * al[3]
assert al[3] == R(1, 2) * (al[0] - al[4])
# 第一步分量式：[k1, k2, k1+k2+k4, 2k1+k2+k4]
k1, k2, k4, x_, y_, z_ = sp.symbols("k1 k2 k4 x y z")
assert sp.Matrix.hstack(al[0], al[1], al[3]) * V(k1, k2, k4) == V(k1, k2, k1 + k2 + k4, 2 * k1 + k2 + k4)
# 第五步分量式：k1α1+k2α2+k5α5 = [k1+k5, k2, k1+k2-k5, 2k1+k2]
k5 = sp.symbols("k5")
assert sp.Matrix.hstack(al[0], al[1], al[4]) * V(k1, k2, k5) == V(k1 + k5, k2, k1 + k2 - k5, 2 * k1 + k2)

# ---------- 典型例题 2 ----------
bs, es = schmidt([V(1, 2, 2), V(-1, 0, 2), V(0, 0, 1)])
assert norm(bs[0]) == 3 and es[0] == V(R(1, 3), R(2, 3), R(2, 3))
assert ip(V(-1, 0, 2), es[0]) == 1
assert bs[1] == V(R(-4, 3), R(-2, 3), R(4, 3)) and norm(bs[1]) == 2
assert es[1] == V(R(-2, 3), R(-1, 3), R(2, 3))
assert ip(V(0, 0, 1), es[0]) == R(2, 3) and ip(V(0, 0, 1), es[1]) == R(2, 3)
assert bs[2] == V(R(2, 9), R(-2, 9), R(1, 9)) and ip(bs[2], bs[2]) == R(1, 9) and norm(bs[2]) == R(1, 3)
assert es[2] == V(R(2, 3), R(-2, 3), R(1, 3))
Q = sp.Matrix.hstack(*es)
assert Q.T * Q == sp.eye(3)
assert ip(es[0], es[1]) == 0 and ip(es[0], es[2]) == 0 and ip(es[1], es[2]) == 0
bt = V(1, 2, 3)
cs = [ip(bt, e) for e in es]
assert cs == [R(11, 3), R(2, 3), R(1, 3)]
assert sum((c * e for c, e in zip(cs, es)), sp.zeros(3, 1)) == bt
assert R(11, 9) - R(4, 9) + R(2, 9) == 1 and R(22, 9) - R(2, 9) - R(2, 9) == 2 and R(22, 9) + R(4, 9) + R(1, 9) == 3

# ---------- 典型例题 3 ----------
g1 = V(1, 1, 0); g2 = V(2, 0, 1)
assert sp.Matrix.hstack(g1, g2).rank() == 2
assert sp.Matrix.hstack(g1, g2) * V(k1, k2) == V(k1 + 2 * k2, k1, k2)
assert 3 * g1 + 1 * g2 == V(5, 3, 1) and 5 == 3 + 2 * 1
# W2 反例、W3 反例
u = V(1, 0, 0); v = V(0, 1, 0)
assert (u + v)[0] * (u + v)[1] == 1 and u[0] * u[1] == 0 and v[0] * v[1] == 0
assert (-1 * u)[0] < 0
# W1 是子空间的符号核对
x1, x2, x3, y1, y2, y3, kk = sp.symbols("x1 x2 x3 y1 y2 y3 kk")
assert sp.expand((x1 + y1) - (x2 + y2) - 2 * (x3 + y3) - ((x1 - x2 - 2 * x3) + (y1 - y2 - 2 * y3))) == 0
assert sp.expand(kk * x1 - kk * x2 - 2 * kk * x3 - kk * (x1 - x2 - 2 * x3)) == 0

# ---------- 典型例题 4 ----------
al = V(1 + I, 2); be = V(I, 1 - I)
assert sp.simplify(ip(al, be) - (3 + I)) == 0
assert sp.simplify(ip(be, al) - (3 - I)) == 0
assert sp.simplify(ip(I * al, be) - (-1 + 3 * I)) == 0
assert sp.simplify(ip(al, I * be) - (1 - 3 * I)) == 0
assert sp.simplify(ip(al, al) - 6) == 0 and sp.simplify(ip(be, be) - 3) == 0
assert abs(float(sp.sqrt(10)) - 3.162) < 5e-4 and abs(float(sp.sqrt(18)) - 4.243) < 5e-4
assert sp.simplify(al + be - V(1 + 2 * I, 3 - I)) == sp.zeros(2, 1)
assert sp.simplify(ip(al + be, al + be) - 15) == 0
assert abs(float(sp.sqrt(15)) - 3.873) < 5e-4 and abs(float(sp.sqrt(6) + sp.sqrt(3)) - 4.182) < 5e-4
assert sp.simplify(sp.expand((1 + I) * I + 2 * (1 - I)) - (1 - I)) == 0           # 漏共轭的错误值
assert sp.simplify(sp.expand((1 + I) * (-I) + 2 * (1 + I)) - (3 + I)) == 0

# ---------- 自测题 ----------
assert ip(V(1), I * V(1)) == -I                                      # 题 1
assert al is not None and norm(V(1, -2, 2)) == 3                      # 题 3
assert V(1, -2, 2) / 3 == V(R(1, 3), R(-2, 3), R(2, 3))
Bm = sp.Matrix.hstack(V(1, 1, 0), V(1, 0, 1), V(0, 1, 1))              # 题 4
assert Bm.det() == -2
assert Bm.solve(V(2, 0, 0)) == V(1, 1, -1)
assert V(1, 1, 0) + V(1, 0, 1) - V(0, 1, 1) == V(2, 0, 0)
bs, es = schmidt([V(1, I), V(1, 0)])                                   # 题 7
assert sp.simplify(norm(V(1, I)) ** 2 - 2) == 0
assert sp.simplify(ip(V(1, 0), es[0]) - 1 / sp.sqrt(2)) == 0
assert sp.simplify(bs[1] - V(R(1, 2), -I / 2)) == sp.zeros(2, 1)
assert sp.simplify(norm(bs[1]) - 1 / sp.sqrt(2)) == 0
assert sp.simplify(es[1] - V(1, -I) / sp.sqrt(2)) == sp.zeros(2, 1)
assert sp.simplify(ip(es[0], es[1])) == 0

# ---------- 习题导航提到的结论（与现有解答一致） ----------
a1 = V(3, 1, 5, 2); a2 = V(10, 5, 1, 10); a3 = V(1, -1, 1, 4)
bt = (3 * a1 + 2 * a2 - 5 * a3) / 10
assert bt == V(R(12, 5), R(9, 5), R(6, 5), R(3, 5))
assert 3 * (a1 - bt) + 2 * (a2 - bt) == 5 * (a3 + bt)
assert 3 * a1 + 2 * a2 - 5 * a3 == V(24, 18, 12, 6)
Bx = sp.Matrix.hstack(V(1, 1, 0), V(0, 0, 2), V(0, 1, 2))
assert Bx.det() != 0 and Bx.solve(V(5, 7, -2)) == V(5, -3, 2)
bs, es = schmidt([V(1, 1, 0, 0), V(0, 0, 1, 1), V(1, 0, 0, -1), V(0, 1, 1, 0)])
assert es[0] == V(1, 1, 0, 0) / sp.sqrt(2) and es[1] == V(0, 0, 1, 1) / sp.sqrt(2)
assert es[2] == V(R(1, 2), R(-1, 2), R(1, 2), R(-1, 2)) and es[3] == V(R(-1, 2), R(1, 2), R(1, 2), R(-1, 2))

print("ALL OK")
