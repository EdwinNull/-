"""M3.3 方阵的谱半径 —— 讲解中全部数值的复算脚本。
运行：python3 .work/kp/verify/M3.3.py   （全部 assert 通过即输出 ALL OK）
"""
import numpy as np
import sympy as sp

I = sp.I

# K1 正文例：A=[[1-i,3],[-1,1+i]]
A = sp.Matrix([[1 - I, 3], [-1, 1 + I]])
lam = sp.symbols("lam")
assert sp.expand((A - lam * sp.eye(2)).det() - ((1 - lam) ** 2 + 4)) == 0
assert set(A.eigenvals()) == {1 + 2 * I, 1 - 2 * I}
An = np.array(A.tolist(), dtype=complex)
assert abs(max(abs(np.linalg.eigvals(An))) - np.sqrt(5)) < 1e-12
# K3 用法一：‖A‖∞ = 3+√2
assert abs(np.linalg.norm(An, np.inf) - (3 + np.sqrt(2))) < 1e-12

# K2 反例
A0 = sp.Matrix([[0, 0], [3, 0]]); B0 = sp.Matrix([[2, 3], [0, 2]])
assert A0 * B0 == sp.Matrix([[0, 0], [6, 9]])
assert set((A0 + B0).eigenvals()) == {5, -1}
assert set((A0 * B0).eigenvals()) == {0, 9}

# K4：J=[[0.8,1],[0,0.8]] 的 2-范数；C=[[2,1],[1,2]]
J = np.array([[0.8, 1.0], [0.0, 0.8]])
assert np.allclose(J.T @ J, [[0.64, 0.8], [0.8, 1.64]])
mu1 = (2.28 + np.sqrt(2.28 ** 2 - 4 * 0.4096)) / 2
assert abs(mu1 - 2.0834) < 1e-4 and abs(np.linalg.norm(J, 2) - 1.4434) < 1e-4
C = np.array([[2.0, 1.0], [1.0, 2.0]])
assert abs(np.linalg.norm(C, 2) - 3) < 1e-12 and np.allclose(sorted(np.linalg.eigvalsh(C @ C)), [1, 9])

# K6 反例：[[1,100],[0,1]]
T = np.array([[1.0, 100.0], [0.0, 1.0]])
assert abs(np.linalg.norm(T, np.inf) * np.linalg.norm(np.linalg.inv(T), np.inf) - 10201) < 1e-9

# 典型例题 1
M = sp.Matrix([[1, 5, -2], [-2, 1, 0], [3, -8, 2]])
assert sp.expand(M.charpoly(lam).as_expr() - (lam ** 3 - 4 * lam ** 2 + 21 * lam + 4)) == 0
assert M.det() == -4
f = lambda x: x ** 3 - 4 * x ** 2 + 21 * x + 4
df = lambda x: 3 * x ** 2 - 8 * x + 21
x = -0.2
rows = []
for k in range(3):
    xn = x - f(x) / df(x)
    rows.append((k, x, f(x), df(x), xn))
    x = xn
assert abs(rows[0][4] - (-0.183803)) < 5e-7 and abs(rows[1][4] - (-0.183750)) < 5e-7
assert abs(rows[1][2] - (-0.001203)) < 5e-7 and abs(rows[1][3] - 22.571773) < 5e-7
lr = x
mod = np.sqrt(-4 / lr)
assert abs(-4 / lr - 21.7688) < 1e-4 and abs(mod - 4.6657) < 1e-4
assert abs((4 - lr) / 2 - 2.0919) < 1e-4 and abs(np.sqrt(-4 / lr - ((4 - lr) / 2) ** 2) - 4.1705) < 1e-4
ratio = mod / abs(lr)
assert abs(ratio - 25.39) < 5e-3
Mn = np.array(M.tolist(), dtype=float)
ev = np.linalg.eigvals(Mn)
assert abs(max(abs(ev)) / min(abs(ev)) - ratio) < 1e-9
Mi = np.linalg.inv(Mn)
assert abs(np.linalg.norm(Mn, np.inf) * np.linalg.norm(Mi, np.inf) - 611 / 4) < 1e-9
assert abs(np.linalg.norm(Mn, 1) * np.linalg.norm(Mi, 1) - 259 / 2) < 1e-9
assert abs(np.linalg.cond(Mn, 2) - 78.37) < 5e-3

# 典型例题 2
D = np.diag([1.0, 0.1])
DJD = np.linalg.inv(D) @ J @ D
assert np.allclose(DJD, [[0.8, 0.1], [0.0, 0.8]])
assert abs(np.linalg.norm(DJD, np.inf) - 0.9) < 1e-12 and abs(np.linalg.norm(DJD, 2) - 0.8516) < 1e-4
table = {1: 1.8000, 2: 2.2400, 3: 2.4320, 5: 2.3757, 10: 1.4496, 20: 0.2998, 40: 0.0068}
for k, v in table.items():
    val = np.linalg.norm(np.linalg.matrix_power(J, k), np.inf)
    assert abs(val - (0.8 ** k + k * 0.8 ** (k - 1))) < 1e-12
    assert abs(val - v) < 5e-5, (k, val)

# 自测题
assert np.allclose(np.linalg.norm(np.diag([1, -3, 2j]), 2), 3)
assert abs(max(abs(np.linalg.eigvals(np.array([[0.5, 10], [0, 0.5]])))) - 0.5) < 1e-12

print("ALL OK")
