"""N2.6 迭代法的数值稳定性。运行：python3 .work/kp/verify/N2.6.py"""
import numpy as np

q = 0.5
eps = 1e-4
# 几何级数
s = sum(q**j for j in range(20))
assert s < eps / (1 - q) / eps  # s < 1/(1-q)
assert abs(s - (1 - q**20) / (1 - q)) < 1e-12
assert (1 - q**20) / (1 - q) < 1 / (1 - q)

# q 接近 1 时界放大
assert abs(1e-4 / (1 - 0.99) - 0.01) < 1e-12
assert abs(1e-4 / (1 - 0.5) - 2e-4) < 1e-15

# 一步舍入后精确迭代：||δx^{(k+1)}|| ≤ q^k ε
B = 0.5 * np.eye(2)
delta = np.array([1e-4, 0.0])
d = delta.copy()
for k in range(1, 6):
    d = B @ d
    assert np.linalg.norm(d, np.inf) <= (0.5) ** k * 1e-4 + 1e-15

# 习题 2 第 11 题的迭代矩阵
# x <- x + ω(b-Ax) 的 B = I-ωA
A = np.array([[2.0, 0], [0, 4.0]])
# SPD 特征值 2,4；0<ω<2/4=0.5 时 ρ(I-ωA)<1
for w in (0.1, 0.4):
    Bw = np.eye(2) - w * A
    assert max(abs(np.linalg.eigvals(Bw))) < 1
Bw = np.eye(2) - 0.6 * A
assert max(abs(np.linalg.eigvals(Bw))) > 1

# 习题 2 第 3 题：ρ<1 与 (I-A)^{-1}(I+A) 特征值实部为正
def cayley_ok(A):
    rho = max(abs(np.linalg.eigvals(A)))
    if rho >= 1:
        return False
    M = np.linalg.inv(np.eye(A.shape[0]) - A) @ (np.eye(A.shape[0]) + A)
    return all(ev.real > 0 for ev in np.linalg.eigvals(M))

assert cayley_ok(np.array([[0.2, 0.1], [0.0, -0.3]]))
A_big = np.array([[1.2, 0], [0, 0.2]])
assert max(abs(np.linalg.eigvals(A_big))) > 1
assert not cayley_ok(A_big)

print("ALL OK")
