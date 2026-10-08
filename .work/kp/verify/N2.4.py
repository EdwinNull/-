"""N2.4 线性方程组的迭代解法。运行：python3 .work/kp/verify/N2.4.py"""
import numpy as np

# 例 1
p, q = 0.2, 0.3
B = np.array([[p, q], [-q, p]])
ev = np.linalg.eigvals(B)
assert np.allclose(sorted(ev.real), [p, p])
assert abs(abs(ev[0]) ** 2 - (p**2 + q**2)) < 1e-12
M = np.array([[1.0, 0], [q, 1]])
BS = np.linalg.inv(M) @ np.array([[p, q], [0, p]])
# 书上 B_S = [[p, q], [-p q, p-q^2]]
expect = np.array([[p, q], [-p * q, p - q**2]])
assert np.allclose(BS, expect)
tr = 2 * p - q**2
det = p**2
char = np.poly(BS)
# np.poly is λ^2 - tr λ + det
assert abs(char[1] + tr) < 1e-12
assert abs(char[2] - det) < 1e-12
assert p**2 + q**2 < 1
assert p**2 < 1 and q**2 < (1 + p) ** 2
assert max(abs(np.linalg.eigvals(BS))) < 1

# 例 2
B2 = np.array(
    [
        [0.0, 0.1, -0.2, 0.0],
        [0.0909, 0.0, 0.0909, -0.2727],
        [-0.2, 0.1, 0.0, 0.1],
        [0.0, -0.375, 0.125, 0.0],
    ]
)
g = np.array([0.6, 2.2727, -1.1, 1.875])
assert abs(np.linalg.norm(B2, np.inf) - 0.5) < 1e-12
x = np.zeros(4)
prev = x.copy()
xs = {}
for k in range(1, 14):
    x = B2 @ x + g
    xs[k] = x.copy()
assert np.allclose(xs[1], g)
assert abs(xs[2][0] - 1.0473) < 5e-5
assert abs(xs[3][1] - 2.0533) < 5e-5
assert abs(xs[8][2] + 0.999) < 5e-4
diff = np.linalg.norm(xs[10] - xs[9], np.inf)
assert abs(diff - 0.0008) < 5e-5
# 先验： (1/2)^{k-1} * 2.2727 < 1e-3 ⇒ k>12.15
assert (0.5) ** 11 * 2.2727 >= 1e-3
assert (0.5) ** 12 * 2.2727 < 1e-3

# 例 3：分裂 A=D+L+R，原号
A = np.array(
    [[5, -1, -1, -1], [-1, 10, -1, -1], [-1, -1, 5, -1], [-1, -1, -1, 10]],
    dtype=float,
)
b = np.array([-4.0, 12, 8, 34])
assert np.allclose(np.linalg.solve(A, b), [1, 2, 3, 4])
D = np.diag(np.diag(A))
L = np.tril(A, -1)
R = np.triu(A, 1)
assert np.allclose(D + L + R, A)
BJ = np.eye(4) - np.linalg.inv(D) @ A
assert np.allclose(BJ, np.array([
    [0, 0.2, 0.2, 0.2],
    [0.1, 0, 0.1, 0.1],
    [0.2, 0.2, 0, 0.2],
    [0.1, 0.1, 0.1, 0],
]))
assert abs(np.linalg.norm(BJ, np.inf) - 0.6) < 1e-12
BS = -np.linalg.inv(D + L) @ R
assert abs(BS[1, 1] - 0.02) < 1e-12
assert abs(BS[3, 3] - 0.0584) < 1e-12
assert abs(np.linalg.norm(BS, np.inf) - 0.6) < 1e-12
x = np.zeros(4)
gS = np.linalg.inv(D + L) @ b
for _ in range(3):
    x = BS @ x + gS
assert abs(x[3] - 3.979) < 2e-3
assert abs(x[3] - 3.929) > 0.04

# 式 (2.4-6) 的不等号：lg q<0 时反向
# k lg q < lg ε - lg d + lg(1-q)  ⇒ k > 右端/lg q
q = 0.5
eps = 1e-3
d = 2.2727
rhs = (np.log10(eps) + np.log10(1 - q) - np.log10(d)) / np.log10(q)
assert 12.1 < rhs < 12.2

print("ALL OK")
