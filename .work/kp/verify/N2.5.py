"""N2.5 SOR 与块迭代。运行：python3 .work/kp/verify/N2.5.py"""
import numpy as np

A = np.array([[4.0, 3, 0], [3, 4, -1], [0, -1, 4]])
b = np.array([24.0, 30, -24])
assert np.allclose(np.linalg.solve(A, b), [3, 4, -5])

def seidel(x):
    x1 = -0.75 * x[1] + 6
    x2 = -0.75 * x1 + 0.25 * x[2] + 7.5
    x3 = 0.25 * x2 - 6
    return np.array([x1, x2, x3])

def sor(x, w=1.25):
    x1 = (1 - w) * x[0] + (w / 4) * (-3 * x[1] + 24)
    x2 = (1 - w) * x[1] + (w / 4) * (-3 * x1 + x[2] + 30)
    x3 = (1 - w) * x[2] + (w / 4) * (x2 - 24)
    return np.array([x1, x2, x3])

x = np.ones(3)
for _ in range(7):
    x = seidel(x)
assert np.allclose(x, [3.0134110, 3.9888241, -5.0027940], atol=5e-7)
x = np.ones(3)
for _ in range(7):
    x = sor(x)
assert np.allclose(x, [3.0000498, 4.0002586, -5.0003486], atol=5e-7)

def steps(fn):
    x = np.ones(3)
    exact = np.array([3.0, 4, -5])
    for k in range(1, 80):
        x = fn(x)
        if np.max(np.abs(x - exact)) < 5e-8:
            return k
    raise AssertionError("did not reach 7 decimals")

assert steps(seidel) == 34
assert steps(sor) == 14

# 例 2
D = np.diag(np.diag(A))
L = np.tril(A, -1)
R = np.triu(A, 1)
assert np.allclose(D + L + R, A)
BJ = -np.linalg.inv(D) @ (L + R)
assert np.allclose(BJ, [[0, -0.75, 0], [-0.75, 0, 0.25], [0, 0.25, 0]])
rhoJ = max(abs(np.linalg.eigvals(BJ)))
assert abs(rhoJ - np.sqrt(0.625)) < 1e-12
BS = -np.linalg.inv(D + L) @ R
assert abs(BS[1, 1] - 0.75**2) < 1e-12
assert abs(max(abs(np.linalg.eigvals(BS))) - 0.625) < 1e-12
wopt = 2 / (1 + np.sqrt(1 - 0.625))
assert abs(wopt - 1.240) < 5e-4
assert abs((wopt - 1) - 0.240) < 5e-4

# 定理 2.5-1：特征值之积为 (1-ω)^n
w = 1.25
Bw = np.linalg.inv(D + w * L) @ ((1 - w) * D - w * R)
assert abs(np.linalg.det(Bw) - (1 - w) ** 3) < 1e-10

print("ALL OK")
