"""M3.2 矩阵范数 —— 讲解中全部数值的复算脚本。"""
import numpy as np

# K1 动机反例：元素最大值不满足相容性
A0 = np.ones((2, 2))
assert np.allclose(A0 @ A0, 2 * np.ones((2, 2)))
assert np.max(np.abs(A0 @ A0)) == 2
assert np.max(np.abs(A0)) * np.max(np.abs(A0)) == 1

# K2 Frobenius 与元素总和
C0 = np.array([[1 + 0j, 1j], [2 + 0j, -1j]])
assert abs(np.linalg.norm(C0, "fro") - np.sqrt(7)) < 1e-12
assert abs(np.sum(np.abs(C0)) - 5) < 1e-12

# K5 典型矩阵 A=[[1,-2],[3,4]]
A = np.array([[1.0, -2.0], [3.0, 4.0]])
assert np.linalg.norm(A, 1) == 6
assert np.linalg.norm(A, np.inf) == 7
AtA = A.T @ A
assert np.allclose(AtA, [[10, 10], [10, 20]])
eig = np.linalg.eigvalsh(AtA)
assert np.allclose(np.sort(eig), [15 - 5 * np.sqrt(5), 15 + 5 * np.sqrt(5)])
assert abs(np.linalg.norm(A, 2) - np.sqrt(15 + 5 * np.sqrt(5))) < 1e-12
assert abs(np.linalg.norm(A, 2) - 5.1167) < 5e-4

# K6 I-A inverse bound, A=(1/4)I
As = 0.25 * np.eye(2)
inv = np.linalg.inv(np.eye(2) - As)
assert np.allclose(inv, (4 / 3) * np.eye(2))
assert abs(np.linalg.norm(inv, np.inf) - 4 / 3) < 1e-12
assert abs(1 / (1 - 0.25) - 4 / 3) < 1e-12

# K7 textbook Example 5 condition numbers
M = np.array([[1.0, 5.0, -2.0], [-2.0, 1.0, 0.0], [3.0, -8.0, 2.0]])
Mi = np.linalg.inv(M)
expected_inv = -0.25 * np.array([[2.0, 6.0, 2.0], [4.0, 8.0, 4.0], [13.0, 23.0, 11.0]])
assert np.allclose(Mi, expected_inv)
assert np.linalg.norm(M, np.inf) == 13
assert np.linalg.norm(M, 1) == 14
assert abs(np.linalg.norm(Mi, np.inf) - 47 / 4) < 1e-12
assert abs(np.linalg.norm(Mi, 1) - 37 / 4) < 1e-12
assert abs(np.linalg.norm(M, np.inf) * np.linalg.norm(Mi, np.inf) - 611 / 4) < 1e-12
assert abs(np.linalg.norm(M, 1) * np.linalg.norm(Mi, 1) - 259 / 2) < 1e-12

# Typical Example 1 complex matrix C
C = np.array([[1 - 1j, 3], [2, 1 + 1j]], dtype=complex)
assert abs(np.linalg.norm(C, 1) - (3 + np.sqrt(2))) < 1e-12
assert abs(np.linalg.norm(C, np.inf) - (3 + np.sqrt(2))) < 1e-12
CHC = C.conj().T @ C
assert np.allclose(CHC, [[6, 5 + 5j], [5 - 5j, 11]])
assert np.allclose(np.sort(np.linalg.eigvalsh(CHC)), [1, 16])
assert abs(np.linalg.norm(C, 2) - 4) < 1e-12

# Typical Example 2 diagonal condition estimate
D = np.diag([2.0, 1.0])
Di = np.linalg.inv(D)
assert np.linalg.norm(D, np.inf) == 2
assert np.linalg.norm(Di, np.inf) == 1
assert np.linalg.cond(D, np.inf) == 2
assert abs(2 * 0.01 - 0.02) < 1e-12

# Typical Example 3 constructed vector norm
b = np.array([1.0, 2.0])
x = np.array([1.0, -1.0])
xb = np.outer(x, b)
assert np.allclose(xb, [[1, 2], [-1, -2]])
assert abs(np.linalg.norm(xb, "fro") - np.sqrt(10)) < 1e-12
assert abs(np.linalg.norm(b) ** 2 * np.linalg.norm(x) ** 2 - 10) < 1e-12

# Self-test values
assert abs(1 / (1 - 0.2) - 1.25) < 1e-12
assert abs(2 * 0.03 - 0.06) < 1e-12

print("ALL OK")
