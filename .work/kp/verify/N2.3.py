"""N2.3 直接法的误差分析。运行：python3 .work/kp/verify/N2.3.py"""
import numpy as np

A = np.array([[1.0, 2.0], [0.4990, 1.001]])
b = np.array([3.0, 1.5])
x_star = np.array([2.0, 0.5])
x = np.array([1.0, 1.0])
r = b - A @ x_star
e = x_star - x
assert np.allclose(r, [0.0, 0.0015])
assert np.allclose(e, [1.0, -0.5])
assert abs(np.linalg.norm(r, np.inf) - 0.0015) < 1e-12
assert abs(np.linalg.norm(e, np.inf) - 1.0) < 1e-12
assert abs(1.0 / 0.0015 - 666.6666666667) < 1e-6
# e = -A^{-1} r
assert np.allclose(e, -np.linalg.solve(A, r))
Ainf = np.linalg.norm(A, np.inf)
Ainv_inf = np.linalg.norm(np.linalg.inv(A), np.inf)
assert abs(Ainf - 3.0) < 1e-12
assert abs(Ainv_inf - 1000.3333333333) < 1e-6
assert abs(Ainf * Ainv_inf - 3001.0) < 1e-6

# 例 2：准确解与残差
A2 = np.array(
    [[3.3330, 15920, -10.333], [2.2220, 16.770, 9.6120], [1.5611, 5.1791, 1.6852]],
    dtype=float,
)
b2 = np.array([15913.0, 28.604, 8.4254])
assert np.allclose(A2 @ np.ones(3), b2)
x_printed = np.array([1.2001, 0.99991, 0.92538])
r_printed = b2 - A2 @ x_printed
assert np.allclose(r_printed, [-0.00518, 0.27413, -0.18616], atol=5e-6)
# 五位回代与印出的 x1 不同
assert abs(1.2001 + (-0.20008) - 1.00002) < 1e-12
# 条件数估计：用 |Δx1|=0.20008 才得到书上的 16672
est = 0.20008 / 1.2001 * 1e5
assert abs(est - 16672) < 1
wrong = 0.2008 / 1.2001 * 1e5
assert abs(wrong - 16672) > 50
cond = np.linalg.cond(A2, np.inf)
assert abs(cond - 15999) / 15999 < 0.002

print("ALL OK")
