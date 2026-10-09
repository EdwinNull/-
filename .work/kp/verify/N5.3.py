"""N5.3 数据的最小二乘拟合。运行：python3 .work/kp/verify/N5.3.py"""
import numpy as np

# 自编三点直线
G = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
y = np.array([1.0, 2.0, 2.0])
A = G.T @ G
rhs = G.T @ y
assert np.allclose(A, [[3, 3], [3, 5]])
assert np.allclose(rhs, [5, 6])
a = np.linalg.solve(A, rhs)
assert abs(a[0] - 7 / 6) < 1e-12
assert abs(a[1] - 0.5) < 1e-12
resid = G @ a - y
assert np.allclose(resid, [1 / 6, -1 / 3, 1 / 6])
assert abs(resid @ np.ones(3)) < 1e-12
assert abs(resid @ np.array([0.0, 1.0, 2.0])) < 1e-12
assert abs(np.sum(resid ** 2) - 1 / 6) < 1e-12
Q, R = np.linalg.qr(G, mode="complete")
h = Q.T @ y
a_qr = np.linalg.solve(R[:2, :2], h[:2])
assert np.allclose(a_qr, a)
assert abs(np.linalg.norm(h[2:]) - np.sqrt(1 / 6)) < 1e-12

# 加权
W = np.diag([1.0, 1.0, 4.0])
Aw = G.T @ W @ G
rw = G.T @ W @ y
assert np.allclose(Aw, [[6, 9], [9, 17]])
assert np.allclose(rw, [11, 18])
aw = np.linalg.solve(Aw, rw)
assert abs(aw[0] - 25 / 21) < 1e-12
assert abs(aw[1] - 3 / 7) < 1e-12
assert abs((G @ aw - y) @ W @ (G @ aw - y) - 4 / 21) < 1e-12

# 例 1
x = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
y1 = np.array([0.9, 1.9, 2.8, 3.3, 4.2])
assert abs(x.sum() - 2) < 1e-12
assert abs((x ** 2).sum() - 1.2) < 1e-12
assert abs(y1.sum() - 13.1) < 1e-12
assert abs((x * y1).sum() - 6.84) < 1e-12
G1 = np.column_stack([np.ones_like(x), x])
a1 = np.linalg.solve(G1.T @ G1, G1.T @ y1)
assert abs(a1[0] - 1.02) < 1e-12
assert abs(a1[1] - 4) < 1e-12
fit = 1.02 + 4 * x
assert np.allclose(fit, [1.02, 1.82, 2.62, 3.42, 4.22])
res = fit - y1
assert np.allclose(res, [0.12, -0.08, -0.18, 0.12, 0.02])
assert abs(np.sum(res ** 2) - 0.068) < 1e-12
res_bad = 1.04 + 4 * x - y1
assert np.allclose(res_bad, [0.14, -0.06, -0.16, 0.14, 0.04])
assert abs(np.sum(res_bad ** 2) - 0.070) < 1e-12
assert abs(5 * 1.04 + 8 - 13.2) < 1e-12

# 例 2 双曲线
x2 = np.array([2, 3, 4, 7, 8, 10, 11, 14, 16, 18, 19], float)
y2 = np.array([106.42, 108.20, 109.50, 110.00, 109.93, 110.49, 110.59, 110.60, 110.76, 111.00, 111.20])
assert len(x2) == 11
inv = 1 / x2
assert abs(inv.sum() - 1.7842152730310623) < 1e-12
assert abs((inv ** 2).sum() - 0.49277353085824055) < 1e-12
assert abs(y2.sum() - 1208.69) < 1e-12
assert abs((y2 / x2).sum() - 194.0516369902028) < 1e-12
G2 = np.column_stack([np.ones_like(x2), inv])
a2 = np.linalg.solve(G2.T @ G2, G2.T @ y2)
assert abs(a2[0] - 111.47568291) < 1e-7
assert abs(a2[1] + 9.83206022) < 1e-7
d2 = np.sum((G2 @ a2 - y2) ** 2)
assert abs(d2 - 0.4613042103488025) < 1e-10
ab = np.array([111.4738, -9.8206])
assert abs(np.sum((G2 @ ab - y2) ** 2) - 0.46133092679827914) < 1e-12
rhs_book = np.array([1208.69, 194.052])
ident = y2 @ y2 - rhs_book @ ab
assert abs(ident - 0.516849200008437) < 1e-9
dropped = ab @ (G2.T @ (G2 @ ab - y2))
assert abs(dropped + 0.05195329919277304) < 1e-12
# x=2 处精确系数
assert abs(a2[0] + a2[1] / 2 - 106.5596528) < 1e-6

# 指数模型
lny = np.log(y2)
book_lny = np.array([4.66739, 4.68398, 4.69592, 4.70048, 4.69984, 4.70493, 4.70583, 4.70592, 4.70737, 4.70953, 4.71133])
assert np.max(np.abs(lny - book_lny)) < 5e-6
c = np.linalg.solve(G2.T @ G2, G2.T @ lny)
phi = np.exp(c[0]) * np.exp(c[1] / x2)
assert abs(np.sum((phi - y2) ** 2) - 0.47193209161234273) < 1e-12
Ae = np.array([[11, 1.7842], [1.7842, 0.49277]])
ce = np.linalg.solve(Ae, np.array([51.6925, 8.36623]))
assert abs(ce[0] - 4.71390817) < 1e-7
assert abs(ce[1] + 0.08995059) < 1e-7
assert abs(np.exp(ce[0]) - 111.48701957) < 1e-6

# 例 3
x3 = np.array([1.2, 2.8, 4.3, 5.4, 6.8, 7.9])
y3 = np.array([2.1, 11.5, 28.1, 41.9, 72.3, 91.4])
assert abs(np.log10(1.2) - 0.07918124604762482) < 1e-12
assert abs(np.log10(7.9) - 0.8976270912904414) < 1e-12
G3 = np.column_stack([np.ones_like(x3), np.log10(x3)])
c3 = np.linalg.solve(G3.T @ G3, G3.T @ np.log10(y3))
a3 = 10 ** c3[0]
b3 = c3[1]
assert abs(a3 - 1.4540186484193842) < 1e-12
assert abs(b3 - 2.01485977) < 1e-8
phi3 = a3 * x3 ** b3
assert abs(np.sum((phi3 - y3) ** 2) - 17.368105540686802) < 1e-10
tab_d = np.array([-0.001, 0.074, -0.628, 1.573, -3.125, 2.176])
assert abs(np.sum(tab_d ** 2) - 17.374791) < 1e-9
# 书上舍入正规方程
cbook = np.linalg.solve(np.array([[6, 3.6224], [3.6224, 2.6427]]), np.array([8.2738, 5.9135]))
assert abs(cbook[0] - 0.16241439) < 1e-7
assert abs(cbook[1] - 2.01504905) < 1e-7
assert abs(10 ** cbook[0] - 1.45349782) < 1e-7

# 病态例子
eps = 0.5e-5
assert abs(eps - 5e-6) < 1e-18
assert abs(eps ** 2 - 0.25e-10) < 1e-22
assert abs(eps ** 2 - 2.5e-11) < 1e-22

print("ALL OK")
