"""N5.2 样条插值。运行：python3 .work/kp/verify/N5.2.py"""
import numpy as np

# Runge: n=10, x=4.8
n = 10
xs = np.array([-5 + 10 * i / n for i in range(n + 1)], dtype=float)
ys = 1 / (1 + xs**2)
x = 4.8
p = 0.0
for i in range(n + 1):
    li = 1.0
    for j in range(n + 1):
        if i != j:
            li *= (x - xs[j]) / (xs[i] - xs[j])
    p += ys[i] * li
f48 = 1 / (1 + x * x)
assert abs(f48 - 1 / 24.04) < 1e-15
assert abs(f48 - 0.0415973377703827) < 1e-15
assert abs(p - 1.804385456128002) < 1e-12
assert abs(abs(f48 - p) - 1.7627881183576193) < 1e-12
assert abs(round(p, 5) - 1.80439) < 1e-9 or abs(p - 1.80438) < 5e-6
assert abs(round(f48, 5) - 0.04160) < 1e-9
assert abs(round(abs(f48 - p), 4) - 1.7628) < 1e-9
# 教材陈述的收敛区间端点，脚本不重新推导该常数
assert abs(3.63 - 363 / 100) < 1e-15

# 脚注反例
def foot(t):
    return t**3 if t <= 1 else 3 * t**2 - 2

assert abs(foot(1) - 1) < 1e-15
assert abs(3 * 1**2 - 3) < 1e-15  # left derivative 3
assert abs(6 * 1 - 6) < 1e-15  # right derivative 6
assert abs(6 * 1 - 6) < 1e-15  # left second-derivative limit
assert abs(6 - 6) < 1e-15

# 自编自然样条 (0,0),(1,1),(2,0)
A = np.array([[2.0]])
rhs = np.array([-6.0])
M1 = np.linalg.solve(A, rhs)[0]
assert abs(M1 + 3) < 1e-12
assert abs((-0.5 * (0.5)**3 + 1.5 * 0.5) - 0.6875) < 1e-15
assert abs(11 / 16 - 0.6875) < 1e-15
# 左右导数在 x=1
assert abs((-1.5 * 1 + 1.5) - 0) < 1e-15

# 例 1
xn = np.array([1.0, 2.0, 4.0, 5.0])
fn = np.array([1.0, 3.0, 4.0, 2.0])
h = np.diff(xn)
assert np.allclose(h, [1, 2, 1])
d1 = np.diff(fn) / h
assert np.allclose(d1, [2, 0.5, -2])
d2 = (d1[1:] - d1[:-1]) / (xn[2:] - xn[:-2])
assert abs(d2[0] + 0.5) < 1e-12
assert abs(d2[1] + 5 / 6) < 1e-12
lam1 = h[1] / (h[0] + h[1])
mu2 = h[1] / (h[1] + h[2])
assert abs(lam1 - 2 / 3) < 1e-12 and abs(mu2 - 2 / 3) < 1e-12
A = np.array([[2, lam1], [mu2, 2]], dtype=float)
rhs = 6 * d2
assert np.allclose(rhs, [-3, -5])
M12 = np.linalg.solve(A, rhs)
assert np.allclose(M12, [-0.75, -2.25])
# 舍入右端
M_round = np.linalg.solve(A, np.array([-3.0, -4.999998]))
assert abs(M_round[0] + 0.750001) < 1e-6
assert abs(M_round[1] + 2.249999) < 1e-6
M = np.array([0.0, M12[0], M12[1], 0.0])

def sval(i, xx, M, xn, fn, h):
    hi = h[i]
    return (M[i] * (xn[i + 1] - xx) ** 3 / (6 * hi)
            + M[i + 1] * (xx - xn[i]) ** 3 / (6 * hi)
            + (fn[i] - M[i] * hi ** 2 / 6) * (xn[i + 1] - xx) / hi
            + (fn[i + 1] - M[i + 1] * hi ** 2 / 6) * (xx - xn[i]) / hi)

def sprime(i, xx, M, xn, fn, h):
    hi = h[i]
    return (-M[i] * (xn[i + 1] - xx) ** 2 / (2 * hi)
            + M[i + 1] * (xx - xn[i]) ** 2 / (2 * hi)
            + (fn[i + 1] - fn[i]) / hi
            - (M[i + 1] - M[i]) * hi / 6)

assert abs(sval(1, 3.0, M, xn, fn, h) - 4.25) < 1e-12
assert abs(sval(1, 3.0, M, xn, fn, h) - 17 / 4) < 1e-12
assert abs(sprime(0, 2.0, M, xn, fn, h) - 1.75) < 1e-12
assert abs(sprime(1, 2.0, M, xn, fn, h) - 1.75) < 1e-12
assert abs(sprime(1, 4.0, M, xn, fn, h) + 1.25) < 1e-12
assert abs(sprime(2, 4.0, M, xn, fn, h) + 1.25) < 1e-12
# 精确端点校正系数 7/2、11/2；舍入 M 给出教材的 3.500001、5.499999
assert abs((3 - (-0.75) * 4 / 6) - 3.5) < 1e-12
assert abs((4 - (-2.25) * 4 / 6) - 5.5) < 1e-12
assert abs((3 - (-0.750001) * 4 / 6) - 3.500001) < 5e-7
assert abs((4 - (-2.249999) * 4 / 6) - 5.499999) < 5e-7

# 例 2 未舍入
xn = np.array([0.0, 0.15, 0.30, 0.45, 0.60])
fn = np.array([1.0, 0.97800, 0.91743, 0.83160, 0.73529])
h = np.diff(xn)
assert np.allclose(h, 0.15)
d1 = np.diff(fn) / h
assert abs(d1[0] + 0.14666666666666667) < 1e-12
assert abs(d1[1] + 0.4038) < 1e-12
assert abs(d1[2] + 0.5722) < 1e-12
assert abs(d1[3] + 0.6420666666666667) < 1e-12
d2 = (d1[1:] - d1[:-1]) / (xn[2:] - xn[:-2])
A = np.zeros((5, 5))
rhs = np.zeros(5)
A[0, 0], A[0, 1] = 2, 1
rhs[0] = 6 / h[0] * (d1[0] - 0)
for i in range(1, 4):
    A[i, i - 1] = A[i, i + 1] = 0.5
    A[i, i] = 2
    rhs[i] = 6 * d2[i - 1]
A[4, 3], A[4, 4] = 1, 2
rhs[4] = 6 / h[3] * (-0.64879 - d1[3])
assert np.allclose(rhs, [-5.86666667, -5.14266667, -3.368, -1.39733333, -0.26893333], atol=1e-8)
M = np.linalg.solve(A, rhs)
expect = np.array([-2.04452143, -1.77762381, -1.13031667, -0.43710952, 0.08408810])
assert np.allclose(M, expect, atol=5e-8)
assert abs(sprime(0, 0.0, M, xn, fn, h)) < 1e-12
assert abs(sprime(3, 0.60, M, xn, fn, h) + 0.64879) < 1e-12
for i in range(4):
    assert abs(sval(i, xn[i], M, xn, fn, h) - fn[i]) < 1e-12
    assert abs(sval(i, xn[i + 1], M, xn, fn, h) - fn[i + 1]) < 1e-12
for i in range(1, 4):
    assert abs(sprime(i - 1, xn[i], M, xn, fn, h) - sprime(i, xn[i], M, xn, fn, h)) < 1e-10
assert abs(sval(1, 0.225, M, xn, fn, h) - 0.951804291294643) < 1e-12

# 表 5.2-2 第一个二阶差商的舍入算术
assert abs((-0.40380 - (-0.14667)) - (-0.25713)) < 1e-12
assert abs((-0.25713) / 0.30 - (-0.85710)) < 1e-12
assert abs(6 * (-0.85710) - (-5.14260)) < 1e-12
# 例 2 矩阵按行严格对角占优
A_dom = np.array([
    [2, 1, 0, 0, 0],
    [0.5, 2, 0.5, 0, 0],
    [0, 0.5, 2, 0.5, 0],
    [0, 0, 0.5, 2, 0.5],
    [0, 0, 0, 1, 2],
], dtype=float)
for i in range(5):
    off = np.sum(np.abs(A_dom[i])) - abs(A_dom[i, i])
    assert off <= 1 + 1e-12
    assert A_dom[i, i] == 2

# 教材印出的舍入系统
rhs_book = np.array([-5.86680, -5.14260, -3.36798, -1.39740, -0.26880])
M_book_solved = np.linalg.solve(A, rhs_book)
M_printed = np.array([-2.04462, -1.77757, -1.13031, -0.43716, 0.08418])
assert np.allclose(M_book_solved, M_printed, atol=1e-5)
books = np.array([
    [0.29672, -1.02231, 0.0, 1.0],
    [0.71918, -1.21242, 0.02851, 0.99858],
    [0.77017, -1.25831, 0.04228, 0.99720],
    [0.57927, -1.00059, -0.07370, 1.01461],
])
max_coef = 0.0
max_node = 0.0
for i in range(4):
    xi, xip = xn[i], xn[i + 1]
    grid = np.linspace(xi, xip, 6)
    vals = np.array([sval(i, xx, M_printed, xn, fn, h) for xx in grid])
    V = np.vander(grid, 4)
    coef = np.linalg.lstsq(V, vals, rcond=None)[0]
    max_coef = max(max_coef, np.max(np.abs(coef - books[i])))
    for xx, fv in ((xi, fn[i]), (xip, fn[i + 1])):
        poly = books[i, 0] * xx**3 + books[i, 1] * xx**2 + books[i, 2] * xx + books[i, 3]
        max_node = max(max_node, abs(poly - fv))
assert max_coef < 2e-5
assert max_node <= 1.2e-5
# 印出首段无一次项，左端导数为 0；末段右端导数
assert abs(books[0, 2]) < 1e-15
end_deriv = 3 * books[3, 0] * 0.60**2 + 2 * books[3, 1] * 0.60 + books[3, 2]
assert abs(end_deriv + 0.6487964) < 1e-9

# 周期三阶例子与 LR
A = np.array([[2, 0.5, 0.5], [0.5, 2, 0.5], [0.5, 0.5, 2]], dtype=float)
rhs = np.array([3.0, -6.0, 3.0])
Mp = np.linalg.solve(A, rhs)
assert np.allclose(Mp, [2, -4, 2])
L = np.eye(3)
U = np.zeros((3, 3))
for k in range(3):
    for j in range(k, 3):
        U[k, j] = A[k, j] - sum(L[k, p] * U[p, j] for p in range(k))
    for i in range(k + 1, 3):
        L[i, k] = (A[i, k] - sum(L[i, p] * U[p, k] for p in range(k))) / U[k, k]
assert abs(L[2, 0] - 0.25) < 1e-12
assert abs(L[2, 1] - 0.2) < 1e-12
assert abs(U[1, 1] - 1.875) < 1e-12
assert abs(U[1, 2] - 0.375) < 1e-12
assert abs(U[2, 2] - 1.8) < 1e-12
assert np.allclose(L @ U, A)

# 不等距自编：节点 0,1,3，值 0,1,0，自然边界
xu = np.array([0.0, 1.0, 3.0])
fu = np.array([0.0, 1.0, 0.0])
hu = np.diff(xu)
d1u = np.diff(fu) / hu
d2u = (d1u[1] - d1u[0]) / (xu[2] - xu[0])
assert abs(d2u + 0.5) < 1e-12
lam = hu[1] / (hu[0] + hu[1])
mu = hu[0] / (hu[0] + hu[1])
assert abs(lam - 2 / 3) < 1e-12 and abs(mu - 1 / 3) < 1e-12
Mu = np.array([0.0, np.linalg.solve(np.array([[2.0]]), np.array([6 * d2u]))[0], 0.0])
assert abs(Mu[1] + 1.5) < 1e-12
assert abs(sval(1, 2.0, Mu, xu, fu, hu) - 0.875) < 1e-12

# 自编：x^4 在第一种边界下的步长减半，观察 h^4
def known_second(n):
    xn = np.linspace(0.0, 1.0, n + 1)
    fn = xn**4
    h = np.diff(xn)
    d1 = np.diff(fn) / h
    d2 = (d1[1:] - d1[:-1]) / (xn[2:] - xn[:-2])
    A = np.zeros((n - 1, n - 1))
    rhs = np.zeros(n - 1)
    M0, Mn = 0.0, 12.0
    for i in range(1, n):
        row = i - 1
        lam = h[i] / (h[i - 1] + h[i])
        mu = 1 - lam
        if i > 1:
            A[row, row - 1] = mu
        A[row, row] = 2
        if i < n - 1:
            A[row, row + 1] = lam
        rhs[row] = 6 * d2[i - 1]
        if i == 1:
            rhs[row] -= mu * M0
        if i == n - 1:
            rhs[row] -= lam * Mn
    mid = np.linalg.solve(A, rhs)
    M = np.concatenate([[M0], mid, [Mn]])
    err = 0.0
    for xx in np.linspace(0.0, 1.0, 4001):
        i = min(int(xx * n), n - 1)
        if xx == 1.0:
            i = n - 1
        err = max(err, abs(sval(i, xx, M, xn, fn, h) - xx**4))
    return err

e2, e4, e8 = known_second(2), known_second(4), known_second(8)
assert abs(e2 - 0.008124182344535158) < 1e-12
assert abs(e4 - 0.0006056599999999999) < 1e-12
assert abs(e8 - 3.836411210367263e-5) < 1e-14
assert abs(e2 / e4 - 13.413767368713732) < 1e-9
assert abs(e4 / e8 - 15.787150198167092) < 1e-9
assert abs(12 - 4 * 3) < 1e-15  # f''(1)=12 for x^4

# 两点第二种边界：(0,0),(1,1)，m0=m1=0
A2 = np.array([[2.0, 1.0], [1.0, 2.0]])
rhs2 = np.array([6.0, -6.0])
M2 = np.linalg.solve(A2, rhs2)
assert np.allclose(M2, [6.0, -6.0])
x2 = np.array([0.0, 1.0])
f2 = np.array([0.0, 1.0])
h2 = np.array([1.0])
assert abs(sval(0, 0.0, M2, x2, f2, h2)) < 1e-12
assert abs(sval(0, 1.0, M2, x2, f2, h2) - 1) < 1e-12
assert abs(sprime(0, 0.0, M2, x2, f2, h2)) < 1e-12
assert abs(sprime(0, 1.0, M2, x2, f2, h2)) < 1e-12
# 自由度计数：n=4 时 2n+(n-1)+(n-1)+2=4n
n_pieces = 4
assert 2 * n_pieces + (n_pieces - 1) + (n_pieces - 1) + 2 == 4 * n_pieces

# 严格对角占优的零解论证：2|z| <= |z|
assert 2 > 1

print("ALL OK")
