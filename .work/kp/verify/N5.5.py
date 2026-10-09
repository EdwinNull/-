"""N5.5 二元插值。运行：python3 .work/kp/verify/N5.5.py"""
import numpy as np

def f(x, y):
    return np.exp(x) * np.sin(y) + y - 0.1

def lagrange(xs, ys, x):
    s = 0.0
    for i in range(len(xs)):
        li = 1.0
        for j in range(len(xs)):
            if i != j:
                li *= (x - xs[j]) / (xs[i] - xs[j])
        s += ys[i] * li
    return s

assert abs(f(1.6, 0.33) - 1.83499562653897) < 1e-12
assert abs(round(f(1.6, 0.33), 4) - 1.8350) < 1e-12
assert abs(f(1.5, 0.3) - 1.5244296802581714) < 1e-12
assert abs(round(f(1.5, 0.3), 3) - 1.524) < 1e-12
assert abs(round(f(2.5, 0.3), 3) - 3.800) < 1e-12

xi = [1.0, 1.5, 2.0]
yj = [0.2, 0.3, 0.4, 0.5]
sub = [[round(f(x, y), 3) for y in yj] for x in xi]
assert sub[1][1] == 1.524
px = [lagrange(yj, row, 0.33) for row in sub]
assert abs(px[0] - 1.1107625) < 1e-9
assert abs(px[1] - 1.681847) < 1e-9
assert abs(px[2] - 2.624497) < 1e-9
L = lagrange(xi, px, 1.6)
assert abs(L - 1.84065176) < 1e-8
py = [lagrange(xi, [sub[i][j] for i in range(3)], 1.6) for j in range(4)]
assert abs(lagrange(yj, py, 0.33) - L) < 1e-9
assert abs(py[0] - 1.08736) < 1e-8
assert abs(py[1] - 1.66888) < 1e-8
assert abs(py[2] - 2.23572) < 1e-8
assert abs(py[3] - 2.78396) < 1e-8
assert abs(round(L, 3) - 1.841) < 1e-9

sub_bad = [row[:] for row in sub]
sub_bad[1][1] = 1.624
Lb = lagrange(xi, [lagrange(yj, row, 0.33) for row in sub_bad], 1.6)
assert abs(Lb - 1.91490776) < 1e-8

# 网格尺寸
assert max(np.diff(xi)) == 0.5
assert abs(max(np.diff(yj)) - 0.1) < 1e-12
assert max(0.5, 0.1) == 0.5

# 自编网格：加密中间点不改变 h
assert max(np.diff([0, 0.5, 1, 3])) == 2

# 双线性
def bilinear(x, y):
    return (1-x)*(1-y)*1 + (1-x)*y*2 + x*(1-y)*3 + x*y*6
assert abs(bilinear(0.5, 0.5) - 3) < 1e-12
assert abs(bilinear(0, 0) - 1) < 1e-12
assert abs(bilinear(0, 1) - 2) < 1e-12
assert abs(bilinear(1, 0) - 3) < 1e-12
assert abs(bilinear(1, 1) - 6) < 1e-12
assert abs(bilinear(0.25, 0.5) - 2.25) < 1e-12
# 改角点后不再等于四角平均
def bilinear2(x, y):
    return (1-x)*(1-y)*1 + (1-x)*y*2 + x*(1-y)*3 + x*y*0
assert abs(bilinear2(0.5, 0.5) - 1.5) < 1e-12

# 三角形
A = np.array([[0.0, 0, 1], [1, 0, 1], [0, 1, 1]])
coef = np.linalg.solve(A, np.array([1.0, 2, 4]))
assert np.allclose(coef, [1, 3, 1])
assert abs(0.5 + 1 - 1.5) < 1e-12

assert abs(1.91490776 - 1.84065176 - 0.074256) < 1e-6
assert abs(1.840652 - 1.834996 - 0.005656) < 1e-6
# 另一侧平面梯度
A2 = np.array([[0.0, 0, 1], [1, 0, 1], [1, 1, 1]], float)
coef2 = np.linalg.solve(A2, np.array([1.0, 2, 0]))
assert np.allclose(coef2, [1, -2, 1])

assert abs(round(f(0.5, 0.1), 3) - 0.165) < 1e-12
assert abs(round(f(3.0, 0.6), 3) - 11.841) < 1e-12
assert abs(1.840652 - 1.834996 - 0.005656) < 1e-5

assert abs((0.75*0.5)*1 + (0.75*0.5)*2 + (0.25*0.5)*3 + (0.25*0.5)*6 - 2.25) < 1e-12
assert abs((0.5 - 0 + 1) - 1.5) < 1e-12  # edge midpoint of z=x-2y+1 at (0.5,0)

assert abs(0.005656 / 1.8350 - 0.003082) < 1e-4

assert abs(1.6 - 1.5 - 0.1) < 1e-12
assert abs(0.33 - 0.3 - 0.03) < 1e-12

print("ALL OK")
