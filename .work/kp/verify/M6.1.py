"""M6.1 线性空间。运行：python3 .work/kp/verify/M6.1.py"""
import sympy as sp

lam = sp.symbols("lam")
# 例 6
assert sp.expand(1 * 1 + (-1) * lam + 2 * lam**2) == 2 * lam**2 - lam + 1
assert sp.expand(-3 * (lam + 1) + 2 * (lam + 2) + 2 * lam**2) == 2 * lam**2 - lam + 1

# 例 8
old = [1, lam - 1, (lam - 1) ** 2]
cols = [sp.Matrix([2, 1, 0]), sp.Matrix([1, -1, 0]), sp.Matrix([1, 1, 1])]
targets = [1 + lam, 2 - lam, lam**2 - lam + 1]
for c, t in zip(cols, targets):
    expr = c[0] * old[0] + c[1] * old[1] + c[2] * old[2]
    assert sp.expand(expr - t) == 0
P8 = sp.Matrix.hstack(*cols)
assert P8 == sp.Matrix([[2, 1, 1], [1, -1, 1], [0, 0, 1]])
assert P8.det() == -3

# 例 9：新基 = 旧基 P
old9 = sp.Matrix([[-1, 1, 0], [1, 0, 1], [1, -1, 1]])
new9 = sp.Matrix([[1, 0, 1], [0, 0, 1], [1, 1, 1]])
P9 = sp.Matrix([[-2, -1, -1], [-1, -1, 0], [2, 1, 2]])
assert old9 * P9 == new9
assert sp.simplify(old9.inv() * new9) == P9

# 例 10
B2 = sp.Matrix([[-3, 3, -2], [-7, 6, -3], [1, -1, 2]])
A = sp.eye(3) - B2
assert A == sp.Matrix([[4, -3, 2], [7, -5, 3], [-1, 1, -1]])
ns = A.nullspace()
assert len(ns) == 1
assert sp.simplify(ns[0] / ns[0][0]) == sp.Matrix([1, 2, 1])

# 例 5：零元素是 1，a 的负元素是 1/a
a, b, k, l = sp.symbols("a b k l", positive=True)
assert sp.simplify(a * 1 - a) == 0
assert sp.simplify(a * (1 / a) - 1) == 0
assert sp.simplify((a * b) ** k - (a**k) * (b**k)) == 0
assert sp.simplify((a**l) ** k - a ** (k * l)) == 0

print("ALL OK")
