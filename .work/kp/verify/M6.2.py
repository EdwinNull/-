"""M6.2 线性变换。运行：python3 .work/kp/verify/M6.2.py"""
import sympy as sp

# 例 6
M = sp.Matrix([[5, 4, 2], [4, 5, 2], [2, 2, 2]])
B = sp.Matrix.hstack(sp.Matrix([2, 2, 1]), sp.Matrix([-1, 1, 0]), sp.Matrix([-1, 0, 2]))
assert B.det() == 9
coords = [B.solve(M * B.col(j)) for j in range(3)]
A6 = sp.Matrix.hstack(*coords)
assert A6 == sp.diag(10, 1, 1)
A0 = M
assert sp.simplify(B.inv() * A0 * B) == sp.diag(10, 1, 1)
assert sp.simplify(B * A0 * B.inv()) != sp.diag(10, 1, 1)

# 例 7
A7 = sp.Matrix([[1, 0, 2], [-1, 2, 1], [1, 2, 5]])
assert A7.rank() == 2
ns = A7.nullspace()
assert len(ns) == 1
assert sp.simplify(ns[0] / ns[0][2]) == sp.Matrix([-2, sp.Rational(-3, 2), 1])
assert A7.col(2) == 2 * A7.col(0) + sp.Rational(3, 2) * A7.col(1)

# 例 9
t = sp.symbols("t")
def Tp(p):
    return sp.expand(p + (t + 1) * sp.diff(p, t))
basis = [1, t, t**2]
cols = []
for p in basis:
    im = Tp(p)
    # coordinates in {1,t,t^2}
    poly = sp.Poly(sp.expand(im), t)
    cols.append(sp.Matrix([poly.coeff_monomial(t**k) for k in range(3)]))
A9 = sp.Matrix.hstack(*cols)
assert A9 == sp.Matrix([[1, 1, 0], [0, 2, 2], [0, 0, 3]])
assert (A9 - sp.eye(3)) * sp.Matrix([1, 0, 0]) == sp.zeros(3, 1)
assert (A9 - 2 * sp.eye(3)) * sp.Matrix([1, 1, 0]) == sp.zeros(3, 1)
assert (A9 - 3 * sp.eye(3)) * sp.Matrix([1, 2, 1]) == sp.zeros(3, 1)
assert sp.expand(1 + 2 * t + t**2 - (1 + t) ** 2) == 0

# 例 10
A10 = sp.Matrix([[0, 1, 0], [0, 0, 2], [0, 0, 0]])
assert A10.rank() == 2
lam = sp.symbols("lam")
assert sp.factor((A10 - lam * sp.eye(3)).det()) == (-lam) ** 3

# 例 11
A11 = sp.Matrix([[4, 0, 1], [2, 3, 2], [1, 0, 4]])
assert sp.factor(A11.charpoly(lam).as_expr()) == (lam - 5) * (lam - 3) ** 2
for val, vec in [(5, [1, 2, 1]), (3, [1, 0, -1]), (3, [0, 1, 0])]:
    assert (A11 - val * sp.eye(3)) * sp.Matrix(vec) == sp.zeros(3, 1)
P = sp.Matrix([[1, 1, 0], [2, 0, 1], [1, -1, 0]])
assert sp.simplify(P.inv() * A11 * P) == sp.diag(5, 3, 3)

print("ALL OK")
