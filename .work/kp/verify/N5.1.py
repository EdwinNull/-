"""N5.1 polynomial interpolation calculations, formulas and examples."""
import math
import numpy as np
import sympy as sp

x = sp.symbols('x')

# Textbook Example 1: four tabulated points and the interpolant coefficients.
xs = [-1, 1, 2, 5]
fs = [-7, 7, -4, 35]
p = sp.interpolate(list(zip(xs, fs)), x)
assert sp.expand(p) == 10 + 5*x - 10*x**2 + 2*x**3
assert [p.subs(x, z) for z in xs] == fs

# Lagrange basis cardinality and partition of unity for the same distinct nodes.
L = []
for i, xi in enumerate(xs):
    li = sp.prod((x-xj)/(xi-xj) for j, xj in enumerate(xs) if j != i)
    L.append(sp.expand(li))
for i, li in enumerate(L):
    assert li.subs(x, xs[i]) == 1
    assert all(li.subs(x, xs[j]) == 0 for j in range(len(xs)) if i != j)
assert sp.simplify(sum(L)-1) == 0
assert sp.expand(sum(fs[i]*L[i] for i in range(4))-p) == 0

# Textbook Example 2: two and three-node interpolants at x=0.3.
xe = [-2, -1, 0, 1, 2]
fe = [sp.Rational(1,4), sp.Rational(1,2), 1, 2, 4]
p1 = sp.interpolate([(0,1),(1,2)], x)
p2 = sp.interpolate([(-1,sp.Rational(1,2)),(0,1),(1,2)], x)
assert sp.expand(p1) == x + 1
assert sp.expand(p2) == sp.Rational(1,4)*x**2 + sp.Rational(3,4)*x + 1
assert p1.subs(x, sp.Rational(3,10)) == sp.Rational(13,10)
assert p2.subs(x, sp.Rational(3,10)) == sp.Rational(499,400)

# Example 3 divided-difference table values as printed; Newton forms interpolate nodes.
x3 = [1.0, 2.7, 3.2, 4.8, 5.6]
f3 = [14.2, 17.8, 22.0, 38.2, 51.7]
dd = [f3[:]]
for order in range(1, len(x3)):
    prev = dd[-1]
    dd.append([(prev[i+1]-prev[i])/(x3[i+order]-x3[i]) for i in range(len(prev)-1)])
assert np.allclose(dd[1], [2.1176470588235294, 8.4, 10.125, 16.875], atol=1e-12)
assert np.allclose(dd[2], [2.855614973262032, 0.8214285714285714, 2.8125], atol=1e-12)
assert abs(dd[3][0] + 0.534658) < 1e-3
# Build Newton form without expansion and verify all tabular values.
def newton_value(z, nodes, table):
    result = table[0][0]
    product = 1.0
    for k in range(1, len(nodes)):
        product *= z - nodes[k-1]
        result += table[k][0]*product
    return result
assert np.allclose([newton_value(z,x3,dd) for z in x3], f3, atol=1e-10)

# Remainder bound for linear interpolation of exp(x) on [0,1], n=1000.
n = 1000
h = 1/n
bound = math.e*h*h/8
assert bound < 0.5e-6
assert math.e/(8*824**2) > 0.5e-6
assert math.e/(8*825**2) < 0.5e-6

# Rounded log-table values: endpoint errors enter as convex weights.
err = 0.5e-5
for theta in np.linspace(0,1,101):
    delta = (1-theta)*err + theta*(-err)
    assert abs(delta) <= err + 1e-15

# Textbook Hermite Example 5 polynomial satisfies the prescribed values/derivatives.
H = -1 - 2*x + 3*x**2 + 6*x**2*(x-1) + 5*x**2*(x-1)**2
assert H.subs(x,0) == -1 and sp.diff(H,x).subs(x,0) == -2
assert H.subs(x,1) == 0 and sp.diff(H,x).subs(x,1) == 10
assert sp.diff(H,x,2).subs(x,1) == 40

print("ALL OK")
