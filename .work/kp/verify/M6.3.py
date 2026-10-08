"""M6.3 内积空间。运行：python3 .work/kp/verify/M6.3.py"""
import sympy as sp

# 极化恒等式：第一变元线性、第二变元共轭线性
a, b = 1 + 2j, -3 + 0.5j
def ip(x, y):
    return x * y.conjugate()
def n2(x):
    return ip(x, x).real
lhs = ip(a, b)
rhs = 0.25 * (n2(a + b) - n2(a - b) + 1j * n2(a + 1j * b) - 1j * n2(a - 1j * b))
assert abs(lhs - rhs) < 1e-12

# 例 1 的权：⟨1, t⟩ 在对称区间上为零，⟨1,1⟩>0
t = sp.symbols("t")
assert sp.integrate(t**2 * 1 * t, (t, -1, 1)) == 0
assert sp.integrate(t**2 * 1 * 1, (t, -1, 1)) == sp.Rational(2, 3)

# 式 (6.3-1)，ρ=1 时 ⟨1,1⟩=b-a>0
a0, b0 = sp.symbols("a0 b0", real=True)
assert sp.simplify(sp.integrate(1, (t, a0, b0)) - (b0 - a0)) == 0

# 酉矩阵保持内积与长度
th = sp.symbols("th", real=True)
Q = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])
assert sp.simplify(Q.T * Q) == sp.eye(2)
x = sp.Matrix([1, 2])
assert sp.simplify((Q * x).norm() ** 2 - x.norm() ** 2) == 0

# 定理 6.3-3：Hermite 矩阵特征值实、不同特征值的特征向量正交
H = sp.Matrix([[2, 1 - sp.I], [1 + sp.I, 3]])
assert sp.simplify(H - H.H) == sp.zeros(2)
vals = H.eigenvects()
lams = [sp.simplify(v[0]) for v in vals]
assert all(sp.im(L) == 0 for L in lams)
v1 = vals[0][2][0]
v2 = vals[1][2][0]
assert sp.simplify(v1.H * v2)[0] == 0

print("ALL OK")
