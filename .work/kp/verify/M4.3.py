"""M4.3 函数矩阵及其应用 —— 讲解中全部数值的复算脚本。
运行：python3 .work/kp/verify/M4.3.py
"""
import sympy as sp

t, tau = sp.symbols("t tau", real=True)

# K1 反例：A=[[t, 1],[0, 0]]
A = sp.Matrix([[t, 1], [0, 0]])
Ap = sp.diff(A, t)
A2 = sp.simplify(A * A)
dA2 = sp.diff(A2, t)
assert A2 == sp.Matrix([[t ** 2, t], [0, 0]])
assert dA2 == sp.Matrix([[2 * t, 1], [0, 0]])
assert sp.simplify(2 * A * Ap) == sp.Matrix([[2 * t, 0], [0, 0]])
assert sp.simplify(Ap * A + A * Ap) == dA2
assert sp.simplify(2 * A * Ap - dA2) != sp.zeros(2)

# 短例积分
B = sp.Matrix([[t, 1], [0, t ** 2]])
assert sp.integrate(B, (t, 0, 1)) == sp.Matrix([[sp.Rational(1, 2), 1], [0, sp.Rational(1, 3)]])
assert sp.diff(B, t) == sp.Matrix([[1, 0], [0, 2 * t]])

# 范数估计的等式情形
C = sp.Matrix([[t, 0], [0, 0]])
integ = sp.integrate(C, (t, 0, 1))
assert float(sp.Matrix(integ).norm(1)) == 0.5
assert sp.integrate(sp.Matrix(C).norm(1), (t, 0, 1)) == sp.Rational(1, 2)

# 定理 4.3-2 对角例子
D = sp.Matrix([[1 + t, 0], [0, 2 + t]])
Dinv = sp.simplify(D.inv())
lhs = sp.diff(Dinv, t)
rhs = sp.simplify(-Dinv * sp.diff(D, t) * Dinv)
assert sp.simplify(lhs - rhs) == sp.zeros(2)
assert Dinv == sp.diag(1 / (1 + t), 1 / (2 + t))

# 例 4 的系数
u, v = sp.symbols("u v", cls=sp.Function)
# 只核对手写的线性组合
# x2' = -4 x1 - x3 - 3 x4 + 8t
# x4' = -4 x1 + x2 + x3 - 3 x4 + 8t - cos t
x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4")
A4 = sp.Matrix([[0, 1, 0, 0], [-4, 0, -1, -3], [0, 0, 0, 1], [-4, 1, 1, -3]])
g4 = sp.Matrix([0, 8 * t, 0, 8 * t - sp.cos(t)])
xx = sp.Matrix([x1, x2, x3, x4])
rhs4 = sp.simplify(A4 * xx + g4)
assert rhs4[0] == x2
assert rhs4[1] == -4 * x1 - x3 - 3 * x4 + 8 * t
assert rhs4[2] == x4
assert sp.simplify(rhs4[3] - (-4 * x1 + x2 + x3 - 3 * x4 + 8 * t - sp.cos(t))) == 0

# 例 5
M = sp.Matrix([[1, 2], [4, 3]])
lam = sp.symbols("lam")
assert sp.factor(M.charpoly(lam).as_expr()) == (lam - 5) * (lam + 1)
eAt = sp.simplify((t * M).exp())
tb = sp.Rational(1, 6) * sp.Matrix([
    [2 * sp.exp(5 * t) + 4 * sp.exp(-t), 2 * sp.exp(5 * t) - 2 * sp.exp(-t)],
    [4 * sp.exp(5 * t) - 4 * sp.exp(-t), 4 * sp.exp(5 * t) + 2 * sp.exp(-t)],
])
assert sp.simplify(eAt - tb) == sp.zeros(2)
g = sp.Matrix([1, -1])
I = sp.integrate(sp.simplify(((t - tau) * M).exp()) * g, (tau, 0, t))
x = sp.simplify(eAt * sp.Matrix([1, 2]) + I)
tbx = sp.Matrix([1 - sp.exp(-t) + sp.exp(5 * t), -1 + sp.exp(-t) + 2 * sp.exp(5 * t)])
assert sp.simplify(x - tbx) == sp.zeros(2, 1)
assert sp.simplify(x.subs(t, 0)) == sp.Matrix([1, 2])
assert sp.simplify(sp.diff(x, t) - M * x - g) == sp.zeros(2, 1)
hom = sp.simplify(eAt * sp.Matrix([1, 2]))
assert sp.simplify(hom - sp.Matrix([sp.exp(5 * t), 2 * sp.exp(5 * t)])) == sp.zeros(2, 1)
assert sp.simplify(I - sp.Matrix([1 - sp.exp(-t), -1 + sp.exp(-t)])) == sp.zeros(2, 1)

# 习题 4.3 第 1 题导航中的矩阵
# x'''-4x''+3x'+8x = t sin t
# x3' = 4 x3 - 3 x2 - 8 x1 + t sin t
Aex = sp.Matrix([[0, 1, 0], [0, 0, 1], [-8, -3, 4]])
assert (Aex * sp.Matrix([x1, x2, x3]))[2] == -8 * x1 - 3 * x2 + 4 * x3

# 习题 4.3 第 2 题答案
s, c = sp.sinh(t), sp.cosh(t)
xex = sp.Matrix([-s, c - 2 * s + 1])
Aex2 = sp.Matrix([[2, -1], [3, -2]])
gex = sp.Matrix([1, 2])
assert sp.simplify(xex.subs(t, 0)) == sp.Matrix([0, 2])
assert sp.simplify(sp.diff(xex, t) - Aex2 * xex - gex) == sp.zeros(2, 1)
assert sp.simplify(xex[0] - (sp.exp(-t) - sp.exp(t)) / 2) == 0
assert sp.simplify(xex[1] - ((3 * sp.exp(-t) - sp.exp(t)) / 2 + 1)) == 0

# 例题 4：A=diag(1,2)，B=[3]，X0=[1,4]
Aa = sp.diag(1, 2)
Xt = sp.simplify((t * Aa).exp() * sp.Matrix([1, 4]) * sp.exp(3 * t))
assert sp.simplify(Xt - sp.Matrix([sp.exp(4 * t), 4 * sp.exp(5 * t)])) == sp.zeros(2, 1)
assert sp.simplify(sp.diff(Xt, t) - Aa * Xt - Xt * 3) == sp.zeros(2, 1)
assert sp.simplify(Xt.subs(t, 0)) == sp.Matrix([1, 4])

print("ALL OK")
