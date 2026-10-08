"""N1.2 误差估计与稳定性。运行：python3 .work/kp/verify/N1.2.py"""
import sympy as sp

a, b, y = sp.symbols("a b y", real=True, positive=True)
yy = -a + sp.sqrt(a**2 + b)
da = sp.simplify(sp.diff(yy, a))
db = sp.simplify(sp.diff(yy, b))
assert sp.simplify(da - (-yy / sp.sqrt(a**2 + b))) == 0
assert sp.simplify(db - 1 / (2 * sp.sqrt(a**2 + b))) == 0
# b>0 时两个条件数因子的绝对值小于 1
coef_a = sp.simplify(sp.Abs(a / sp.sqrt(a**2 + b)))
coef_b = sp.simplify((a + sp.sqrt(a**2 + b)) / (2 * sp.sqrt(a**2 + b)))
f = sp.lambdify((a, b), coef_a, "numpy")
g = sp.lambdify((a, b), coef_b, "numpy")
assert f(2.0, 1.0) < 1
assert g(2.0, 1.0) < 1

# 例 3 乘法次数
assert 10 * 20 * 50 + 10 * 50 * 1 + 10 * 1 * 100 == 11500
assert 50 * 1 * 100 + 20 * 50 * 100 + 10 * 20 * 100 == 125000
assert 20 * 50 * 1 + 10 * 20 * 1 + 10 * 1 * 100 == 2200

# 例 4
aa, bb = 0.3237, 0.3134
assert abs(aa * aa - 0.10478169) < 1e-12
assert abs(bb * bb - 0.09821956) < 1e-12
assert abs(0.1048 - 0.09822 - 0.00658) < 1e-12
assert abs((aa + bb) * (aa - bb) - 0.00656213) < 1e-8
assert abs(0.6371 * 0.01030 - 0.00656213) < 1e-8
assert abs(aa / bb - 1.032865348) < 1e-8
sq = (aa / bb) ** 2
assert 1 / 3 < sq < 3
true = aa * aa - bb * bb
assert abs(true - 0.656213e-2) < 1e-8
# 书上印 0.66e-2；四位有效数字的差是 0.658e-2。二者都比算法 II 离准确值更远
assert abs(0.66e-2 - true) > abs(0.6562e-2 - true)
assert abs(0.658e-2 - true) > abs(0.6562e-2 - true)

# x^n 的条件数是 n
x, n = sp.symbols("x n", positive=True)
fn = x**n
cond = sp.simplify(sp.Abs(x * sp.diff(fn, x) / fn))
assert cond == n

assert abs(3 / 5 - 0.6) < 1e-12
assert abs((3 + 5) / (2 * 5) - 0.8) < 1e-12
assert abs(3 / 0.1 - 30) < 1e-12
assert abs((2.01**2 - 4) - 0.0401) < 1e-12
assert abs(2 * 2.01 * 0.01 - 0.0402) < 1e-12
print("ALL OK")
