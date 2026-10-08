"""N2.1 Gauss 主元消去法。运行：python3 .work/kp/verify/N2.1.py"""
import sympy as sp

n, k = sp.symbols("n k", integer=True, positive=True)
# 第 k 步乘除 (n-k)(n-k+2)，k=1..n-1
mult_k = (n - k) * (n - k + 2)
mult = sp.summation(mult_k, (k, 1, n - 1))
assert sp.factor(sp.simplify(mult - (n**3 / 3 + n**2 / 2 - sp.Rational(5, 6) * n))) == 0
add_k = (n - k) * (n - k + 1)
add = sp.summation(add_k, (k, 1, n - 1))
assert sp.simplify(add - (n**3 / 3 - n / 3)) == 0
# 回代
i = sp.symbols("i", integer=True, positive=True)
back_m = sp.summation(n - i + 1, (i, 1, n))
assert sp.simplify(back_m - (n**2 / 2 + n / 2)) == 0
back_a = sp.summation(n - i, (i, 1, n - 1))
assert sp.simplify(back_a - (n**2 / 2 - n / 2)) == 0
total_m = mult + back_m
assert sp.simplify(total_m - (n**3 / 3 + n**2 - n / 3)) == 0

# 例 1
l = sp.Rational(531, 100) / sp.Rational(3, 100)
assert l == 177
a22 = sp.Rational(-610, 100) - 177 * sp.Rational(589, 10)
b2 = sp.Rational(470, 10) - 177 * sp.Rational(592, 10)
assert a22 == sp.Rational(-104314, 10)
assert b2 == a22
# 三位舍入后
assert abs(float(a22) - (-1.04e4)) / 1.04e4 < 0.005
x2 = sp.Rational(-105, 10) / sp.Rational(-104, 10) * 1000
# book uses -1.05e4 / -1.04e4
x2b = (-1.05e4) / (-1.04e4)
assert abs(x2b - 1.01) < 0.005
# 三位：58.9*1.01 → 59.5，59.2-59.5=-0.3，/0.0300=-10
assert abs(58.9 * 1.01 - 59.489) < 1e-9
x1_3digit = (59.2 - 59.5) / 0.0300
assert abs(x1_3digit - (-10.0)) < 1e-9

l2 = 0.0300 / 5.31
assert abs(l2 - 0.00565) < 5e-6
a22p = 58.9 - l2 * (-6.10)
b2p = 59.2 - l2 * 47.0
assert abs(a22p - 58.9) < 0.05
assert abs(b2p - 58.9) < 0.05
assert abs((47.0 + 6.10) / 5.31 - 10.0) < 1e-9

print("ALL OK")
