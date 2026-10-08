"""N1.1 绝对误差与有效数字。运行：python3 .work/kp/verify/N1.1.py"""
import math

# 例 2：书上把差写成 0.0000073 后再除
e_book = 0.0000073
xs = 3.1416
er = e_book / xs
assert abs(er - 0.2324e-5) < 5e-10
# 用更多位的 π 时，差约为 7.346e-6，相对误差仍是 10^{-6} 量级
pi = 3.141592653589793
assert abs((xs - pi) - 7.346e-6) < 1e-9

# 例 3：误差限 0.5e-2
eps = 0.5e-2
# a=1.38=0.138e1，三位有效数字的门槛正好等于 eps
assert abs(0.5 * 10 ** (1 - 3) - eps) < 1e-15
assert 0.5 * 10 ** (1 - 4) < eps
# b=-0.0312=0.312e-1，一位有效数字
assert abs(0.5 * 10 ** (-1 - 1) - eps) < 1e-15
assert 0.5 * 10 ** (-1 - 2) < eps
# c=0.86e-4，连一位都达不到
assert 0.5 * 10 ** (-4 - 1) < eps
assert abs(eps / 1.38 - 3.6e-3) < 2e-4
assert abs(eps / 0.0312 - 0.16) < 0.01
assert abs(eps / (0.86e-4) - 58) < 1

# 例 4
# (1/4)*10^{-(n-1)} < 0.01 ⇒ n>2.397...
nmin = 1 - math.log10(0.04)
assert 2.39 < nmin < 2.40
s5 = math.sqrt(5)
assert abs(2.24 - s5) / s5 < 0.01
assert abs(2.24 - s5) / 2.24 < 0.01

# 式 (1.1-2) 与 (1.1-2') 的差约为 (e/x*)^2
e = 1e-4
xstar = 2.0
x = xstar - e
diff = e / x - e / xstar
assert abs(diff - (e / xstar) ** 2 / (1 - e / xstar)) < 1e-18

assert abs((3.1416 - 3.14) - 0.0016) < 1e-15
assert abs((3.1456 - 3.15) - (-0.0044)) < 1e-15 or abs(abs(3.1456 - 3.15) - 0.0044) < 1e-15
assert abs(0.5 * 10 ** (1 - 3) - 0.005) < 1e-15
print("ALL OK")
