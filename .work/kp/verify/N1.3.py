"""N1.3 数值计算原则。运行：python3 .work/kp/verify/N1.3.py"""
import math

assert abs((math.log(6) - math.log(5)) - 0.1823) < 5e-5
y100 = 0.5 * (1 / 606 + 1 / 505)
assert abs(y100 - 0.1815e-2) < 2e-7
assert abs(1 / (6 * 101) - 1 / 606) < 1e-15
assert abs(1 / (5 * 101) - 1 / 505) < 1e-15

x, y = 1.232, 1.231
# 四位十进制：平方与乘积先舍入到四位有效数字
x2, y2, xy = 1.518, 1.515, 1.517
assert abs(x * x - 1.517824) < 1e-9
assert abs(y * y - 1.515361) < 1e-9
assert abs(x * y - 1.516592) < 1e-9
z = 0.001 * (x2 + xy + y2)
assert abs(z - 0.00455) < 1e-12
# 直接相减的四位结果
x3 = 1.870
y3 = 1.865
assert abs(x * 1.518 - 1.870176) < 1e-9
assert abs(y * 1.515 - 1.864965) < 1e-9
assert abs((x3 - y3) - 0.005) < 1e-12
# 误差限
eps = 0.1e-2 * (0.5e-3 + 0.5e-3 + 0.5e-3)
assert abs(eps - 0.15e-5) < 1e-15
assert eps < 0.5e-5

# 秦九韶运算次数
n = 10
assert n * (n + 1) // 2 == 55
assert n == 10

print("ALL OK")
