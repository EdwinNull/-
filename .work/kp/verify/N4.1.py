"""N4.1 不动点迭代。运行：python3 .work/kp/verify/N4.1.py"""
import math

# 例 1
x = 1.5
x = (1 + x) ** (1 / 3)
assert abs(x - 1.35721) < 5e-6
x = (1 + x) ** (1 / 3)
assert abs(x - 1.33086) < 5e-5
x = 1.5
for _ in range(9):
    x = (1 + x) ** (1 / 3)
assert abs(x - 1.32472) < 5e-6
# 发散格式
x = 1.5
assert abs((x**3 - 1) - 2.375) < 1e-12
x = 2.375
assert abs(x**3 - 1 - 12.396484375) < 1e-9

# 习题 4 第 1 题(2) 的印刷不一致：题面等价式 x=∛(1+x²)，但印出的迭代映射是 ∛(1+1/x²)。
# 按照印刷映射，从 1.5 出发收敛到另一个固定点 x^5-x²-1=0。
root_original = 1.465571231876768
assert abs(root_original**3 - root_original**2 - 1) < 1e-12
phi_printed = lambda z: (1 + 1 / z**2) ** (1 / 3)
assert abs(phi_printed(root_original) - root_original) > 0.3
x = 1.5
for _ in range(20):
    x = phi_printed(x)
assert abs(x - 1.1938591113212231) < 1e-10
assert abs(x**5 - x**2 - 1) < 1e-10
assert abs(-2 / (3 * x**5)) < 1  # attracts to the wrong fixed point

# 例 2
assert abs(math.exp(-0.5) - 0.60653) < 5e-6
x = 0.5
for _ in range(15):
    x = math.exp(-x)
assert abs(x - 0.56714) < 5e-5

# 例 4 简单迭代与一步 Steffensen
x = 3.0
z1 = 2 + math.log(x)
z2 = 2 + math.log(z1)
assert abs(z1 - 3.098612289) < 1e-8
assert abs(z2 - 3.130954363) < 1e-8
xnew = z2 - (z2 - z1) ** 2 / (z2 - 2 * z1 + x)
assert abs(xnew - 3.146738373) < 1e-8

# 例 5
x1, x2 = 1.0, -1.5
assert abs(math.log(1 - x2) - 0.916290731) < 1e-8
assert abs(-math.sqrt(4 - x1**2) + 1.732050808) < 1e-8
x1n = math.log(1 - x2)
x2n = -math.sqrt(4 - x1n**2)
assert abs(x2n + 1.777754565) < 1e-8
q = 5 / math.sqrt(39)
assert q < 1
assert abs(q - 0.80064) < 1e-4

# 先验：L=0.8 时 L^k/(1-L) * d < eps 的方向
L = 0.8
assert abs(L / (1 - L) - 4) < 1e-12

print("ALL OK")
