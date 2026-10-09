"""N6.1 Newton-Cotes。运行：python3 .work/kp/verify/N6.1.py"""
import math

# 章首例 1
assert abs(0.5 * (1 + 1) - 1) < 1e-15
assert abs(0.5 * (0 + 1) - 0.5) < 1e-15
assert abs(1 / 3 - 0.5) > 0.1

# 章首例 2
# A0+A1+A2=2, -A0+A2=0, A0+A2=2/3
A0, A1, A2 = 1 / 3, 4 / 3, 1 / 3
assert abs(A0 + A1 + A2 - 2) < 1e-15
assert abs(-A0 + A2) < 1e-15
assert abs(A0 + A2 - 2 / 3) < 1e-15
assert abs((1 / 3) * ((-1) ** 3 + 1) ) < 1e-15  # x^3 exact
assert abs(2 / 5 - 2 / 3) > 0.1  # x^4 not exact

# 节例 1
assert abs(0.5 * (1 + 0.5) - 0.75) < 1e-15
assert abs((1 + 4 / 1.25 + 0.5) / 6 - 0.7833333333333333) < 1e-12
def f(x):
    return 1 / (1 + x * x)
xs = [0, 0.25, 0.5, 0.75, 1]
ws = [7, 32, 12, 32, 7]
cotes = sum(w * f(x) for w, x in zip(ws, xs)) / 90
assert abs(cotes - 0.7855294117647059) < 1e-15
assert abs(math.pi / 4 - 0.7853981633974483) < 1e-15

# Cotes 系数和与对称、n=8 负系数
assert abs(7 / 90 + 32 / 90 + 12 / 90 + 32 / 90 + 7 / 90 - 1) < 1e-15
assert abs(1 / 8 + 3 / 8 + 3 / 8 + 1 / 8 - 1) < 1e-15
n8 = [989, 5888, -928, 10496, -4540, 10496, -928, 5888, 989]
assert sum(n8) == 28350
assert n8[2] < 0 and n8[4] < 0

# 节例 2
trap = 0.5 * (math.e + math.exp(0.5))
simp = (math.e + 4 * math.exp(1 / 1.5) + math.exp(0.5)) / 6
assert abs(trap - 2.183501549579587) < 1e-12
assert abs(simp - 2.0263232105629796) < 1e-12
f2 = 3 * math.e
assert abs(f2 - 8.154845485377136) < 1e-12
assert abs(f2 / 12 - 0.6795704571147613) < 1e-12
f4 = 73 * math.e
assert abs(f4 - 198.4345734775103) < 1e-10
assert abs(f4 / 180 / 16 - 0.06890089356857997) < 1e-12

exact = math.pi / 4
assert abs(exact - 0.75 - 0.03539816339744828) < 1e-12
assert abs(exact - 0.7833333333333333 - 0.00206483006411498) < 1e-12
assert abs(0.7855294117647059 - exact - 0.0001312483672576) < 1e-12
assert abs(73 * math.e - 198.4345734775103) < 1e-10

assert abs(1/1.0625 - 0.9411764705882353) < 1e-12
assert abs(70.69764705882352 / 90 - 0.7855294117647059) < 1e-12
assert abs(5/18 - 1/3) > 0.01
assert abs(0.5 + 0.5 - 1) < 1e-15

assert abs(2/3 - 0) > 0.5
assert abs(2.183501549579587 - 2.0263232105629796 - 0.1571783390166074) < 1e-12

assert 19+75+50+50+75+19 == 288
assert 41+216+27+272+27+216+41 == 840
assert abs(1 + 0.2 - 1.2) < 1e-15
assert abs(math.atan(2) - 1.1071487177940904) < 1e-12
assert abs(1.2 - math.atan(2) - 0.09285128220590958) < 1e-12

print("ALL OK")
