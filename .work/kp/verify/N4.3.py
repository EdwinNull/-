"""N4.3 无约束优化下降法。运行：python3 .work/kp/verify/N4.3.py"""
import numpy as np

# 例 1 最速下降法，F=1/2 x^T A x
A = np.array([[4.0, 2.0], [2.0, 10.0]])
x = np.array([1.0, -1.0])
expected = [
    (5.0, 0.11486486486486487),
    (1.0945945945945947, 0.18888888888888886),
    (0.23962746530314102, 0.11486486486486484),
    (0.052458985647444376, 0.188888888888889),
]
for f_exp, t_exp in expected:
    g = A @ x
    Ag = A @ g
    t = (g @ g) / (g @ Ag)
    f = 0.5 * x @ A @ x
    assert abs(f - f_exp) < 1e-10
    assert abs(t - t_exp) < 1e-12
    assert abs((A @ (x - t*g)) @ g) < 1e-10  # exact line-search orthogonality
    x = x - t * g
    assert 0.5*x@A@x < f
assert np.allclose(x, [0.0479254931, -0.0479254931], atol=1e-9)

# 自编例：diagonal quadratic exact line minimizer
Ad = np.diag([2.0, 4.0])
xd = np.array([1.0, 1.0])
gd = Ad @ xd
td = (gd @ gd) / (gd @ Ad @ gd)
assert abs(td - 5/18) < 1e-12
xfd = xd - td*gd
assert np.allclose(xfd, [4/9, -1/9])
assert 0.5*xfd@Ad@xfd < 0.5*xd@Ad@xd

# 变尺度例 2: DFP, exact quadratic line search
H = np.array([[3.0, -1.0], [-1.0, 1.0]])
b = np.array([-2.0, 0.0])
def F2(z): return 0.5*z@H@z+b@z
def grad2(z): return H@z+b
x = np.array([-2.0, 4.0])
B = np.eye(2)
assert np.allclose(grad2(x), [-12, 6])
assert abs(F2(x)-26) < 1e-12
for _ in range(2):
    g = grad2(x)
    p = -B@g
    t = -(g@p)/(p@H@p)
    xn = x+t*p
    gn = grad2(xn)
    s = xn-x
    y = gn-g
    sy = s@y
    assert sy > 0
    B = B + np.outer(s,s)/sy - np.outer(B@y,B@y)/(y@B@y)
    x = xn
assert np.allclose(x,[1,1],atol=1e-10)
assert np.allclose(B, np.array([[0.5,0.5],[0.5,1.5]]),atol=1e-10)

# BFGS example 3, exact quadratic line search
H3 = np.diag([2.0, 8.0])
x = np.array([1.0,1.0])
B = np.eye(2)
expected_t=[0.13076923076923078,0.47794117647058815]
for t_exp in expected_t:
    g=H3@x
    p=-B@g
    t=-(g@p)/(p@H3@p)
    assert abs(t-t_exp)<1e-12
    xn=x+t*p
    gn=H3@xn
    s=xn-x
    y=gn-g
    sy=s@y
    By=B@y
    B=B+(1+(y@By)/sy)*np.outer(s,s)/sy-(np.outer(By,s)+np.outer(s,By))/sy
    x=xn
assert np.allclose(x,[0,0],atol=1e-12)
assert np.allclose(B,np.diag([0.5,0.125]),atol=1e-10)

# Exercise 9 Newton downhill: one accepted step to local minimum; second is stationary
F9=lambda z: 4*z[0]**2+z[1]**2-z[0]**2*z[1]
g9=lambda z: np.array([8*z[0]-2*z[0]*z[1],2*z[1]-z[0]**2])
H9=lambda z: np.array([[8-2*z[1],-2*z[0]],[-2*z[0],2.0]])
x=np.array([0.0,2.0])
assert abs(F9(x)-4)<1e-12
assert np.allclose(g9(x),[0,4])
p=-np.linalg.solve(H9(x),g9(x))
assert np.allclose(p,[0,-2])
x=x+p
assert np.allclose(x,[0,0])
assert abs(F9(x))<1e-12
assert np.allclose(g9(x),[0,0])
# Unboundedness along x2=8
assert abs(F9(np.array([10.0,8.0]))-(-336.0))<1e-12

# Exercise 10 first DFP and BFGS updates coincide in this aligned case
x0=np.array([0.0,2.0]); g0=g9(x0); p0=-g0; t0=0.5
x1=x0+t0*p0; g1=g9(x1); s=x1-x0; y=g1-g0; sy=s@y
assert np.allclose(x1,[0,0]) and np.allclose(g1,[0,0])
B0=np.eye(2)
Bdfp=B0+np.outer(s,s)/sy-np.outer(B0@y,B0@y)/(y@B0@y)
By=B0@y
Bbfgs=B0+(1+(y@By)/sy)*np.outer(s,s)/sy-(np.outer(By,s)+np.outer(s,By))/sy
assert np.allclose(Bdfp,np.diag([1,0.5]))
assert np.allclose(Bbfgs,Bdfp)

print("ALL OK")
