#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:33:10 2021

@author: jwhitty
"""

from sympy.abc import A, B, C, D
from sympy import symbols, Eq, solve
from sympy import sin, cos, sinh, cosh, pi
from sympy import Integral, Sum
from sympy import lambdify

x, y, λ =symbols('x y lamda', positive=True)
n, m, N = symbols('n m N',positive=True)

X = A*cos(λ*x)+B*sin(λ*x)
Y = C*cosh(λ*y)+D*sinh(λ*y)
u=X*Y
Eq(u.subs(x,0),0)
u=u.subs(A,0)
Eq(u.subs(y,0),0)
u=u.subs({C: 0, D: 1})
Eq(u.subs(x,2),0)
u=u.subs(λ, n*pi/2)
u=Sum(u,(n,1,N))
eqn=Eq(u.subs(y,1),150*x*(2-x))
fn = eqn.args[0].args[0].args[1]/eqn.args[0].args[0].args[2]*150*x*(2-x)
Bn = (2/2)*Integral(fn,(x,0,2)).doit()
u_analytic = lambdify((x,y,N),u.subs(B,Bn))

import numpy as np
import matplotlib.pyplot as plt
x_s = np.linspace(0,2,25)
y_s = np.linspace(0,1,25)

plt.figure(0)
plt.plot(x_s,u_analytic(x_s,1,8),'o')
plt.plot(x_s,150*x_s*(2-x_s),'k--')

xv,yv = np.meshgrid(x_s,y_s)
plt.figure(1)
plt.contourf(xv, yv, u_analytic(x_s, yv, 16))

plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot_surface(xv, yv, u_analytic(xv, yv, 16))

plt.figure(3)
ANSYSdata=np.loadtxt('ANSYSPlateMidTemps.txt',skiprows=5)
plt.plot(ANSYSdata[:,0], ANSYSdata[:,1],'k^')
plt.plot(y_s,u_analytic(1,y_s,8),'--')

from numpy.linalg import norm
err=ANSYSdata[:,1]-u_analytic(1,ANSYSdata[:,0],8)
err=100*err/u_analytic(1,ANSYSdata[:,0],8)
[norm(err[1:-1], ord=1),
 norm(err[1:-1], ord=2),
 norm(err[1:-1], ord=np.inf)]

