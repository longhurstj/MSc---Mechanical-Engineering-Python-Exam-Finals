#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 21:00:17 2021

@author: jwhitty
"""

from sympy.abc import x, t, A, B 
from sympy import symbols, Eq, solve, plot
from sympy import exp, sin, cos, pi
from sympy import Sum, Function, Integral
from sympy import lambdify, pprint

λ, ξ = symbols('lamda xi')
n, m, N = symbols('n, m, N', integer=True)

"Part (a) ..."
"Assume the solution..."
u = exp(-λ**2*t)*(A*cos(λ*x)+B*sin(λ*x))
c, ρ, k = symbols('c rho k', positive=True)
u=u.subs(t,k*t/c/ρ)
"Boundary condtions..."
pprint(Eq(u.subs(x,0),20))
pprint(Eq(u.subs(x,4),20))
"Initial conditons..."
pprint(Eq(u.subs(t,0),100*4*x*(x-4)/16))

"Part (b)"
"Enforce homogeneous BCs..."
Eq(u.subs(x,0),20-20) 
u = u.subs(A,0)
Eq(u.subs(x,4),20-20) 
u=u.subs(λ,solve(4*λ-n*pi, λ)[0])/B
"The initial condtion..."#
Bn = (2/4)*Integral(u.subs(t,0)*(100*4*x*(4-x)/16),(x,0,4)).doit()
u = 20+Sum(Bn*u,(n,1,N))

"Part (c) ..."
u_analytic = lambdify((x,t,N),u.subs({c: 900, k: 200, ρ: 2700}))
import numpy as np
import matplotlib.pyplot as plt
t_s = np.linspace(0,25*3600,101); plt.plot(t_s/3600,u_analytic(2,t_s,32))
plt.figure(3)
plt.grid();plt.xlabel('time (hours)'); plt.ylabel('Temperature (Celsius)')

"Part (d)..."
pprint(Sum(
        u.subs({c: 900, k: 200, ρ: 2700}).args[1].args[0].args[0][0],(n,1,N)))
x_s = np.linspace(0,4,101)
plt.figure(4); 
plt.plot(x_s,100*x_s*4/16*(4-x_s)+20)
x_s = np.linspace(0,4,21)
plt.plot(x_s,u_analytic(x_s,0,8),'x')

"Part (e)..."
xv,tv = np.meshgrid(np.linspace(0,4,31),
                    np.linspace(0,24*3600,31))
plt.figure(5)
ax = plt.axes(projection='3d')
ax.plot_surface(xv, tv/3600, u_analytic(xv,tv,4))

"Part (f)"
plt.figure(6)
plt.contourf(xv,tv/3600,u_analytic(xv,tv,25))
plt.colorbar()
