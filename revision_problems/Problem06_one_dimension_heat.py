#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 00:07:28 2021

@author: jwhitty
"""

from sympy.abc import x, t, A, B 
from sympy import symbols, Eq, solve, plot
from sympy import exp, sin, cos, pi
from sympy import Sum, Function, Integral, Rational
from sympy import lambdify, pprint

λ, ξ = symbols('lamda xi')
n, m, N = symbols('n, m, N', integer=True)

"Part (a)"
"Assume the solution..."
u = exp(-λ**2*t)*(A*cos(λ*x)+B*sin(λ*x))
"With the homogenous boundary condtions..."
u.diff(x).subs(x,0)
u=u.subs(B,0)
u.subs(x,1)

"Part (b)"
c = symbols('c')
T = exp(-λ**2*t)
X = (A*cos(λ*x)+B*sin(λ*x)).subs(λ, λ/c)
eqn = (X*T).diff(t)-c**2*(X*T).diff(x,2)
from sympy import simplify
simplify(eqn)

"Part (c)"
u.diff(x).subs(x,0)
u.subs(x,1)
u=u.subs(λ,(2*n+1)*pi/2)
An = 2*Integral(u.subs(t,0)/A,(x,0,Rational(1,2))).doit()
u = Sum(u.subs(A,An),(n,0,N))
u_analytic = lambdify((x,t,N),u.subs(N,N).doit())

"Part (d)..."
import numpy as np
import matplotlib.pyplot as plt

xv,tv = np.meshgrid(np.linspace(0,1,31),
                    np.linspace(0,1,31))
plt.contourf(xv,tv,u_analytic(xv,tv,64))
plt.colorbar()

"Part (e) ..."
plt.plot(np.linspace(0,1,31),
         u_analytic(np.linspace(0,1,31),0,32),'k')

"Part (f)"

