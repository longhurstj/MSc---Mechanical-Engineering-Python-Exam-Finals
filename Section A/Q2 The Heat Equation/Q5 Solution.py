# -*- coding: utf-8 -*-
"""
Created on Fri May 21 17:56:26 2021

@author: JoshuaL
"""

from sympy import symbols, Eq, solve, plot
from sympy import inverse_laplace_transform, lambdify
from sympy import exp, cos
from sympy import Function, Eq, Derivative

A, λ, x, t = symbols('A, λ, x, t')
 
u = Function('u')
u_t = Derivative(u, t)
u_xx = Derivative(u, x, 2)

eqn1 = Eq(u(x,t), A*exp(-λ**2*t)*cos(λ*x))

eqn1.rhs.diff(t)-eqn1.rhs.diff(x,2)     # differential equation solution

eqn1.rhs.diff(x)                        # 1st BC is satisfied as sin(0) equals 0


from sympy.abc import x, t, A, B 
from sympy import symbols, Eq, solve, plot
from sympy import exp, sin, cos, pi
from sympy import Sum, Function, Integral, Rational
from sympy import lambdify, pprint

λ, ξ = symbols('lamda xi')
n, m, N = symbols('n, m, N', integer=True)

"Assume the solution"
u = exp(-λ**2*t)*(A*cos((2*n-1)*pi/2*x)+B*sin(λ*x))
u = u.subs(B,0)
u = u.subs(λ,(pi*n))
An = 2*Integral(u.subs(t,0)/A,(x,0,Rational(1,2))).doit()
u = Sum(u,(n,1,4))
u_analytic = lambdify((x,t,N),u.subs(A,A(n).doit())

"Plot Graph"
import numpy as np
import matplotlib.pyplot as plt

xv,tv = np.meshgrid(np.linspace(0,1,31),
                    np.linspace(0,1,31))
plt.contourf(xv,tv,u_analytic(xv,tv,64))
plt.colorbar()
