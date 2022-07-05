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

