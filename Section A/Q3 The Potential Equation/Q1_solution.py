# -*- coding: utf-8 -*-
"""
Created on Sat May 22 17:17:52 2021

@author: JoshuaL
"""

from sympy import symbols, Eq, solve, plot
from sympy import inverse_laplace_transform, lambdify
from sympy import exp, cosh, tanh, sinh, sin
from sympy import Function, Derivative
import numpy as np

π = np.pi

B, y, x, t, n, H, L = symbols('B, y, x, t, n, H, L')
 
u = Function('u')
u_t = Derivative(u, t)
u_xx = Derivative(u, x, 2)
u_yy = Derivative(u, y, 2)

u = Eq(u(x,t), B*(cosh(n*π*x/H)*tanh(n*π*L/H)-sinh(n*π*x/H))*sin(n*π*y/H))

eq1 = u.rhs.diff(y,2)+u.rhs.diff(x,2)       # differential equation solution

eq2 = u.rhs.diff(t,0)                       # 1st BC is satisfied when sin(0)

eq3 = u.rhs.diff(L,y)                       # 2nd BC is satisfied when H is 0

eq4 = u.rhs.diff(x,H)                       # 3rd BC is satisfied when B is 0
