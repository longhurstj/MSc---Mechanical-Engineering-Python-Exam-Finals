# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:39:08 2021

@author: JoshuaL
"""

from sympy import symbols, solve, Eq, sqrt, expand, simplify # Algebraic
from sympy import Function, dsolve, Derivative, Integral # Calculus
from sympy import lambdify # Utilities

μ, μ_PL, μ_o, μ_oo, λ, n, τ_Y, R, v_max = symbols('mu mu_PL mu_o mu_oo \
                                           lambda n tau_Y R v_max', positive=True)
Y, r, P_z = symbols('Y r P_z')
v = Function('v')
v_r = Derivative('v(r)',r)
eq1 = Eq(Derivative(μ*r*v_r, r)*1/r, P_z)

''' Newtonian '''
eq2 = dsolve(eq1)
eq2.subs({r:0, v(r):v_max})
# therefore c2 is zero
C1, C2 = symbols('C1 C2')
eq2 = eq2.subs(C2,0)
eq3 = eq2.subs({r:R, v(r):0})
eq4 = eq2.subs(C1, solve(eq3, C1)[0])
v_newton = lambdify((P_z, μ, R, r), eq4.rhs)
eq5 = eq4.subs({v(r):v_max, r:0})
solve(eq5, P_z)[0].subs({μ:3e-3, R:1.25e-3, v_max:0.5})
