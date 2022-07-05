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

''' Casson '''
μ_casson = ((sqrt(μ_PL)+sqrt(τ_Y/Y))**2)
eq6 = Eq(eq1.lhs*r)                             # help the solver along
eq7 = Eq(eq6.args[0].args[0],Integral(eq6.rhs,r).doit()) # reduce the order
eq8 = eq7.subs(μ,μ_casson).subs(Y,v_r)
eq9 = dsolve(eq8,ics={v(R):0})
eq9a = eq9[0]; eq9b = eq9[1];
v_casson = lambdify((P_z, μ_PL, τ_Y, R, r),-eq9a.rhs, modules='numpy')

''' Cross '''
μ_cross = μ_oo + (μ_o-μ_oo)/(1+(λ*Y)**(1-n))
eq10 = eq7.subs(μ,μ_cross).subs(Y,v_r)
eq11 = dsolve(eq10,ics={v(R):0})
eq11a = eq11[0]; eq11b = eq11[1];
v_cross = lambdify((P_z, μ_PL, τ_Y, R, r),-eq11a.rhs, modules='numpy')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import norm
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

P_z_newton = fsolve(lambda x: v_newton(x,3e-3,1.25e-3,0)-0.5,0)[0]
P_z_casson = fsolve(lambda x: v_casson(x,3e-3,6e-3,1.25e-3,0)-0.5,-P_z_newton)[0]
#P_z_cross = fsolve(lambda x: v_cross(x,3e-3,6e-3,1.25e-3,0)-0.5,-P_z_newton)[0]

r = np.linspace(0, 1.25e-3, 101)
plt.plot(r, v_newton(P_z_newton, 3e-3, 1.25e-3, r), '--', label = 'Newtonian')
r = np.linspace(0, 1.25e-3, 21)
plt.plot(r, v_casson(P_z_casson, 3e-3, 6e-3, 1.25e-3, r), 'go', label = 'Casson')
#r = np.linspace(0, 1.25e-3, 21)
#plt.plot(r, v_cross(P_z_cross, 3e-3, 6e-3, 1.25e-3, r), 'go', label = 'Cross')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Radial Cord (m)')
plt.legend()