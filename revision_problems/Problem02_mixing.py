#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 23:57:25 2021

@author: jwhitty
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

fn = lambda t,m, q, c, V_1, V_2: np.array([q*c-q*m[0]/V_1, 
                                           q*m[0]/V_1-q*m[1]/V_2])

t=np.linspace(0,3*60,31)
mrk = solve_ivp(fn, (t[0],t[-1]),[5,10], method='RK45', 
                args=(15,0.1,500, 250), t_eval=t)

plt.plot(mrk.t,mrk.y[0],'x')
plt.plot(mrk.t,mrk.y[1],'+')

"Using Laplace Transforms ..."
from sympy import symbols, Eq, solve
from sympy import inverse_laplace_transform
from sympy import lambdify

m_1, M_1,m_2, M_2, s= symbols('m_1 M_1 m_2 M_2 s')
c, q, V_1, V_2 = symbols('c, q, V_1, V_2',positive=True)
m0_1, m0_2 = symbols('m0_1 m0_2')

eq1 = Eq(s*M_1-m0_1, q*c/s-q*M_1/V_1)
eq2 = Eq(s*M_2-m0_2, q*M_1/V_1-q*M_2/V_2)

eqns3=solve((eq1,eq2),(M_1, M_2))

t = symbols('t',positive=True)
m_1=inverse_laplace_transform(eqns3[M_1],s,t)
m_2=inverse_laplace_transform(eqns3[M_2],s,t)

#m_one = lambdify((t, q, c, V_1, V_2), m_1)
#m_one(0,15,0.1,500,250)
