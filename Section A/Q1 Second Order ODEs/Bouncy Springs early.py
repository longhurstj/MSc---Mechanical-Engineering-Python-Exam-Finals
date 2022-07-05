# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:23:04 2021

@author: Josh
"""

from sympy.abc import t
from sympy import Eq
from sympy import Function, Derivative, dsolve, diff

v_1=Function('v_1')
v_2=Function('v_2')
y_1=Function('y_1')
y_2=Function('y_2')

y_1_t = Derivative('y_1(t)',t)
y_2_t = Derivative('y_2(t)',t)
v_1_t = Derivative('v_1(t)',t)
v_2_t = Derivative('v_2(t)',t)

eq1 = y_1_t-v_1(t)
eq2 = v_1_t+2*y_1(t)+5*(y_1(t)-y_2(t))
eq3 = y_2_t-v_2(t)
eq4 = v_2_t-5*(y_1(t)-y_2(t))+4*y_2(t)

eqns5 = dsolve([eq1, eq2, eq3, eq4], ics={y_1(0): 0, v_1(0): 2,
                                          y_2(0): 0, v_2(0): 0})

eqns5[0]
eqns5[1]
eqns5[2]
eqns5[3]

import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify

''' Import Data '''
A = np.loadtxt('data_A.txt')
B = np.loadtxt('data_B.txt')
C = np.loadtxt('data_C.txt')
D = np.loadtxt('data_D.txt')
A2 = np.loadtxt('data_A2.csv')

t_s = np.linspace(0,10,51)
plt.plot(A[1], '--', label = 'data_A')
plt.plot(B, '--', label = 'data_B')

plt.plot(C, '--', label = 'data_C')
plt.plot(D, '-', label = 'data_D')

y1 = lambdify(t,eqns5[2].rhs)
y2 = lambdify(t,eqns5[0].rhs)

t_s=np.linspace(0,10,101);
plt.plot(t_s,y1(t_s),'b-');
plt.plot(t_s,y2(t_s),'k-');
t_s=np.linspace(0,10,101);
plt.plot(t_s,y1(t_s),'r.');
plt.plot(t_s,y2(t_s),'g.');

plt.plot(A2, '--', label = 'data_A');




from scipy.integrate import solve_ivp

A=np.array([[0, -1, 0, 0],
            [7, 0, -5, 0],
            [0, 0, 0, -1],
            [-5, 0, 9, 0]])

t_s=np.linspace(0,3,101)
fun = lambda t,y: A@y-[np.sin(t),0,0,0]
yrk = solve_ivp(fun, (t_s[0],t_s[-1]), [0,0,0,0],
                method='RK45')

plt.plot(yrk.t,yrk.y[0],'b-')
plt.plot(yrk.t,yrk.y[2],'k-')


