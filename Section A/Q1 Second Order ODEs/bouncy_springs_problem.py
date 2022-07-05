#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 13:05:02 2021

@author: JoshuaL
"""

from sympy import symbols, Eq, solve, plot
from sympy import inverse_laplace_transform, lambdify, laplace_transform

Y_1, Y_2, s, F_1, F_2 = symbols('Y_1 Y_2 s F_1 F_2')

eq1 = Eq(s**2*Y_1+2*Y_1+5*(Y_1-Y_2), F_1)
eq2 = Eq(2*s**2*Y_2-5*(Y_1-Y_2)+4*Y_2, F_2)

exprs_1 = solve([eq1.subs(F_1, 2),eq2.subs(F_2, 0)],[Y_1, Y_2])

t = symbols('t',positive=True)
expr_y1=inverse_laplace_transform(exprs_1[Y_1],s ,t)
expr_y2=inverse_laplace_transform(exprs_1[Y_2],s ,t)
y_1_analytic = lambdify(t, expr_y1)
y_2_analytic = lambdify(t, expr_y2)


exprs_2 = solve([eq1.subs(F_1, 1),eq2.subs(F_2, 2)],[Y_1, Y_2])

expr_y3 = inverse_laplace_transform(exprs_2[Y_1],s ,t)
expr_y4 = inverse_laplace_transform(exprs_2[Y_2],s ,t)
y_3_analytic = lambdify(t, expr_y3)
y_4_analytic = lambdify(t, expr_y4)

'''exprs = solve([eq1.subs(F_1,1),eq2.subs(F_2,1)],[Y_1, Y_2])
inverse_laplace_transform(exprs[Y_1],s ,t)
inverse_laplace_transform(exprs[Y_2],s ,t)'''

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

def fn_x(t,y,f):
    y_1=y[0]; v_1=y[1]; y_2=y[2]; v_2=y[3]; f_1=f[0]; f_2=f[1];
    return np.array([v_1,
                     -2*y_1-(y_1-y_2)*5+f_1,
                     v_2,
                     +5*(y_1-y_2)-4*y_2+f_2
                     ])

f_x = lambda t, y, f: np.array([y[1],
                                -2*y[0]-5*(y[0]-y[2])+f[0],
                                y[3],
                                +5*(y[0]-y[2])-4*y[2]+f[1]
                                 ])
t_s = np.linspace(0,10,101)
plt.plot(t_s,y_1_analytic(t_s),'b-',label='analytic (y_1)')
plt.plot(t_s,y_2_analytic(t_s),'k-',label='analytic (y_2)')
t_s = np.linspace(0,10,51)
plt.plot(t_s,y_1_analytic(t_s),'ro',label='XCOS (y_1)')
plt.plot(t_s,y_2_analytic(t_s),'go',label='XCOS (y_2)')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend()
plt.grid()

t_s = np.linspace(0,10,51)
plt.plot(t_s,y_3_analytic(t_s),'--',label='analytic $$(y_3)$$')
plt.plot(t_s,y_4_analytic(t_s),'--',label='analytic $$(y_4)$$')

yrk = solve_ivp(f_x, (t_s[0],t_s[-1]) , [0,1,0,0], method='RK45', 
                args=np.array([[0,0]]),t_eval=t_s)
plt.plot(yrk.t,yrk.y[0,:],'o');
plt.plot(yrk.t,yrk.y[2,:],'s');

fnrk = solve_ivp(fn_x, (t_s[0],t_s[-1]) , [0,1,0,0], method='RK45', 
                 args=np.array([[0,0]]), t_eval=t_s)
plt.plot(fnrk.t,fnrk .y[0,:],'o--');
plt.plot(fnrk.t,fnrk .y[2,:],'s--');

"Part (c)..."
H = lambda t: 1*(t>0)
f_x = lambda t, y: np.array([y[1],
                                -6*y[0]-2*(y[0]-y[2])+3*(1-H(t-5)),
                                y[3],
                                +2*(y[0]-y[2])-3*y[2]+0
                                 ])
t_s=np.linspace(0,10,51)
yrk = solve_ivp(f_x, (t_s[0],t_s[-1]) , [0,0,0,0], method='RK45', t_eval=t_s)


exprs = solve([eq1.subs(F_1,3/s),eq2.subs(F_2,0)],[Y_1, Y_2])
t = symbols('t',positive=True)
expr_y1=inverse_laplace_transform(exprs[Y_1],s ,t)
expr_y2=inverse_laplace_transform(exprs[Y_2],s ,t)
y_1_analytic = lambdify(t, expr_y1)
y_2_analytic = lambdify(t, expr_y2)

plt.figure()
plt.plot(yrk.t, yrk.y[0,:],'ko')
plt.plot(t_s,y_1_analytic(t_s)-H(t_s-5)*y_1_analytic(t_s-5),'k')


"Part (e) ..."
H = lambda t: 1*(t>0)
f_x = lambda t, y, c_1, c_2: np.array([y[1],
                                -6*y[0]-c_1*y[1]-2*(y[0]-y[2])-c_2*(y[1]-y[3])+3*(1-H(t-5)),
                                y[3],
                                +2*(y[0]-y[2])+c_2*(y[1]-y[3])-3*y[2]+0
                                 ])
t=np.linspace(0,50,1001)
yrk = solve_ivp(f_x, (t[0],t[-1]) , [0,0,0,0], method='RK45', 
                args=(0.5,0.5), t_eval=t)

plt.figure()
plt.plot(yrk.t,yrk.y[0,:],'--')




