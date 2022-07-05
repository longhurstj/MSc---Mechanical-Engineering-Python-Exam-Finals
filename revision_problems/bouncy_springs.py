#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:19:02 2021

@author: JoshuaL
"""

from sympy import symbols, Eq, solve, plot
from sympy import inverse_laplace_transform, lambdify

Y_1, Y_2, s, F_1, F_2 = symbols('Y_1 Y_2 s F_1 F_2')

eq1 = Eq(s**2*Y_1+6*Y_1+2*(Y_1-Y_2), F_1)
eq2 = Eq(s**2*Y_2-2*(Y_1-Y_2)+3*Y_2, F_2)

exprs = solve([eq1.subs(F_1,1),eq2.subs(F_2,0)],[Y_1, Y_2])

t = symbols('t',positive=True)
expr_y1=inverse_laplace_transform(exprs[Y_1],s ,t)
expr_y2=inverse_laplace_transform(exprs[Y_2],s ,t)
y_1_analytic = lambdify(t, expr_y1)
y_2_analytic = lambdify(t, expr_y2)

exprs = solve([eq1.subs(F_1,0),eq2.subs(F_2,1)],[Y_1, Y_2])
inverse_laplace_transform(exprs[Y_1],s ,t)
inverse_laplace_transform(exprs[Y_2],s ,t)

exprs = solve([eq1.subs(F_1,1),eq2.subs(F_2,1)],[Y_1, Y_2])
inverse_laplace_transform(exprs[Y_1],s ,t)
inverse_laplace_transform(exprs[Y_2],s ,t)

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

def fn_x(t,y,f):
    y_1=y[0]; v_1=y[1]; y_2=y[2]; v_2=y[3]; f_1=f[2]; f_2=f[0];
    return np.array([v_1,
                     -6*y_1-(y_1-y_2)*2+f_1,
                     v_2,
                     +2*(y_1-y_2)-3*y_2+f_2
                     ])

f_x = lambda t, y, f: np.array([y[1],
                                -6*y[0]-2*(y[0]-y[2])+f[0],
                                y[3],
                                +2*(y[0]-y[2])-3*y[2]+f[1]
                                 ])
t=np.linspace(0,10,51)
plt.plot(t,y_1_analytic(t),'--',label='analytic $$(y_1)$$')
plt.plot(t,y_2_analytic(t),'--',label='analytic $$(y_2)$$')

yrk = solve_ivp(f_x, (t[0],t[-1]) , [0,1,0,0], method='RK45', 
                args=np.array([[0,0]]),t_eval=t)
plt.plot(yrk.t,yrk.y[0,:],'o');
plt.plot(yrk.t,yrk.y[2,:],'s');

fnrk = solve_ivp(fn_x, (t[0],t[-1]) , [0,1,0,0], method='RK45', 
                 args=np.array([[0,0]]), t_eval=t)
plt.plot(fnrk.t,fnrk .y[0,:],'o--');
plt.plot(fnrk.t,fnrk .y[2,:],'s--');

"Part (d)..."
H = lambda t: 1*(t>0)
f_x = lambda t, y: np.array([y[1],
                                -6*y[0]-2*(y[0]-y[2])+2*(1-H(t-2)),
                                y[3],
                                +2*(y[0]-y[2])-3*y[2]+0
                                 ])
yrk = solve_ivp(f_x, (t[0],t[-1]) , [0,0,0,0], method='RK45', t_eval=t)
#plt.plot(yrk.t, yrk.y[1,:])

exprs = solve([eq1.subs(F_1,2/s),eq2.subs(F_2,0)],[Y_1, Y_2])
t = symbols('t',positive=True)
expr_y1=inverse_laplace_transform(exprs[Y_1],s ,t)
expr_y2=inverse_laplace_transform(exprs[Y_2],s ,t)
y_1_analytic = lambdify(t, expr_y1)
y_2_analytic = lambdify(t, expr_y2)
t = np.linspace(0,10,51)

plt.figure()
plt.plot(yrk.t, yrk.y[0,:],'ko')
np.linspace(0,10,1001)
plt.plot(t,y_1_analytic(t)-H(t-2)*y_1_analytic(t-2),'k')


"Part (e) ..."
H = lambda t: 1*(t>0)
f_x = lambda t, y, c_1, c_2: np.array([y[1],
                                -6*y[0]-c_1*y[1]-2*(y[0]-y[2])-c_2*(y[1]-y[3])+2*(1-H(t-2)),
                                y[3],
                                +2*(y[0]-y[2])+c_2*(y[1]-y[3])-3*y[2]+0
                                 ])
t=np.linspace(0,20,1001)
yrk = solve_ivp(f_x, (t[0],t[-1]) , [0,0,0,0], method='RK45', 
                args=(0.5,0.5), t_eval=t)

plt.figure()
plt.plot(yrk.t,yrk.y[0,:],'--')




