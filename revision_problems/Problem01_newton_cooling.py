#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 20:39:33 2021

@author: jwhitty
"""

"Newton's law of cooling...."
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize  import fsolve

u_t = lambda t,u,k,T_oo: -k*(u-T_oo)

t =np.linspace(0,3,25); 

fn = lambda k: solve_ivp(lambda t,u: u_t(t, u, k, 26), 
              (t[0],t[-1]), (525,), method='RK45').y[0][-1]
k_val=fsolve(fn,0)[0]

urk = solve_ivp(lambda t,u: u_t(t, u, k_val, 26), (t[0],t[-1]), 
                (525,), method='RK45',t_eval=t)

plt.plot(urk.t,urk.y[0],'ko',label='Standard annealing')


"Part (c) ..."
t =np.linspace(0,5,25); 
u_5hrs = solve_ivp(lambda t,u: u_t(t, u, k_val/5, 26), (t[0],t[-1]), 
                (525,), method='RK45',t_eval=t).y[0][-1]

"Part (d) ... "
" Euler forward differences ... y(x+h)=h*y'(x)+y(h)"
t =np.linspace(0,3,25); h=t[1]-t[0]; 
uFD=np.array([500])
for n in range(len(t)-1):
    uFD=np.append(uFD,uFD[n]+h*u_t(t[n],uFD[n],k_val,25))
#plt.plot(t,uFD,'>')

"Central difference scheme y(x+h)=y(x-h)+2*h*y_x(x)"
h=t[1]-t[0]
uCD=np.array([500, 500+h*u_t(t[0],525,k_val,25)])
for n in range(2,len(t)):
    uCD=np.append(uCD,uCD[n-2]+2*h*u_t(t[n-1],uCD[n-1],k_val,25))
plt.plot(t,uCD,'v')

"Analytic verification..."
from sympy import symbols, Eq
from sympy import Function, Derivative, dsolve
from sympy import lambdify

t   = symbols('t') 
u   = Function('u')
u_t = Derivative('u(t)',t)
k, T_o, T_oo = symbols('k T_o T_oo',positive=True)

eqn = Eq(u_t,-k*(u(t)-T_oo))
eq1 =  dsolve(eqn,ics={u(0): T_o})
u = lambdify([t,k,T_o,T_oo], eq1.rhs)
t =np.linspace(0,3,25)
plt.plot(t,u(t, k_val, 525, 25),label='Analytic' )
plt.xlabel('time (h)'); plt.ylabel('Temperature (Celsius)'); plt.legend()