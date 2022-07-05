# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:04:20 2021

@author: Josh
"""


from sympy.abc import t
from sympy import init_printing, plot, lambdify
from sympy import symbols
from sympy import Function, Derivative, dsolve, oo, Eq, solve
import numpy as np

''' Import Data '''

cooling_data = np.genfromtxt('cooling_data.txt')
#my_data = genfromtxt('SNData.csv', delimiter=",")

T = Function ('T')
k, T_amb, T_o = symbols ('k, T_amb, T_o', positive=True)

eq1 = Derivative(T(t),t)+k*(T(t)-T_amb)
eq2 = dsolve(eq1,T(t),ics={T(0): T_o})

eq3 = eq2.subs({T_o: 727, T_amb: 27})
eq4 = eq3.subs({T(t): 150, t: 4})
#solve(eq4, k)
eq5 = Eq(k,solve(eq4,k)[0])

eq6 = eq3.subs(k, 0.433)
plot(eq6.rhs,(t,0,15))

#import numpy as np
import matplotlib.pyplot as plt

t_s=np.linspace(0,15,num=25);
T_s=lambdify(t, eq6.rhs)
plt.plot(t_s,T_s(t_s),'ko', label = 'Analytical')
plt.plot(cooling_data[:,0], cooling_data[:,1:6], 'b--', label = 'XCOS')
plt.xlabel('Time (Hrs)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid()
#plt.errorbar(t_s,T_s(t_s),'.')
