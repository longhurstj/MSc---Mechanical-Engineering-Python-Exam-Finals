#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 02:05:08 2021

@author: jwhitty
"""
from sympy.abc import x, t, A, B 
from sympy import symbols, Eq, solve, plot
from sympy import exp, sin, cos, pi
from sympy import Sum, Function, Integral
from sympy import lambdify, pprint

λ, ξ = symbols('lamda xi')
n, m, N = symbols('n, m, N', integer=True)

"Assume the solution..."
u = exp(-λ**2*t)*(A*cos(λ*x)+B*sin(λ*x))
"The BCs..."
u = u.subs(A,0); u = u.subs(λ, n*pi)/B
"and the IC ..."
f = lambda x: x*(1-x)
B = 2*Integral(u.subs({t:0,x:ξ})*f(ξ),(ξ,0,1)).doit()

"Part (a)..."
"Solution implelemtation"
u =Sum(B.subs(n,n)*u,(n,1,N))
pprint(u.args[0].args[0][0])
u_analytic = lambdify((x,t,N),u.subs(N,N).doit())


"Part (b)..."
import numpy as np
import matplotlib.pyplot as plt

xv,tv = np.meshgrid(np.linspace(0,1,31),
                    np.linspace(0,1,31))
plt.contourf(xv,tv,u_analytic(xv,tv,25))
plt.colorbar()

"Part (c)"
pprint(u.subs(N,16).doit())
plt.plot(np.linspace(0,1,31),f(np.linspace(0,1,31)))
plt.plot(np.linspace(0,1,31),
         u_analytic(np.linspace(0,1,31),0,20),'kx')

"Part (d): implementation of FD scheme..."
"Forward in time ... Central in space"
from scipy.interpolate import interp2d
from numpy.linalg import norm "Cool can be used on matrices..."
def uFDiff(x,bcs,f):
    bcs=(0,0)
    N=len(x); h_x=x[1]-x[0]; # Spatial spacing .
    h_t=(h_x/2)**2; M=int(np.ceil(1/h_t)) # Temporal spacing
    u_FD=np.zeros((M,N)); # Initialization
    u_FD[:,0]=bcs[0]; u_FD[0,:]=f(x); u_FD[:,-1]=bcs[-1];
    λ_1=(h_t/h_x**2)
    for m in range(0,M-1):  
        for n in range(1,N-1):
            u_FD[m+1,n]=u_FD[m,n]+λ_1*(u_FD[m,n-1]-2*u_FD[m,n]+u_FD[m,n+1]);
    return u_FD

u_FD=uFDiff(np.linspace(0,1,31),(0,0),f)
u_interp=interp2d(np.linspace(0,1,np.shape(u_FD)[1]),
                  np.linspace(0,1,np.shape(u_FD)[0]),u_FD)                 
uFD_iterp=u_interp(np.linspace(0,1,31),np.linspace(0,1,31))
plt.contourf(xv,tv,uFD_iterp)

plt.colorbar()
uFD_iterp-u_analytic(xv,tv,25)
100*norm(uFD_iterp-u_analytic(xv,tv,25))/ \
norm(u_analytic(xv,tv,25))


