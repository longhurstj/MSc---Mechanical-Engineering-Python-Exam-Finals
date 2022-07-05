#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 13:05:02 2021

@author: JoshuaL
"""
from sympy import symbols, Eq, solve
from sympy import inverse_laplace_transform

Y_1, Y_2, s, F_1, F_2 = symbols('Y_1 Y_2 s F_1 F_2')

eq1 = Eq(s**2*Y_1,-6*Y_1-(Y_1-Y_2)*2+F_1)
eq2 = Eq(s**2*Y_2, (Y_1-Y_2)*2-3*Y_2+F_2)

exprs = solve([eq1.subs(F_1,1),eq2.subs(F_2,2)],[Y_1, Y_2])

t = symbols('t',positive=True)
inverse_laplace_transform(exprs[Y_1],s ,t)
