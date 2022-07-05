# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:30:10 2021

@author: JoshuaL
"""

from sympy.vector import CoordSys3D
from sympy import symbols, Eq, solve
from sympy import sin, cos, tan, pi
from sympy import atan

i=CoordSys3D(' ').i
j=CoordSys3D(' ').j
k=CoordSys3D(' ').k

sind = lambda x: sin(x*pi/180)
cosd = lambda x: cos(x*pi/180)
tand = lambda x: tan(x*pi/180)

r_DC = 0.6*j
r_CE = -0.3*i
r_CB = -0.6*i
r_AB = 0.6/tand(30)*i+0.6*j

v_OC = 0*i+(6*k).cross(r_DC)
w_ab, w_bc = symbols('w_ab, w_bc')

v_OE = v_OC + (w_bc*k).cross(r_CE)

v_OB = v_OC + (w_bc*k).cross(r_CB)
v_OB = (w_ab*k).cross(r_AB)

eqv = Eq(v_OC + (w_bc*k).cross(r_CB),(w_ab*k).cross(r_AB))

"equating components"
eq1 = eqv.lhs.dot(i)-eqv.rhs.dot(i)
eq2 = eqv.lhs.dot(j)-eqv.rhs.dot(j)
exprs = solve((eq1, eq2))
eqvE = v_OE.subs(w_bc,exprs[w_bc])
eqvE.magnitude()

(atan(eqvE.dot(j)/eqvE.dot(i))*180/pi).evalf()

"Acceleration..."
a_OC = 6**2*(-r_DC)
a_OE = a_OC-exprs[w_bc]**2*r_CE