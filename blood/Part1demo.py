# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:34:21 2021

@author: JoshuaL
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares

''' Import Data '''
ViscosityData = np.loadtxt('blud_data.txt')

''' Creat optimize model '''
μ_cross = lambda μ_o, μ_oo, λ, n, Y: μ_oo + (μ_o-μ_oo)/(1+(λ*Y)**(1-n))
μ_casson = lambda τ_Y, μ_PL, Y: (np.sqrt(τ_Y/Y)+np.sqrt(μ_PL))**2

f_cross = lambda a, x, y: μ_cross(a[0], a[1], a[2], a[3], x)-y
f_casson = lambda a, x, y: μ_casson(a[0], a[1], x)-y

lse_cross = least_squares(f_cross,[1,1,0,0],
                          args=(ViscosityData[:,0],ViscosityData[:,1]))
lse_casson = least_squares(f_casson,[0,0],
                          args=(ViscosityData[:,0],ViscosityData[:,1]))


''' Plot Data '''
plt.figure(1)
plt.semilogx(ViscosityData[:,0], ViscosityData[:,1], 'ko', label = 'Rheological data')
# Cross
plt.semilogx(np.logspace(0,3,101), 
             μ_cross(lse_cross.x[0], lse_cross.x[1], 
                     lse_cross.x[2], lse_cross.x[3],
                     np.logspace(0,3,101)), 'g-', label = 'Cross(1965)')
# Casson
#plt.semilogx(np.logspace(0,3,101), 
             #μ_casson(lse_casson.x[0], lse_casson.x[1], 
                      #np.logspace(0,3,101)), 'g--', label = 'Casson(1959)')
# Add graph labels
plt.xlabel('Shear Strain Rate (Hz)')
plt.ylabel('Effective Viscosity (cP)')
plt.legend()
plt.grid()

''' Error norms '''
from numpy.linalg import norm

crossL1 = 100*norm(ViscosityData[:,1] - μ_cross(lse_cross.x[0], lse_cross.x[1], 
                                     lse_cross.x[2], lse_cross.x[3],
                                     np.logspace(0,3,len(ViscosityData[:,1]))),1)/ \
                                        norm(ViscosityData[:,1],1)
crossL2 = 100*norm(ViscosityData[:,1] - μ_cross(lse_cross.x[0], lse_cross.x[1], 
                                     lse_cross.x[2], lse_cross.x[3],
                                     np.logspace(0,3,len(ViscosityData[:,1]))),1)/ \
                                        norm(ViscosityData[:,1],2)                                        
