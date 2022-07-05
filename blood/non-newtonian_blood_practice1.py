# -*- coding: utf-8 -*-
"""
Created on Mon May 17 20:56:22 2021

@author: JoshuaL
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares

''' Import Data '''
surgeon_viscosity_data = np.loadtxt('surgeon_data_transpose.txt')

''' Creat optimize model '''
μ_cross = lambda μ_o, μ_oo, λ, n, Y: μ_oo + (μ_o-μ_oo)/(1+(λ*Y)**(1-n))
μ_casson = lambda τ_Y, μ_PL, Y: (np.sqrt(τ_Y/Y)+np.sqrt(μ_PL))**2

f_cross = lambda a, x, y: μ_cross(a[0], a[1], a[2], a[3], x)-y
f_casson = lambda a, x, y: μ_casson(a[0], a[1], x)-y

surgeon_lse_cross = least_squares(f_cross,[1,1,0,0],
                          args=(surgeon_viscosity_data[1:,0],surgeon_viscosity_data[1:,1]))
surgeon_lse_casson = least_squares(f_casson,[0,0],
                          args=(surgeon_viscosity_data[1:,0],surgeon_viscosity_data[1:,1]))

''' Plot Data '''
plt.figure(1)
plt.semilogx(surgeon_viscosity_data[1:,0], surgeon_viscosity_data[1:,1], '.', label = 'Sample A')
plt.semilogx(surgeon_viscosity_data[1:,0], surgeon_viscosity_data[1:,2], '.', label = 'Sample B')
plt.semilogx(surgeon_viscosity_data[1:,0], surgeon_viscosity_data[1:,3], '.', label = 'Sample C')
plt.semilogx(surgeon_viscosity_data[1:,0], surgeon_viscosity_data[1:,4], '.', label = 'Sample D')
plt.semilogx(surgeon_viscosity_data[1:,0], surgeon_viscosity_data[1:,5], '.', label = 'Sample E')

# Cross
plt.semilogx(np.logspace(0,3,101), 
             μ_cross(surgeon_lse_cross.x[0], surgeon_lse_cross.x[1], 
                     surgeon_lse_cross.x[2], surgeon_lse_cross.x[3],
                     np.logspace(0,3,101)), 'r--', label = 'Cross(1965)')
# Casson
plt.semilogx(np.logspace(0,3,101), 
             μ_casson(surgeon_lse_casson.x[0], surgeon_lse_casson.x[1], 
                      np.logspace(0,3,101)), 'g--', label = 'Casson(1959)')
# Add graph labels
plt.xlabel('Shear Strain Rate (Hz)')
plt.ylabel('Viscosity (cP)')
plt.legend()
plt.grid()