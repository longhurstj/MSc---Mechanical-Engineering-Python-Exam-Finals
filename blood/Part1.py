''' Part 1 '''

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares

μ_cross = lambda μ_o,μ_oo,n,λ,Y: μ_oo+(μ_o-μ_oo)/(1+(λ*Y)**(1-n))
μ_casson = lambda τ_y, μ_PL, Y: (np.sqrt(τ_y/Y)+np.sqrt(μ_PL))**2

fn_cross = lambda a,x,y: μ_cross(a[0],a[1],a[2],a[3],x)-y
fn_casson = lambda a,x,y: μ_casson(a[0],a[1],x)-y

ViscocityData=np.loadtxt('Blood_Viscocity_Data.txt')

lse_cross=least_squares(fn_cross,[1,1,0,0],
                      args=(ViscocityData[:,0],ViscocityData[:,1]))
lse_casson=least_squares(fn_casson,[1,1],
                      args=(ViscocityData[:,0],ViscocityData[:,1]))

plt.figure(0)
plt.semilogx(ViscocityData[:,0],ViscocityData[:,1],'ko', 
             label='Rheological data')
plt.semilogx(np.logspace(0,3,101),
             μ_cross(lse_cross.x[0],lse_cross.x[1],lse_cross.x[2],lse_cross.x[3],
                     np.logspace(0,3,101)),'--', label='Cross(1965)')
plt.semilogx(np.logspace(0,3,101),
             μ_casson(lse_casson.x[0],lse_casson.x[1],np.logspace(0,3,101)), 
             '--',label='Casson(1959)')
plt.xlabel('Shear strain rate (Hz)')
plt.ylabel('Viscosity (cP)')
plt.grid(); plt.legend();