#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:24:53 2023

@author: sgupt
"""

import numpy as np
import itertools
import os
directory_in_str = '/home/sgupt/POD_DMD_data/3e-5pr5/magnetic/Y10/Q_1.5/S_0/Ra993mag/velocity'

# Extracting the Ur values from the state files

file_list = os.listdir(directory_in_str)
def last_8chars(x):
    return(x[-8:])


 
h_0 = np.empty((3981312,900))
#arr = np.arange(47,1179600,96)
arr = np.arange(71,3981240,144)
h_0_eq = h_0[arr,0:900]
#h_0_s = h_0[3939840:3981312,:]



file_list_sorted = sorted(file_list, key = last_8chars)

    
i = 0    
for filename in file_list_sorted:    
    with open(filename,'rb') as thefile:
        
        #h_0[:,i] = np.loadtxt(filename,usecols=0)
        h_0_eq[:,i] = np.loadtxt(itertools.islice(thefile, 71, 3981240, 144),usecols=0)
        #h_0_s[:,i] = np.loadtxt(itertools.islice(thefile, 3939840, 3981312, 1),usecols=0)
        i += 1;
        
        
        
    
np.savetxt('Origmagnetic.dat',h_0_s[:,[800,825,850,899]])    
      
        
h_0_s = np.load('h_0_s.npy')        
        
        
#POD Computation  
        
        
        
        
        
import modred as mr

# Compute POD
#num_modes = 5

POD_res = mr.compute_POD_arrays_snaps_method(h_0_s[:,200:700])
POD_modes = POD_res.modes
eigvals = POD_res.eigvals
eigvecs = POD_res.eigvecs

#Save the first n dominant modes obtained from the POD analysis

np.savetxt('PODmodes.dat',POD_modes[:,0:10])  

      



#DMD Computation 

i = 0;
t = np.zeros(900)
filename = 'stimes'
with open(filename,'rb') as thefile:
    
    #h_0[:,i] = np.loadtxt(filename,usecols=0)
    t = np.loadtxt(thefile, usecols=0)
    i += 1;    
    
del_t = np.mean(np.diff(t))


import matplotlib.pyplot as plt
from pydmd import DMD


dmd = DMD(svd_rank=0, opt = True, sorted_eigs = 'real')
dmd.fit(h_0_s[:,500:700])
dmd.plot_eigs(show_axes=True, show_unit_circle=True)
eig = dmd.eigs
DMD_modes = dmd.modes
amp = dmd.amplitudes



eig_cont = np.log(eig) / del_t
beta = np.real(eig_cont)
freq = np.imag(eig_cont)

# extract real part
x = [ele.real for ele in eig_cont]
# extract imaginary part
y = [ele.imag for ele in eig_cont]
  
# plot the complex numbers
plt.scatter(y, x)
plt.xlabel('Imaginary')
plt.ylabel('Real')
plt.show()

np.savetxt('Frequency_text.dat',freq)
np.save('freq.npy', freq)
np.savetxt('Beta_text.dat',beta)
np.save('beta.npy', freq)

eig_ind_beta = beta.argsort()

eig_ind_freq = freq.argsort()

top_modes = [68, 66, 64, 32, 62, 0]

dmd_top_modes = np.real(DMD_modes[:,top_modes])
#np.savetxt('DMDanalysis.dat',dmd_top_modes)
np.savetxt('PyDMDmodes.dat',dmd_top_modes)

beta_top = beta[top_modes]
freq_top = freq[top_modes]



# POD and DMD correlation




nPOD = 800
nDMD = 291

proj_coeff = np.zeros((nDMD,nPOD))
proj = np.zeros((nDMD,))

for i in range(0,nPOD):
    for j in range(0,nDMD):    
        proj_coeff[j,i] = (np.inner(np.abs(DMD_modes[:,j]),POD_modes[:,i]))/(np.linalg.norm(np.abs(DMD_modes[:,j]))*np.linalg.norm(POD_modes[:,i]))
    
for i in range(0,nDMD):
    proj[i] = np.linalg.norm(proj_coeff[i,:])
    
proj_ind = proj.argsort()

m = np.zeros((5,))
beta_m = np.zeros((5,))
freq_m = np.zeros((5,))

for i in range(0,5):
    m[i] = proj_coeff[:,i].argmax() #Which DMD mode has the highest correlation with the given POD mode
    beta_m[i] = beta[int(m[i])]
    freq_m[i] = freq[int(m[i])]

#n = proj_coeff[79,:].argmax() #Which POD mode has the highest correlation with the given DMD mode


#x = np.unravel_index(proj_coeff.argmin(), proj_coeff.shape)


m_list = list(m)
m_list = [ int(x) for x in m_list ]
dmd_analysis_modes = np.real(DMD_modes[:,m_list])
np.savetxt('DMDanalysis.dat',dmd_analysis_modes)





# Save everything




np.save('proj_coeff.npy', proj_coeff)
np.save('DMD_modes.npy', DMD_modes)
np.save('POD_modes.npy', POD_modes)
np.save('h_0_eq.npy', h_0_eq)
