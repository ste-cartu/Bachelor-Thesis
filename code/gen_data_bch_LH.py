import numpy as np
import functions as f
import sys
from scipy.stats import qmc

# parametri cosmologici
# nk = 100        # dimensione array dei k
# nz = 20         # dimensione array dei redshift
# nc = 200        # dimensione array delle combinazioni di omega_b, omega_c, h

nk = int(sys.argv[1])
nz = int(sys.argv[2])
nc = int(sys.argv[3])
filepath = '../files'




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# REDSHIFT

zz = f.Redshift(0, 5, nz, filepath)




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# MASSE DEL NEUTRINO, OMEGA_B, OMEGA_C, H

# ipercubo 4d con nc combinazioni
sampler = qmc.LatinHypercube(d=4)
sample = sampler.random(n=nc)

# ipercubo: [massa, omega_b,    omega_c,    h]
inf =       [0.06,  0.020,      0.10,     0.6]
sup =       [1,     0.024,      0.14,     0.8]
sample = qmc.scale(sample, inf, sup)




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# DATI SU CUI ALLENARE I MODELLI (m,omega_b,omega_c,h,k,z)

print('\ngenerating dataset: data_bch_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + '].npy\n')

data = np.zeros([nc*nk*nz, 7])
for c in range(nc) :
    cosmo = f.DataFromCLASS(
        dim_k = nk,
        dim_z = nz,
        m_neutrino = sample[c,0],
        omega_b = sample[c,1],
        omega_c = sample[c,2],
        h = sample[c,3], 
        path = '../files')
    
    kk = cosmo['scale']
    mu = cosmo['growth_ratio']
    
    for k in range(nk) :
        for z in range(nz) :
            data[c*nk*nz + k*nz + z, 0] = sample[c,0]
            data[c*nk*nz + k*nz + z, 1] = kk[k]
            data[c*nk*nz + k*nz + z, 2] = zz[z]
            data[c*nk*nz + k*nz + z, 3] = sample[c,1]
            data[c*nk*nz + k*nz + z, 4] = sample[c,2]
            data[c*nk*nz + k*nz + z, 5] = sample[c,3]
            data[c*nk*nz + k*nz + z, 6] = mu[k,z]

            print(c*nk*nz + k*nz + z + 1, '/', nc*nk*nz)

np.save('../files/data_bch_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + ']', data)

# l'array 'data' è così strutturato:
# masse neutrino [eV] | k [h/Mpc] (scala) | redshift | omega_b | omega_c | h | valore di mu