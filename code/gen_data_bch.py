import numpy as np
import functions as f
import sys

# parametri cosmologici
# nm = 10         # dimensione array delle masse del neutrino
# nk = 100        # dimensione array dei k
# nz = 20         # dimensione array dei redshift
# nob = 20        # dimensione array degli omega_b
# noc = 20        # dimensione array degli omega_c
# nh = 20         # dimensione array degli h

nm = int(sys.argv[1])
nk = int(sys.argv[2])
nz = int(sys.argv[3])
nob = int(sys.argv[4])
noc = int(sys.argv[5])
nh = int(sys.argv[6])
filepath = '../files'



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# MASSE DEL NEUTRINO, REDSHIFT, OMEGA_B, OMEGA_C, H

mm = f.Masses(0.06, 1, nm, filepath)
zz = f.Redshift(0, 5, nz, filepath)
obob = f.Omega_b(0.020, 0.024, nob, filepath)
ococ = f.Omega_c(0.10, 0.14, noc, filepath)
hh = f.H(0.6, 0.8, nh, filepath)



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# DATI SU CUI ALLENARE I MODELLI (m,omega_b,k,z)

print('\ngenerating dataset: data_ob_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nob) + '].npy\n')

data = np.zeros([nm*nob*nk*nz, 5])
for m in range(nm) :
    for b in range(nob) :
        cosmo = f.DataFromCLASS(
            dim_k = nk,
            dim_z = nz,
            m_neutrino = mm[m],
            omega_b = obob[b], 
            path = '../files')
        
        kk = cosmo['scale']
        mu = cosmo['growth_ratio']
        
        for k in range(nk) :
            for z in range(nz) :
                data[m*nob*nk*nz + b*nk*nz + k*nz + z, 0] = mm[m]
                data[m*nob*nk*nz + b*nk*nz + k*nz + z, 1] = kk[k]
                data[m*nob*nk*nz + b*nk*nz + k*nz + z, 2] = zz[z]
                data[m*nob*nk*nz + b*nk*nz + k*nz + z, 3] = obob[b]
                data[m*nob*nk*nz + b*nk*nz + k*nz + z, 4] = mu[k,z]

                print(m*nob*nk*nz + b*nk*nz + k*nz + z + 1, '/', nm*nob*nk*nz)

np.save('../files/data_ob_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nob) + ']', data)

# l'array 'data' è così strutturato:
# masse neutrino [eV] | k [h/Mpc] (scala) | redshift | omega_b | valore di mu



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# DATI SU CUI ALLENARE I MODELLI (m,omega_c,k,z)

print('\ngenerating dataset: data_oc_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(noc) + '].npy\n')

data = np.zeros([nm*noc*nk*nz, 5])
for m in range(nm) :
    for c in range(noc) :
        cosmo = f.DataFromCLASS(
            dim_k = nk,
            dim_z = nz,
            m_neutrino = mm[m],
            omega_c = ococ[c], 
            path = '../files')
        
        kk = cosmo['scale']
        mu = cosmo['growth_ratio']

        for k in range(nk) :
            for z in range(nz) :
                data[m*noc*nk*nz + c*nk*nz + k*nz + z, 0] = mm[m]
                data[m*noc*nk*nz + c*nk*nz + k*nz + z, 1] = kk[k]
                data[m*noc*nk*nz + c*nk*nz + k*nz + z, 2] = zz[z]
                data[m*noc*nk*nz + c*nk*nz + k*nz + z, 3] = ococ[c]
                data[m*noc*nk*nz + c*nk*nz + k*nz + z, 4] = mu[k,z]

                print(m*noc*nk*nz + c*nk*nz + k*nz + z + 1, '/', nm*noc*nk*nz)

np.save('../files/data_oc_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(noc) + ']', data)

# l'array 'data' è così strutturato:
# masse neutrino [eV] | k [h/Mpc] (scala) | redshift | omega_c | valore di mu



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# DATI SU CUI ALLENARE I MODELLI (m,omega_c,k,z)

print('\ngenerating dataset: data_h_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nh) + '].npy\n')

data = np.zeros([nm*nh*nk*nz, 5])
for m in range(nm) :
    for h in range(nh) :
        mu = f.DataFromCLASS(
            dim_k = nk,
            dim_z = nz,
            m_neutrino = mm[m],
            h = hh[h], 
            path = '../files')
        
        kk = cosmo['scale']
        mu = cosmo['growth_ratio']
        
        for k in range(nk) :
            for z in range(nz) :
                data[m*nh*nk*nz + h*nk*nz + k*nz + z, 0] = mm[m]
                data[m*nh*nk*nz + h*nk*nz + k*nz + z, 1] = kk[k]
                data[m*nh*nk*nz + h*nk*nz + k*nz + z, 2] = zz[z]
                data[m*nh*nk*nz + h*nk*nz + k*nz + z, 3] = hh[h]
                data[m*nh*nk*nz + h*nk*nz + k*nz + z, 4] = mu[k,z]

                print(m*nh*nk*nz + h*nk*nz + k*nz + z + 1, '/', nm*nh*nk*nz)

np.save('../files/data_h_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nh) + ']', data)

# l'array 'data' è così strutturato:
# masse neutrino [eV] | k [h/Mpc] (scala) | redshift | h | valore di mu