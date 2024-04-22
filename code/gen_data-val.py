import numpy as np
#from pysr import *
import functions as f
import os
from scipy.stats import qmc

nm = 21
nk = 201
nz = 41
nas = 41
nns = 41
nob = 41
noc = 41
nh = 41
nc = 301
filepath = '../files'



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# PARAMETRI COSMOLOGICI

mm = f.Masses(0.06, 1, nm, filepath)
zz = f.Redshift(0, 5, nz, filepath)
asas = f.A_s(2.5, 3.5, nas, filepath)
nsns = f.N_s(0.9, 1, nns, filepath)
obob = f.Omega_b(0.020, 0.024, nob, filepath)
ococ = f.Omega_c(0.10, 0.14, noc, filepath)
hh = f.H(0.6, 0.8, nh, filepath)



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CREAZIONE DEL VALIDATION DATASET (m,k,z)

if os.path.exists('../files/data-val_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + '].npy') :
    data = np.load('../files/data-val_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + '].npy')
else :
    print('\ngenerating dataset: data-val_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + '].npy\n')
    data = np.zeros([nm*nk*nz, 4])
    for m in range(nm) :
        cosmo = f.DataFromCLASS(
            dim_k = nk,
            dim_z = nz,
            m_neutrino = mm[m],
            path = '../files')

        kk = cosmo['scale']
        mu = cosmo['growth_ratio']

        for k in range(nk) :
            for z in range(nz) :
                data[m*nk*nz+k*nz+z,0] = mm[m]
                data[m*nk*nz+k*nz+z,1] = kk[k]
                data[m*nk*nz+k*nz+z,2] = zz[z]
                data[m*nk*nz+k*nz+z,3] = mu[k,z]

                print(m*nk*nz + k*nz + z + 1, '/', nm*nk*nz)

    np.save('../files/data-val_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ']', data)

    # l'array 'data' è così strutturato:
    # masse neutrino [eV] | k [h/Mpc] (scala) | redshift | valore di mu



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CREAZIONE DEL VALIDATION DATASET (m,as,ns,k,z)
'''
if os.path.exists('../files/data-val_asns_[' +str(nm)+ ',' +str(nk)+ ',' +str(nz)+ ',' +str(nas)+ ',' +str(nns)+ '].npy') :   
    data = np.load('../files/data-val_asns_[' +str(nm)+ ',' +str(nk)+ ',' +str(nz)+ ',' +str(nas)+ ',' +str(nns)+ '].npy')
else :
    print('\ngenerating dataset: data-val_asns_[' +str(nm)+ ',' +str(nk)+ ',' +str(nz)+ ',' +str(nas)+ ',' +str(nns)+ '].npy\n')
    data = np.zeros([nm*nk*nz*nas*nns, 6])
    for m in range(nm) :
        for a in range(nas) :
            for n in range(nns) :
                cosmo = f.DataFromCLASS(
                    dim_k = nk,
                    dim_z = nz,
                    m_neutrino = mm[m],
                    ln_a_s = asas[a],
                    n_s = nsns[n], 
                    path = '../files')

                kk = cosmo['scale']
                mu = cosmo['growth_ratio']
                
                for k in range(nk) :
                    for z in range(nz) :
                        data[m*nas*nns*nk*nz + a*nns*nk*nz + n*nk*nz + k*nz + z, 0] = mm[m]
                        data[m*nas*nns*nk*nz + a*nns*nk*nz + n*nk*nz + k*nz + z, 1] = kk[k]
                        data[m*nas*nns*nk*nz + a*nns*nk*nz + n*nk*nz + k*nz + z, 2] = zz[z]
                        data[m*nas*nns*nk*nz + a*nns*nk*nz + n*nk*nz + k*nz + z, 3] = asas[a]
                        data[m*nas*nns*nk*nz + a*nns*nk*nz + n*nk*nz + k*nz + z, 4] = nsns[n]
                        data[m*nas*nns*nk*nz + a*nns*nk*nz + n*nk*nz + k*nz + z, 5] = mu[k,z]

                        print(m*nas*nns*nk*nz + a*nns*nk*nz + n*nk*nz + k*nz + z + 1, '/', nm*nas*nns*nk*nz)

    np.save('../files/data-val_asns_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nas) + ',' + str(nns) + ']', data)

    # l'array 'data' è così strutturato:
    # masse neutrino [eV] | k [h/Mpc] (scala) | redshift | As | ns | valore di mu
'''


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CREAZIONE DEL VALIDATION DATASET (m,omega_b,k,z)

if os.path.exists('../files/data-val_ob_[' +str(nm)+ ',' +str(nk)+ ',' +str(nz)+ ',' +str(nob)+ '].npy') :   
    data = np.load('../files/data-val_ob_[' +str(nm)+ ',' +str(nk)+ ',' +str(nz)+ ',' +str(nob)+ '].npy')
else :
    print('\ngenerating dataset: data-val_ob_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nob) + '].npy\n')
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

    np.save('../files/data-val_ob_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nob) + ']', data)

    # l'array 'data' è così strutturato:
    # masse neutrino [eV] | k [h/Mpc] (scala) | redshift | omega_b | valore di mu



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CREAZIONE DEL VALIDATION DATASET (m,omega_c,k,z)

if os.path.exists('../files/data-val_oc_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(noc) + '].npy') :
    data = np.load('../files/data-val_oc_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(noc) + '].npy')
else :
    print('\ngenerating dataset: data-val_oc_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(noc) + '].npy\n')
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

    np.save('../files/data-val_oc_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(noc) + ']', data)

    # l'array 'data' è così strutturato:
    # masse neutrino [eV] | k [h/Mpc] (scala) | redshift | omega_c | valore di mu



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CREAZIONE DEL VALIDATION DATASET (m,h,k,z)

if os.path.exists('../files/data-val_h_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nh) + '].npy') :
    data = np.load('../files/data-val_h_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nh) + '].npy')
else :
    print('\ngenerating dataset: data-val_h_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nh) + '].npy\n')
    data = np.zeros([nm*nh*nk*nz, 5])
    for m in range(nm) :
        for h in range(nh) :
            cosmo = f.DataFromCLASS(
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

    np.save('../files/data-val_h_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nh) + ']', data)

    # l'array 'data' è così strutturato:
    # masse neutrino [eV] | k [h/Mpc] (scala) | redshift | h | valore di mu



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CREAZIONE DEL VALIDATION DATASET (m,omega_b,omega_c,h,k,z)

if os.path.exists('../files/data-val_bch_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + '].npy') :
    data = np.load('../files/data-val_bch_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + '].npy')
else :
    print('\ngenerating dataset: data-val_bch_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + '].npy\n')

    # ipercubo 4d con nc combinazioni
    sampler = qmc.LatinHypercube(d=4)
    sample = sampler.random(n=nc)

    # ipercubo: [massa, omega_b,    omega_c,    h]
    inf =       [0.06,  0.020,      0.10,     0.6]
    sup =       [1,     0.024,      0.14,     0.8]
    sample = qmc.scale(sample, inf, sup)

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

    np.save('../files/data-val_bch_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + ']', data)

    # l'array 'data' è così strutturato:
    # masse neutrino [eV] | k [h/Mpc] (scala) | redshift | omega_b | omega_c | h | valore di mu




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CREAZIONE DEL VALIDATION DATASET (Omega_nu,Omega_b,Omega_c,h,k,z)

if os.path.exists('../files/data-val_VBCh_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + '].npy') :
    data = np.load('../files/data-val_VBCh_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + '].npy')
else :
    print('\ngenerating dataset: data-val_VBCh_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + '].npy\n')

    # ipercubo 4d con nc combinazioni
    sampler = qmc.LatinHypercube(d=4)
    sample = sampler.random(n=nc)

    # ipercubo: [Omega_nu,                  Omega_b,    Omega_c,   h]
    inf =       [0.06/(93.14*(0.8**2)),     0.046,      0.23,      0.6]
    sup =       [1/(93.14*(0.6**2)),        0.052,      0.29,      0.8]
    sample = qmc.scale(sample, inf, sup)

    data = np.zeros([nc*nk*nz, 7])
    for c in range(nc) :
        cosmo = f.DataFromCLASS2(
            dim_k = nk,
            dim_z = nz,
            Omega_nu = sample[c,0],
            Omega_b = sample[c,1],
            Omega_c = sample[c,2],
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

    np.save('../files/data-val_VBCh_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + ']', data)

    # l'array 'data' è così strutturato:
    # Omega_nu | k [h/Mpc] (scala) | redshift | Omega_b | Omega_c | h | valore di mu




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CREAZIONE DEL VALIDATION DATASET (Omega_nu,Omega_b,Omega_m,h,k,z)

if os.path.exists('../files/data-val_VBMh_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + '].npy') :
    data = np.load('../files/data-val_VBMh_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + '].npy')
else :
    print('\ngenerating dataset: data-val_VBMh_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + '].npy\n')

    # ipercubo 4d con nc combinazioni
    sampler = qmc.LatinHypercube(d=4)
    sample = sampler.random(n=nc)

    # Omega_m = Omega_nu + Omega_b + Omega_c, presi dal file 'gen_data_VBCh.py'
    # ipercubo: [Omega_nu,                  Omega_b,    Omega_m,   h]
    inf =       [0.06/(93.14*(0.8**2)),     0.04,       0.27,      0.6]
    sup =       [1/(93.14*(0.6**2)),        0.06,       0.40,      0.8]
    sample = qmc.scale(sample, inf, sup)

    data = np.zeros([nc*nk*nz, 7])
    for c in range(nc) :
        cosmo = f.DataFromCLASS2(
            dim_k = nk,
            dim_z = nz,
            Omega_nu = sample[c,0],
            Omega_b = sample[c,1],
            Omega_c = sample[c,2],
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

    np.save('../files/data-val_VBMh_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + ']', data)

    # l'array 'data' è così strutturato:
    # Omega_nu | k [h/Mpc] (scala) | redshift | Omega_b | Omega_m | h | valore di mu