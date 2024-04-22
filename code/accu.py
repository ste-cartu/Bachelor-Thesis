import os
import numpy as np
from pysr import *
import functions as f
import sys


nm = 10
nk = 100
nz = 20

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ITERAZIONI
print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
print('ITERAZIONI\n')

#ni = 10000
comp = 20
pop = 15

iter = np.linspace(1000, 20000, 20).astype(int)
for ni in iter :

    if os.path.exists('../files/neutrino_mass_[' + str(nm) +'].npy') :
        mm = np.load('../files/neutrino_mass_[' + str(nm) +'].npy')
    else :
        mm = f.Masses(0.06, 1, nm, '../files')

    if os.path.exists('../files/Mu(k,z)_[' + str(nk) + ',' + str(nz) + ']_m=0.06.npy') :
        mu = np.zeros([nm, nk, nz])
        for m in range(nm) :
            mupath = '../files/Mu(k,z)_[' + str(nk) + ',' + str(nz) + ']_m=' + str(round(mm[m], 3)) + '.npy'
            mu[m,:,:] = np.load(mupath)
    else :
        cosmos = []
        mus =  []
        for i in range(nm) :
            cosmos.append(f.DataFromCLASS(nk, nz, mm[i], '../files'))
            mus.append(cosmos[i]['growth_ratio'])
        mu = np.array(mus)

    # creazione dei dati su cui allenare il modello
    if os.path.exists('../files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + '].npy') :
        data = np.load('../files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + '].npy')
    else :        
        data = np.zeros([nm*nk*nz, 4])
        for m in range(nm) :
            for k in range(nk) :
                for z in range(nz) :
                    data[m*nk*nz+k*nz+z,0] = mm[m]
                    data[m*nk*nz+k*nz+z,1] = kk[k]
                    data[m*nk*nz+k*nz+z,2] = zz[z]
                    data[m*nk*nz+k*nz+z,3] = mu[m,k,z]

        np.save('../files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ']', data)

    kk = np.load('../files/scale_[' + str(nk) +'].npy')
    zz = np.load('../files/redshift_[' + str(nz) +'].npy')

    # l'array 'data' è così strutturato:
    # masse neutrino [eV] | k [1/Mpc] (scala) | redshift | valore di mu

    if os.path.exists('../models/mu(' + str(ni) + ',' + str(comp) + ',' + str(pop) + ').pkl') :
        model_mu = PySRRegressor().from_file('../models/mu(' + str(ni) + ',' + str(comp) + ',' + str(pop) + ').pkl')
    else :
        model_mu = PySRRegressor(
            # operatori
            binary_operators = ['+', '-', '*', '/', '^'],
            unary_operators=['exp', 'log'],

            # complessità
            niterations = ni,
            populations = pop,
            maxsize = comp,

            batching = True,
            batch_size = 50,

            loss = 'L2DistLoss()',
            model_selection = 'best',
            equation_file = '../models/mu(' + str(ni) + ',' + str(comp) + ',' + str(pop) + ').csv'
            )

        model_mu.fit(data[:,:3], data[:,3])



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# COMPLESSITÀ

ni = 1000
#comp = 20
pop = 15

comple = np.linspace(20, 100, 21).astype(int)
for comp in comple :

    if os.path.exists('../files/neutrino_mass_[' + str(nm) +'].npy') :
        mm = np.load('../files/neutrino_mass_[' + str(nm) +'].npy')
    else :
        mm = f.Masses(0.06, 1, nm, '../files')

    if os.path.exists('../files/Mu(k,z)_[' + str(nk) + ',' + str(nz) + ']_m=0.06.npy') :
        mu = np.zeros([nm, nk, nz])
        for m in range(nm) :
            mupath = '../files/Mu(k,z)_[' + str(nk) + ',' + str(nz) + ']_m=' + str(round(mm[m], 3)) + '.npy'
            mu[m,:,:] = np.load(mupath)
    else :
        cosmos = []
        mus =  []
        for i in range(nm) :
            cosmos.append(f.DataFromCLASS(nk, nz, mm[i], '../files'))
            mus.append(cosmos[i]['growth_ratio'])
        mu = np.array(mus)

    kk = np.load('../files/scale_[' + str(nk) +'].npy')
    zz = np.load('../files/redshift_[' + str(nz) +'].npy')

    # creazione dei dati su cui allenare il modello
    
    if os.path.exists('../files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + '].npy') :
        data = np.load('../files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + '].npy')
    else :
        data = np.zeros([nm*nk*nz, 4])
        for m in range(nm) :
            for k in range(nk) :
                for z in range(nz) :
                    data[m*nk*nz+k*nz+z,0] = mm[m]
                    data[m*nk*nz+k*nz+z,1] = kk[k]
                    data[m*nk*nz+k*nz+z,2] = zz[z]
                    data[m*nk*nz+k*nz+z,3] = mu[m,k,z]

        np.save('../files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ']', data)

    # l'array 'data' è così strutturato:
    # masse neutrino [eV] | k [1/Mpc] (scala) | redshift | valore di mu
    
    if os.path.exists('../models/mu(' + str(ni) + ',' + str(comp) + ',' + str(pop) + ').csv') :
        model_mu = PySRRegressor().from_file('../models/mu(' + str(ni) + ',' + str(comp) + ',' + str(pop) + ').pkl')
    else :
        model_mu = PySRRegressor(
            # operatori
            binary_operators = ['+', '-', '*', '/', '^'],
            unary_operators=['exp', 'log'],

            # complessità
            niterations = ni,
            populations = pop,
            maxsize = comp,

            batching = True,
            batch_size = 50,

            loss = 'L2DistLoss()',
            model_selection = 'best',
            equation_file = '../models/mu(' + str(ni) + ',' + str(comp) + ',' + str(pop) + ').csv'
            )
        
        model_mu.fit(data[:,:3], data[:,3])



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# POPOLAZIONI

ni = 1000
comp = 20
#pop = 15

popul = np.linspace(15, 300, 20).astype(int)
for pop in popul :

    if os.path.exists('../files/neutrino_mass_[' + str(nm) +'].npy') :
        mm = np.load('../files/neutrino_mass_[' + str(nm) +'].npy')
    else :
        mm = f.Masses(0.06, 1, nm, '../files')

    if os.path.exists('../files/Mu(k,z)_[' + str(nk) + ',' + str(nz) + ']_m=0.06.npy') :
        mu = np.zeros([nm, nk, nz])
        for m in range(nm) :
            mupath = '../files/Mu(k,z)_[' + str(nk) + ',' + str(nz) + ']_m=' + str(round(mm[m], 3)) + '.npy'
            mu[m,:,:] = np.load(mupath)
    else :
        cosmos = []
        mus =  []
        for i in range(nm) :
            cosmos.append(f.DataFromCLASS(nk, nz, mm[i], '../files'))
            mus.append(cosmos[i]['growth_ratio'])
        mu = np.array(mus)

    kk = np.load('../files/scale_[' + str(nk) +'].npy')
    zz = np.load('../files/redshift_[' + str(nz) +'].npy')

    # creazione dei dati su cui allenare il modello
    if os.path.exists('../files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + '].npy') :
        data = np.load('../files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + '].npy')
    else :
        data = np.zeros([nm*nk*nz, 4])
        for m in range(nm) :
            for k in range(nk) :
                for z in range(nz) :
                    data[m*nk*nz+k*nz+z,0] = mm[m]
                    data[m*nk*nz+k*nz+z,1] = kk[k]
                    data[m*nk*nz+k*nz+z,2] = zz[z]
                    data[m*nk*nz+k*nz+z,3] = mu[m,k,z]

        np.save('../files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ']', data)

    # l'array 'data' è così strutturato:
    # masse neutrino [eV] | k [1/Mpc] (scala) | redshift | valore di mu

    if os.path.exists('../models/mu(' + str(ni) + ',' + str(comp) + ',' + str(pop) + ').pkl') :
        model_mu = PySRRegressor().from_file('../models/mu(' + str(ni) + ',' + str(comp) + ',' + str(pop) + ').pkl')
    else :
        model_mu = PySRRegressor(
            # operatori
            binary_operators = ['+', '-', '*', '/', '^'],
            unary_operators=['exp', 'log'],

            # complessità
            niterations = ni,
            populations = pop,
            maxsize = comp,

            batching = True,
            batch_size = 50,

            loss = 'L2DistLoss()',
            model_selection = 'best',
            equation_file = '../models/mu(' + str(ni) + ',' + str(comp) + ',' + str(pop) + ').csv'
            )

        model_mu.fit(data[:,:3], data[:,3])

