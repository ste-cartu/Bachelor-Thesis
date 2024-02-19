import os
import numpy as np
from pysr import *
from matplotlib import pyplot as plt
from importlib import reload
#from classy import Class
import functions as f
import sys

# ricaricare f se ci sono modifiche nel file 'functions.py'
# reload(f)

nm = 10
nk = 100 
nz = 20
nas = 20
nns = 20

ni = 4000
comp = 40 
pop = 120 


# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# DEFINIZIONE VARIABILI

if os.path.exists('files/neutrino_mass_[' + str(nm) +'].npy') :
    mm = np.load('files/neutrino_mass_[' + str(nm) +'].npy')
else :
    mm = f.Masses(0.06, 1, nm, 'files')

if os.path.exists('files/Mu(k,z)_[' + str(nk) + ',' + str(nz) + ']_m=0.06.npy') :
    mu = np.zeros([nm, nk, nz])
    for m in range(nm) :
        mupath = 'files/Mu(k,z)_[' + str(nk) + ',' + str(nz) + ']_m=' + str(round(mm[m], 3)) + '.npy'
        mu[m,:,:] = np.load(mupath)
else :
    cosmos = []
    mus =  []
    for i in range(nm) :
        cosmos.append(f.DataFromCLASS(nk, nz, mm[i], 'files'))
        mus.append(cosmos[i]['growth_ratio'])
    mu = np.array(mus)

kk = np.load('files/scale_[' + str(nk) +'].npy')
zz = np.load('files/redshift_[' + str(nz) +'].npy')

asas = np.linspace(2.5, 3.5, nas)
nsns = np.linspace(0.9, 1, nns)


# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CONTROLLO SU MU

if (len(mu[:,0,0]) != nm) :
    print('Error! dim(Mu[:,0,0]) ≠ dim(mass)')
elif (len(mu[0,:,0]) != nk) :
    print('Error! dim(Mu[0,:,0]) ≠ dim(scale)')
elif (len(mu[0,0,:]) != nz) :
    print('Error! dim(Mu[0,:,0]) ≠ dim(redshift)')
'''
for m in range(nm) :
    for k in range(nk) :
        for z in range(nz) :
            if mu[m,k,z] - mus[m][k][z] != 0 :
                print('Error! mu_list[' +str(m)+ ',' +str(k)+ ',' +str(z)+'] ≠ mu_array[' +str(m)+ ',' +str(k)+ ',' +str(z)+']')
'''

# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CREAZIONE DEL MODELLO E ALLENAMENTO
                
model_mu = PySRRegressor(
    # operatori
    binary_operators = ['+', '-', '*', '/', '^'],
    unary_operators=['exp', 'log'],

    # complessità
    niterations = ni,
    maxsize = comp,
    populations = pop,

    batching = True,
    batch_size = 50,

    loss = 'L2DistLoss()',
    model_selection = 'best',
    equation_file = 'models/mu_mp(' +str(ni)+ ',' +str(comp)+ ',' +str(pop)+ ').csv'
    )

# creazione dei dati su cui allenare il modello, chiedere a Marco se è più efficiente così o salvare tutto su un dataframe
if os.path.exists('files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nas) + ',' + str(nns) + '].npy') :
    data = np.load('files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nas) + ',' + str(nns) + '].npy') 
else :
    data = np.zeros([nm*nk*nz*nas*nns, 6])
    for m in range(nm) :
        for k in range(nk) :
            for z in range(nz) :
                for a in range(nas) :
                    for n in range(nns) :
                        data[m*nk*nz*nas*nns + k*nz*nas*nns + z*nas*nns + a*nns + n, 0] = mm[m]
                        data[m*nk*nz*nas*nns + k*nz*nas*nns + z*nas*nns + a*nns + n, 1] = kk[k]
                        data[m*nk*nz*nas*nns + k*nz*nas*nns + z*nas*nns + a*nns + n, 2] = zz[z]
                        data[m*nk*nz*nas*nns + k*nz*nas*nns + z*nas*nns + a*nns + n, 3] = asas[a]
                        data[m*nk*nz*nas*nns + k*nz*nas*nns + z*nas*nns + a*nns + n, 4] = nsns[n]
                        data[m*nk*nz*nas*nns + k*nz*nas*nns + z*nas*nns + a*nns + n, 5] = mu[m,k,z]

    np.save('files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nas) + ',' + str(nns) + ']', data)

# l'array 'data' è così strutturato:
# masse neutrino [eV] | k [1/Mpc] (scala) | redshift | As | ns | valore di mu

model_mu.fit(data[:,:5], data[:,5])
