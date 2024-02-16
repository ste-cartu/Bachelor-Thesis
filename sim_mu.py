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

# parametri cosmologici
nm = 10         # dimensione array delle masse del neutrino
nk = 100        # dimensione array dei k
nz = 20         # dimensione array dei redshift

# parametri del modello
ni = 50       # numero di iterazioni del modello                                  DEFAULT 40 / BEST 5000
comp = 20       # complessità del modello (numero degli elementi nell'equazione)    DEFAULT 20 / BEST 50
pop = 15       # numero di popolazioni da cui parte il modello                     DEFAULT 15 / BEST 150

'''
nm = int(sys.argv[1])
nk = int(sys.argv[2])
nz = int(sys.argv[3])
ni = int(sys.argv[4])
comp = int(sys.argv[6])
pop = int(sys.argv[5])
'''

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
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

    # controllo su mu
    if (len(mu[:,0,0]) != nm) :
        print('Error! dim(Mu[:,0,0]) ≠ dim(mass)')
    elif (len(mu[0,:,0]) != nk) :
        print('Error! dim(Mu[0,:,0]) ≠ dim(scale)')
    elif (len(mu[0,0,:]) != nz) :
        print('Error! dim(Mu[0,:,0]) ≠ dim(redshift)')
    
    for m in range(nm) :
        for k in range(nk) :
            for z in range(nz) :
                if mu[m,k,z] - mus[m][k][z] != 0 :
                    print('Error! mu_list['+str(m)+','+str(k)+','+str(z)+'] ≠ mu_array['+str(m)+','+str(k)+','+str(z)+']')

kk = np.load('files/scale_[' + str(nk) +'].npy')
zz = np.load('files/redshift_[' + str(nz) +'].npy')


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CREAZIONE DEL DATASET SU CUI ALLENARE IL MODELLO, chiedere a Marco se è più efficiente così o salvare tutto su un dataframe

if os.path.exists('files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + '].npy') :
    data = np.load('files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + '].npy')
else :
    data = np.zeros([nm*nk*nz, 4])
    for m in range(nm) :
        for k in range(nk) :
            for z in range(nz) :
                data[m*nk*nz+k*nz+z,0] = mm[m]
                data[m*nk*nz+k*nz+z,1] = kk[k]
                data[m*nk*nz+k*nz+z,2] = zz[z]
                data[m*nk*nz+k*nz+z,3] = mu[m,k,z]

            #print(data[m*nk*nz+k*nz+z,0],'\t',data[m*nk*nz+k*nz+z,1],'\t',data[m*nk*nz+k*nz+z,2],'\t',data[m*nk*nz+k*nz+z,3])
    
    np.save('files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ']', data)

# l'array 'data' è così strutturato:
# masse neutrino [eV] | k [1/Mpc] (scala) | redshift | valore di mu
            
# data1 = np.load('data1.npy')
# print(data-data1)

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CREAZIONE DEL MODELLO E ALLENAMENTO

if os.path.exists('models/mu(' + str(ni) + ',' + str(comp) + ',' + str(pop) + ').pkl') :
        model_mu = PySRRegressor().from_file('models/mu(' + str(ni) + ',' + str(comp) + ',' + str(pop) + ').pkl')
else :
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
        equation_file = 'models/mu(' + str(ni) + ',' + str(comp) + ',' + str(pop) + ').csv'
        )

    model_mu.fit(data[:,:3], data[:,3])
