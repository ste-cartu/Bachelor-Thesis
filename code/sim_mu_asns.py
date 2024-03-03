import os
import numpy as np
from pysr import *
from importlib import reload
import functions as f

# ricaricare f se ci sono modifiche nel file 'functions.py'
# reload(f)

# parametri cosmologici
nm = 10
nk = 100 
nz = 20
nas = 20
nns = 20

# parametri del modello
ni = 200
comp = 20
pop = 50



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CREAZIONE DEL DATASET SU CUI ALLENARE IL MODELLO

if os.path.exists('../files/data_asns_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nas) + ',' + str(nns) + '].npy') :
    data = np.load('../files/data_asns_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nas) + ',' + str(nns) + '].npy') 
else :
    os.system(f'python3 gen_data_asns.py {nm} {nk} {nz} {nas} {nns}')
    data = np.load('../files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nas) + ',' + str(nns) + '].npy')

# l'array 'data' è così strutturato:
# masse neutrino [eV] | k [1/Mpc] (scala) | redshift | As | ns | valore di mu
    


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
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
    equation_file = '../models/mu_asns(' +str(ni)+ ',' +str(comp)+ ',' +str(pop)+ ').csv'
)


model_mu.fit(data[:,:-1], data[:,-1])
