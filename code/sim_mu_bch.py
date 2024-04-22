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
nob = 20
noc = 20
nh = 20

# parametri del modello
ni = 1000
comp = 25 
pop = 30


# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CREAZIONE DEI DATASET SU CUI ALLENARE I MODELLI

# omega_b
if os.path.exists('../files/data_ob_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nob) + '].npy') :
    data_ob = np.load('../files/data_ob_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nob) + '].npy') 
else :
    os.system(f'python3 gen_data_bch.py {nm} {nk} {nz} {nob} {noc} {nh}')
    data_ob = np.load('../files/data_ob_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nob) + '].npy')

# l'array 'data' è così strutturato:
# masse neutrino [eV] | k [1/Mpc] (scala) | redshift | omega_b | valore di mu
    
# omega_c
if os.path.exists('../files/data_oc_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(noc) + '].npy') :
    data_oc = np.load('../files/data_oc_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(noc) + '].npy') 
else :
    os.system(f'python3 gen_data_bch.py {nm} {nk} {nz} {nob} {noc} {nh}')
    data_oc = np.load('../files/data_oc_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(noc) + '].npy')

# l'array 'data' è così strutturato:
# masse neutrino [eV] | k [1/Mpc] (scala) | redshift | omega_c | valore di mu
    
# h
if os.path.exists('../files/data_h_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nh) + '].npy') :
    data_h = np.load('../files/data_h_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nh) + '].npy') 
else :
    os.system(f'python3 gen_data_bch.py {nm} {nk} {nz} {nob} {noc} {nh}')
    data_h = np.load('../files/data_h_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nh) + '].npy')

# l'array 'data' è così strutturato:
# masse neutrino [eV] | k [1/Mpc] (scala) | redshift | h | valore di mu
    


# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CREAZIONE DEI MODELLI E ALLENAMENTO
                
model_mu_ob = PySRRegressor(
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
    equation_file = '../models/mu_ob(' +str(ni)+ ',' +str(comp)+ ',' +str(pop)+ ').csv'
)

model_mu_oc = PySRRegressor(
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
    equation_file = '../models/mu_oc(' +str(ni)+ ',' +str(comp)+ ',' +str(pop)+ ').csv'
)

model_mu_h = PySRRegressor(
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
    equation_file = '../models/mu_h(' +str(ni)+ ',' +str(comp)+ ',' +str(pop)+ ').csv'
)


model_mu_ob.fit(data_ob[:,:-1], data_ob[:,-1])
model_mu_oc.fit(data_oc[:,:-1], data_oc[:,-1])
model_mu_h.fit(data_h[:,:-1], data_h[:,-1])