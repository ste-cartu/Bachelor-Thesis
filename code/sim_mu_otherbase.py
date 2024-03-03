import os
import numpy as np
from pysr import *
from importlib import reload
import functions as f

# ricaricare f se ci sono modifiche nel file 'functions.py'
# reload(f)

# parametri cosmologici
nk = 100 
nz = 20
nc = 500

# parametri del modello
ni = 2000
comp = 30
pop = 100



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CREAZIONE DEL DATASET SU CUI ALLENARE IL MODELLO

if os.path.exists('../files/data_otherbase_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + '].npy') :
    data = np.load('../files/data_otherbase_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + '].npy') 
else :
    os.system(f'python3 gen_data_otherbase.py {nk} {nz} {nc}')
    data = np.load('../files/data_otherbase_[' + str(nk) + ',' + str(nz) + ',comb=' + str(nc) + '].npy')

# l'array 'data' è così strutturato:
# Omega_nu | k [1/Mpc] (scala) | redshift | Omega_b | Omega_c | h | valore di mu


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
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
    equation_file = '../models/mu_bch_otherbase(' + str(ni) + ',' + str(comp) + ',' + str(pop) + ').csv'
)

model_mu.fit(data[:,:-1], data[:,-1])