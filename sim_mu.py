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
ni = 4000       # numero di iterazioni del modello                                  DEFAULT 40 / BEST 4000
comp = 40       # complessità del modello (numero degli elementi nell'equazione)    DEFAULT 20 / BEST 40
pop = 120       # numero di popolazioni da cui parte il modello                     DEFAULT 15 / BEST 120

# per inserire i parametri da terminale
# ni = int(sys.argv[1])
# comp = int(sys.argv[2])
# pop = int(sys.argv[3])



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# CREAZIONE DEL DATASET SU CUI ALLENARE IL MODELLO, chiedere a Marco se è più efficiente così o salvare tutto su un dataframe

if os.path.exists('files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + '].npy') :
    data = np.load('files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + '].npy')
else :
    os.system(f'python3 gen_data.py {nm} {nk} {nz}')
    data = np.load('files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + '].npy')


# l'array 'data' è così strutturato:
# masse neutrino [eV] | k [1/Mpc] (scala) | redshift | valore di mu



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
