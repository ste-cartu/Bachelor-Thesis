import numpy as np
import functions as f
import sys

# parametri cosmologici
# nm = 10         # dimensione array delle masse del neutrino
# nk = 100        # dimensione array dei k
# nz = 20         # dimensione array dei redshift

nm = int(sys.argv[1])
nk = int(sys.argv[2])
nz = int(sys.argv[3])
filepath = '../files'


print('\ngenerating dataset: data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + '].npy\n')


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# MASSE DEL NEUTRINO, SCALA E REDSHIFT

mm = f.Masses(0.06, 1, nm, filepath)
kk = f.Scale(1e-4, 3, nk, filepath)
zz = f.Redshift(0, 5, nz, filepath)


# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# MU + CONTROLLO
'''
cosmos = []
mus =  []
for i in range(nm) :
    cosmos.append(f.DataFromCLASS(dim_k = nk,
                                  dim_z = nz,
                                  m_neutrino = mm[i],
                                  path = '../files'))
    mus.append(cosmos[i]['growth_ratio'])
mu = np.array(mus)

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
                print('Error! mu_list[' +str(m)+ ',' +str(k)+ ',' +str(z)+'] ≠ mu_array[' +str(m)+ ',' +str(k)+ ',' +str(z)+']')

'''


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# DATI SU CUI ALLENARE I MODELLI (m,k,z)

data = np.zeros([nm*nk*nz, 4])
for m in range(nm) :
    mu = f.DataFromCLASS(
    dim_k = nk,
    dim_z = nz,
    m_neutrino = mm[m],
    path = '../files')['growth_ratio']

    for k in range(nk) :
        for z in range(nz) :
            data[m*nk*nz+k*nz+z,0] = mm[m]
            data[m*nk*nz+k*nz+z,1] = kk[k]
            data[m*nk*nz+k*nz+z,2] = zz[z]
            data[m*nk*nz+k*nz+z,3] = mu[k,z]

            print(m*nk*nz + k*nz + z + 1, '/', nm*nk*nz)

        #print(data[m*nk*nz+k*nz+z,0], '\t', data[m*nk*nz+k*nz+z,1], '\t', data[m*nk*nz+k*nz+z,2], '\t', data[m*nk*nz+k*nz+z,3])

np.save('../files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ']', data)

# l'array 'data' è così strutturato:
# masse neutrino [eV] | k [1/Mpc] (scala) | redshift | valore di mu
