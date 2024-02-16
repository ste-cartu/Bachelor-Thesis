import numpy as np
import functions as f

# parametri cosmologici
nm = 10         # dimensione array delle masse del neutrino
nk = 100        # dimensione array dei k
nz = 20         # dimensione array dei redshift
# nas = 20
# nns = 20


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# MASSE DEL NEUTRINO

mm = f.Masses(0.06, 1, nm, 'files')



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# MU + CONTROLLO

cosmos = []
mus =  []
for i in range(nm) :
    cosmos.append(f.DataFromCLASS(nk, nz, mm[i], 'files'))
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



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# SCALA

kk = np.load('files/scale_[' + str(nk) +'].npy')



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# REDSHIFT

zz = np.load('files/redshift_[' + str(nz) +'].npy')



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# AS

# asas = np.linspace(2.5, 3.5, nas)



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# NS

# nsns = np.linspace(0.9, 1, nns)



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# DATI SU CUI ALLENARE I MODELLI (m,k,z)

data = np.zeros([nm*nk*nz, 4])
for m in range(nm) :
    for k in range(nk) :
        for z in range(nz) :
            data[m*nk*nz+k*nz+z,0] = mm[m]
            data[m*nk*nz+k*nz+z,1] = kk[k]
            data[m*nk*nz+k*nz+z,2] = zz[z]
            data[m*nk*nz+k*nz+z,3] = mu[m,k,z]

        #print(data[m*nk*nz+k*nz+z,0], '\t', data[m*nk*nz+k*nz+z,1], '\t', data[m*nk*nz+k*nz+z,2], '\t', data[m*nk*nz+k*nz+z,3])

np.save('files/data_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ']', data)

# l'array 'data' è così strutturato:
# masse neutrino [eV] | k [1/Mpc] (scala) | redshift | valore di mu



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# DATI SU CUI ALLENARE I MODELLI (m,k,z,as,ns)
'''
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
'''
# l'array 'data' è così strutturato:
# masse neutrino [eV] | k [1/Mpc] (scala) | redshift | As | ns | valore di mu