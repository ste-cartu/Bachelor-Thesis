import numpy as np
import functions as f
import sys

# parametri cosmologici
# nm = 10         # dimensione array delle masse del neutrino
# nk = 100        # dimensione array dei k
# nz = 20         # dimensione array dei redshift
# nas = 20        # dimensione dell'array degli As
# nns = 20        # dimensione dell'array degli ns

nm = int(sys.argv[1])
nk = int(sys.argv[2])
nz = int(sys.argv[3])
nas = int(sys.argv[4])
nns = int(sys.argv[5])
filepath = 'files'


print('\ngenerating dataset: data_asns_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nas) + ',' + str(nns) + '].npy\n')


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# MASSE DEL NEUTRINO, SCALA, REDSHIFT, AS, NS

mm = f.Masses(0.06, 1, nm, filepath)
kk = f.Scale(1e-4, 3, nk, filepath)
zz = f.Redshift(0, 5, nz, filepath)
asas = f.A_s(2.5, 3.5, nas, filepath)
nsns = f.N_s(0.9, 1, nns, filepath)



# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# DATI SU CUI ALLENARE I MODELLI (m,as,ns,k,z)

data = np.zeros([nm*nk*nz*nas*nns, 6])
for m in range(nm) :
    for a in range(nas) :
        for n in range(nns) :
            mu = f.DataFromCLASS(
                dim_k = nk, 
                dim_z = nz, 
                m_neutrino = mm[m], 
                ln_a_s = asas[a], 
                n_s = nsns[n], 
                path = 'files')['growth_ratio']
            
            for k in range(nk) :
                for z in range(nz) :
                    data[m*nas*nns*nk*nz + a*nns*nk*nz + n*nk*nz + k*nz + z, 0] = mm[m]
                    data[m*nas*nns*nk*nz + a*nns*nk*nz + n*nk*nz + k*nz + z, 1] = kk[k]
                    data[m*nas*nns*nk*nz + a*nns*nk*nz + n*nk*nz + k*nz + z, 2] = zz[z]
                    data[m*nas*nns*nk*nz + a*nns*nk*nz + n*nk*nz + k*nz + z, 3] = asas[a]
                    data[m*nas*nns*nk*nz + a*nns*nk*nz + n*nk*nz + k*nz + z, 4] = nsns[n]
                    data[m*nas*nns*nk*nz + a*nns*nk*nz + n*nk*nz + k*nz + z, 5] = mu[k,z]

                    print(m*nas*nns*nk*nz + a*nns*nk*nz + n*nk*nz + k*nz + z + 1, '/', nm*nas*nns*nk*nz)

np.save('files/data_asns_[' + str(nm) + ',' + str(nk) + ',' + str(nz) + ',' + str(nas) + ',' + str(nns) + ']', data)

# l'array 'data' è così strutturato:
# masse neutrino [eV] | k [1/Mpc] (scala) | redshift | As | ns | valore di mu