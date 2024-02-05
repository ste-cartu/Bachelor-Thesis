import matplotlib.pyplot as plt
import numpy as np
from classy import Class
import os



# funzione che, data in input la massa del neutrino, restituisce il power density spectrum P(k,z) o P(k,a) e il growth factor D(z) o D(a) e D(k,z) o D(k,a)


def DataFromCLASS(dim_k = 500, dim_z=200, m_neutrino = 0.06, path = 'files') :
        # creo il modello CLASS, imposto i suoi parametri e lo eseguo
        LCDM = Class()

        LCDM.set({'omega_b': 0.02238280, 
                'omega_cdm': 0.1201075, 
                'h': 0.67810, 
                'A_s': 2.100549e-09, 
                'n_s':0.9660499, 
                'tau_reio':0.05430842, 
                'z_max_pk':10,
                'N_ncdm': 1,
                'm_ncdm': m_neutrino          # massa del neutrino in eV
                })
        LCDM.set({'output': 'tCl,pCl,lCl,mPk',
                'lensing':'yes',
                'P_k_max_h/Mpc': 10, 
                'z_max_pk': 10})

        LCDM.compute()
        h = LCDM.h()

        # redshift z = [0, 10] e scala k = [10^-4; 3]
        nk = dim_k
        nz = dim_z
        zz = np.linspace(0, 10, nz)
        kk = np.logspace(-4, np.log10(3), nk)*h    # k in 1/Mpc

        filename = 'redshift'
        save = os.path.join(path, filename)
        np.save(save, zz)
        filename = 'scale'
        save = os.path.join(path, filename)
        np.save(save, kk)


        # growth factor scale independent D(z)
        Dz = np.array([LCDM.scale_independent_growth_factor(z) for z in zz])
        filename = 'D(z)_m-neu=' + str(round(m_neutrino, 3))
        save = os.path.join(path, filename)
        np.save(save, Dz)


        # calcolo il Power Spectrum Pkz(k,z) per tutti i valori di k e z
        Pkz = np.zeros([nk,nz])
        for k in range(nk) :
                for z in range(nz) :
                        Pkz[k,z] = LCDM.pk_lin(kk[k], zz[z])*(h**3)
        filename = 'P(k,z)_m-neu=' + str(round(m_neutrino, 3))
        save = os.path.join(path, filename)
        np.save(save, Pkz)


        # estraggo il growth factor normalizato dal rapporto Pka(k,z)/Pka(k,0)
        Dkz = np.zeros([nk,nz])
        for k in range(nk) :
                for z in range(nz) :
                        Dkz[k,z] = np.sqrt(Pkz[k,z]/Pkz[k,0])
        filename = 'D(k,z)_m-neu=' + str(round(m_neutrino, 3))
        save = os.path.join(path, filename)
        np.save(save, Dkz)


        # estraggo i mu
        mu = np.zeros([nk,nz])
        for k in range(nk) :
                mu[k,:] = Dkz[k,:] / (Dz/Dz[0])
        filename = 'Mu(k,z)_m-neu=' + str(round(m_neutrino, 3))
        save = os.path.join(path, filename)
        np.save(save, mu)

        return {'redshift': zz, 'scale':kk, 'power_spectrum': Pkz, 'growth_numeric': Dz, 'growth_analitic': Dkz, 'growth_ratio': mu}

        # plotto D(k,z) con variabile z e parametro k, plotto anche D(z)
'''
        plt.plot(zz, Dkz[0,:], color='blue', label='D(' + str(kk[0]) + ',z)', linewidth='0.5')
        plt.plot(zz, Dkz[n-1,:], color='blue', label='D(' + str(kk[n-1]) + ',z)', linewidth='0.5')
        plt.plot(zz, Dz, color='red', label='D(z)', linewidth='0.5')
        plt.axhline(y=1, color='green', linestyle='--')

        plt.xlabel('z')
        plt.ylabel('D(z)')

        plt.legend()
        plt.show()
'''






def Masses(min = 0.06, max = 1, dim = 10, path = 'files') :

        filename = 'neutrino_mass.npy'
        save = os.path.join(path, filename)

        m = np.linspace(min, max, dim)
        np.save(save, m)

        return m


# DataFromCLASS(m_neutrino = 0.06)