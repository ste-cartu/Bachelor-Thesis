import matplotlib.pyplot as plt
import numpy as np
from classy import Class
import os



# funzione che, dati in input alcuni parametri cosmologici, restituisce il power density spectrum P(k,z) o P(k,a) e il growth factor D(z) o D(a) e D(k,z) o D(k,a)
# input: massa del neutrino, A_s, n_s, omega_b, omega_c, h
def DataFromCLASS(dim_k = 100,
                  dim_z = 20,
                  m_neutrino = 0.06,
                  ln_a_s = 3.04478383,
                  n_s = 0.9660499,
                  omega_b = 0.02238280,
                  omega_c = 0.1201075,
                  h = 0.67810,
                  path = '../files'
                  ) :
        
        # creo il modello CLASS, imposto i suoi parametri e lo eseguo
        LCDM = Class()

        LCDM.set({'omega_b': omega_b, 
                'omega_cdm': omega_c, 
                'h': h,
                'ln_A_s_1e10': ln_a_s, 
                'n_s': n_s, 
                'tau_reio': 0.05430842,
                'z_max_pk': 10,
                'N_ncdm': 1,
                'm_ncdm': m_neutrino
                })
        LCDM.set({'output': 'tCl,pCl,lCl,mPk',
                'lensing':'yes',
                'P_k_max_h/Mpc': 10, 
                'z_max_pk': 10
                })

        LCDM.compute()
        h = LCDM.h()

        # redshift z = [0, 10] e scala k = [10^-4; 3]
        nk = dim_k
        nz = dim_z
        zz = np.linspace(0, 5, nz)
        kk = np.logspace(-4, np.log10(3), nk)*h    # k in h/Mpc

        filename = 'redshift_[' + str(nz) + ']'
        save = os.path.join(path, filename)
        np.save(save, zz)
        filename = 'scale_[' + str(nk) + ']'
        save = os.path.join(path, filename)
        np.save(save, kk)


        # growth factor scale independent D(z)
        dz = np.array([LCDM.scale_independent_growth_factor(z) for z in zz])

        # calcolo il Power Spectrum pkz(k,z) per tutti i valori di k e z
        pkz = np.zeros([nk,nz])
        for k in range(nk) :
                for z in range(nz) :
                        pkz[k,z] = LCDM.pk_lin(kk[k], zz[z])*(h**3)

        # estraggo il growth factor normalizato dal rapporto Pka(k,z)/Pka(k,0)
        dkz = np.zeros([nk,nz])
        for k in range(nk) :
                for z in range(nz) :
                        dkz[k,z] = np.sqrt(pkz[k,z]/pkz[k,0])

        # estraggo i mu
        mu = np.zeros([nk,nz])
        for k in range(nk) :
                mu[k,:] = dkz[k,:] / (dz/dz[0])
        filename = 'Mu(k,z)_[' +str(nk)+ ',' +str(nz)+ ']_m=' +str(round(m_neutrino, 3))+ '_Ob=' +str(round(omega_b, 3))+ '_Oc=' +str(round(omega_c, 3))+ '_h=' +str(round(h, 2))
        # save = os.path.join(path, filename)
        # np.save(save, mu)

        return {'redshift': zz, 'scale':kk, 'power_spectrum': pkz, 'growth_numeric': dz, 'growth_analitic': dkz, 'growth_ratio': mu}




# input: Omega_nu, Omega_b, Omega_c, h
def DataFromCLASS2(dim_k = 100,
                  dim_z = 20,
                  Omega_nu = 0.06/((0.67810**2) * 93.14),
                  Omega_b = 0.02238280/(0.67810**2),            # Omega = omega/(h^2)
                  Omega_c = 0.1201075/(0.67810**2),
                  h = 0.67810,
                  path = '../files'
                  ) :
        
        # creo il modello CLASS, imposto i suoi parametri e lo eseguo
        LCDM = Class()

        LCDM.set({'Omega_b': Omega_b, 
                'Omega_cdm': Omega_c, 
                'h': h,
                'tau_reio': 0.05430842,
                'z_max_pk': 10,
                'N_ncdm': 1,
                'Omega_ncdm': Omega_nu,
                })
        LCDM.set({'output': 'tCl,pCl,lCl,mPk',
                'lensing':'yes',
                'P_k_max_h/Mpc': 10, 
                'z_max_pk': 10
                })

        LCDM.compute()
        h = LCDM.h()

        # redshift z = [0, 10] e scala k = [10^-4; 3]
        nk = dim_k
        nz = dim_z
        zz = np.linspace(0, 5, nz)
        kk = np.logspace(-4, np.log10(3), nk)*h    # k in h/Mpc

        filename = 'redshift_[' + str(nz) + ']'
        save = os.path.join(path, filename)
        np.save(save, zz)
        filename = 'scale_[' + str(nk) + ']'
        save = os.path.join(path, filename)
        np.save(save, kk)


        # growth factor scale independent D(z)
        dz = np.array([LCDM.scale_independent_growth_factor(z) for z in zz])

        # calcolo il Power Spectrum pkz(k,z) per tutti i valori di k e z
        pkz = np.zeros([nk,nz])
        for k in range(nk) :
                for z in range(nz) :
                        pkz[k,z] = LCDM.pk_lin(kk[k], zz[z])*(h**3)

        # estraggo il growth factor normalizato dal rapporto Pka(k,z)/Pka(k,0)
        dkz = np.zeros([nk,nz])
        for k in range(nk) :
                for z in range(nz) :
                        dkz[k,z] = np.sqrt(pkz[k,z]/pkz[k,0])

        # estraggo i mu
        mu = np.zeros([nk,nz])
        for k in range(nk) :
                mu[k,:] = dkz[k,:] / (dz/dz[0])
        filename = 'Mu(k,z)_[' +str(nk)+ ',' +str(nz)+ ']_Onu=' +str(round(Omega_nu, 3))+ '_OB=' +str(round(Omega_b, 3))+ '_OC=' +str(round(Omega_c, 3))+ '_h=' +str(round(h, 2))
        # save = os.path.join(path, filename)
        # np.save(save, mu)

        return {'redshift': zz, 'scale':kk, 'power_spectrum': pkz, 'growth_numeric': dz, 'growth_analitic': dkz, 'growth_ratio': mu}




# input: Omega_nu, Omega_b, Omega_m, h
def DataFromCLASS3(dim_k = 100,
                  dim_z = 20,
                  Omega_nu = 0.06/((0.67810**2) * 93.14),
                  Omega_b = 0.02238280/(0.67810**2),                                            # Omega = omega/(h^2)
                  Omega_m = ((0.06/93.14) + 0.02238280 + 0.1201075)/(0.67810**2),               # Omega_m = Omega_nu + Omega_b + Omega_c
                  h = 0.67810,
                  path = '../files'
                  ) :
        
        # creo il modello CLASS, imposto i suoi parametri e lo eseguo
        LCDM = Class()

        LCDM.set({'Omega_b': Omega_b, 
                'Omega_m': Omega_m, 
                'h': h,
                'tau_reio': 0.05430842,
                'z_max_pk': 10,
                'N_ncdm': 1,
                'Omega_ncdm': Omega_nu,
                })
        LCDM.set({'output': 'tCl,pCl,lCl,mPk',
                'lensing':'yes',
                'P_k_max_h/Mpc': 10, 
                'z_max_pk': 10
                })

        LCDM.compute()
        h = LCDM.h()

        # redshift z = [0, 10] e scala k = [10^-4; 3]
        nk = dim_k
        nz = dim_z
        zz = np.linspace(0, 5, nz)
        kk = np.logspace(-4, np.log10(3), nk)*h    # k in h/Mpc

        filename = 'redshift_[' + str(nz) + ']'
        save = os.path.join(path, filename)
        np.save(save, zz)
        filename = 'scale_[' + str(nk) + ']'
        save = os.path.join(path, filename)
        np.save(save, kk)


        # growth factor scale independent D(z)
        dz = np.array([LCDM.scale_independent_growth_factor(z) for z in zz])

        # calcolo il Power Spectrum pkz(k,z) per tutti i valori di k e z
        pkz = np.zeros([nk,nz])
        for k in range(nk) :
                for z in range(nz) :
                        pkz[k,z] = LCDM.pk_lin(kk[k], zz[z])*(h**3)

        # estraggo il growth factor normalizato dal rapporto Pka(k,z)/Pka(k,0)
        dkz = np.zeros([nk,nz])
        for k in range(nk) :
                for z in range(nz) :
                        dkz[k,z] = np.sqrt(pkz[k,z]/pkz[k,0])

        # estraggo i mu
        mu = np.zeros([nk,nz])
        for k in range(nk) :
                mu[k,:] = dkz[k,:] / (dz/dz[0])
        filename = 'Mu(k,z)_[' +str(nk)+ ',' +str(nz)+ ']_Onu=' +str(round(Omega_nu, 3))+ '_OB=' +str(round(Omega_b, 3))+ '_OM=' +str(round(Omega_m, 3))+ '_h=' +str(round(h, 2))
        # save = os.path.join(path, filename)
        # np.save(save, mu)

        return {'redshift': zz, 'scale':kk, 'power_spectrum': pkz, 'growth_numeric': dz, 'growth_analitic': dkz, 'growth_ratio': mu}




def Masses(min = 0.06, max = 1, dim = 10, path = '../files') :

        filename = 'neutrino_mass_[' + str(dim) + '].npy'
        save = os.path.join(path, filename)

        m = np.linspace(min, max, dim)
        np.save(save, m)

        return m



'''
def Scale(min = 1e-4, max = 3, dim = 100, path = '../files') :

        filename = 'scale_[' + str(dim) + '].npy'
        save = os.path.join(path, filename)

        k = np.logspace(np.log10(min), np.log10(max), dim)*0.67810
        np.save(save, k)

        return k
'''



def Redshift(min = 0, max = 5, dim = 20, path = '../files') :

        filename = 'redshift_[' + str(dim) + '].npy'
        save = os.path.join(path, filename)

        z = np.linspace(min, max, dim)
        np.save(save, z)

        return z




def A_s(min = 2.5, max = 3.5, dim = 20, path = '../files') :

        filename = 'a_s_[' + str(dim) + '].npy'
        save = os.path.join(path, filename)

        asas = np.linspace(min, max, dim)
        np.save(save, asas)

        return asas




def N_s(min = 0.9, max = 1, dim = 20, path = '../files') :

        filename = 'n_s_[' + str(dim) + '].npy'
        save = os.path.join(path, filename)

        nsns = np.linspace(min, max, dim)
        np.save(save, nsns)

        return nsns




def Omega_b(min = 0.020, max = 0.024, dim = 20, path = '../files') :

        filename = 'omega_b_[' + str(dim) + '].npy'
        save = os.path.join(path, filename)

        ob = np.linspace(min, max, dim)
        np.save(save, ob)

        return ob




def Omega_c(min = 0.10, max = 0.14, dim = 20, path = '../files') :

        filename = 'omega_c_[' + str(dim) + '].npy'
        save = os.path.join(path, filename)

        oc = np.linspace(min, max, dim)
        np.save(save, oc)

        return oc




def H(min = 0.6, max = 0.8, dim = 20, path = '../files') :

        filename = 'h_[' + str(dim) + '].npy'
        save = os.path.join(path, filename)

        h = np.linspace(min, max, dim)
        np.save(save, h)

        return h

