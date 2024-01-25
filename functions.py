import matplotlib.pyplot as plt
import numpy as np
from classy import Class



# funzione che, data in input la massa del neutrino, restituisce il power density spectrum P(k,a) e il growth factor D(a) e D(k,a)

def DataFromClass(m_neutrino = 0.06) :
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
            'z_max_pk': 1})

    LCDM.compute()



    # dal modello estraggo il background
    bg = LCDM.get_background()
    # h = LCDM.h()

    # redshift z = [0, 10] e conversione in fattore di scala a
    zz = bg['z'][37000:]     # dell'array di redshift seleziono solo la parte compresa tra z=0 e z=10.2
    aa = 1./(zz+1)
    n = len(zz)

    # print(aa[0])
    # print(aa[-1])
    # print(n)


    # growth factor scale independent D(a)
    Da = bg['gr.fac. D'][37000:]
    np.save('D(a)_m-neu=' + str(m_neutrino), Da)


    # definisco il range di scale k = [10^-4; 3] con n punti e poi calcolo il growth factor scale dependent D(k,a) normalizzato su D(k,1)
    kk = np.logspace(-4, np.log10(3), n)    # k in h/Mpc


    # pka = [[0 for i in range(n)] for j in range(n)]
    Pka = np.zeros([n,n])

    # calcolo il Power Spectrum Pka(k,a) per tutti i valori di k e a
    for k in range(n) :
        for a in range(n) :
    #       pka[k][a] = LCDM.pk(kk[k], aa[a])
            Pka[k,a] = LCDM.pk(kk[k], aa[a])

    # Pka = np.array(pka)
    np.save('P(k,a)_m-neu=' + str(m_neutrino), Pka)

    # se z=0 => a=1
    # estraggo il growth factor normalizato dal rapporto Pka(k,a)/Pka(k,1)
    # d = [[0 for i in range(n)] for j in range(n)]
    Dka = np.zeros([n,n])

    for k in range(n) :
        for a in range(n) :
    #       d[k][a] = np.sqrt(pka[k][a]/pka[k][-1])
            Dka[k,a] = np.sqrt(Pka[k,a]/Pka[k,-1])
    

    #Dka = np.array(d)
    np.save('D(k,a)_m-neu=' + str(m_neutrino), Dka)


    # plotto D(k,a) con variabile a e parametro k, plotto anche D(a)

    # plt.plot(aa, Dka[0,:], color='blue', label='D(' + str(kk[0]) + ',a)') # , linewidth='0.5')
    # plt.plot(aa, Dka[2976,:], color='blue', label='D(' + str(kk[2976]) + ',a)') # , linewidth='0.5')
    # plt.plot(aa, Da, color='red', label='D(a)')
    # plt.axhline(y=1, color='green', linestyle='--')

    # plt.xlabel('a')
    # plt.ylabel('D')

    # plt.legend()
    # plt.show()
