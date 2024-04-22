import numpy as np
from classy import Class
import functions as f
from scipy.stats import qmc
import scipy.integrate
import warnings
from colossus.cosmology import cosmology


# dimensions
nk = 100
nz = 20
nc = 2000
ncv = 1000




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Pedro code for power spectrum zero-baryon and transfer function Eisensten - Hu
# I modified slightly the original script: the function 'get_pk' now is 'get_pk_tk' and returns a list with both the power 
#   spectrum and the transfer function

def pk_EisensteinHu_zb(k, sigma8, Om, Ob, h, ns, use_colossus=False):
    """
    Compute the Eisentein & Hu 1998 zero-baryon approximation to P(k) at z=0
    
    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :use_colossus (bool, default=False): Whether to use the external package colossus
            to compute this term
        
    Returns:
        :pk_eh (np.ndarray): The Eisenstein & Hu 1998 zero-baryon P(k) [(Mpc/h)^3]
    """

    if use_colossus:
        cosmo_params = {
            'flat':True,
            'sigma8':sigma8,
            'Om0':Om,
            'Ob0':Ob,
            'H0':h*100.,
            'ns':ns,
        }
        cosmo = cosmology.setCosmology('myCosmo', **cosmo_params)
        pk_eh = cosmo.matterPowerSpectrum(k, z = 0.0, model='eisenstein98_zb')
    else:
        ombom0 = Ob / Om
        om0h2 = Om * h**2
        ombh2 = Ob * h**2
        theta2p7 = 2.7255 / 2.7 # Assuming Tcmb0 = 2.7255 Kelvin

        def get_pk_tk(kk, Anorm):
        
            # Compute scale factor s, alphaGamma, and effective shape Gamma
            s = 44.5 * np.log(9.83 / om0h2) / np.sqrt(1.0 + 10.0 * ombh2**0.75)
            alphaGamma = 1.0 - 0.328 * np.log(431.0 * om0h2) * ombom0 + \
            0.38 * np.log(22.3 * om0h2) * ombom0**2
            Gamma = Om * h * (alphaGamma + (1.0 - alphaGamma) / \
                (1.0 + (0.43 * kk * h * s)**4))
            
            # Compute q, C0, L0, and tk_eh
            q = kk * theta2p7**2 / Gamma
            C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
            L0 = np.log(2.0 * np.exp(1.0) + 1.8 * q)
            tk_eh = L0 / (L0 + C0 * q**2)

            # Calculate Pk with unit amplitude
            pk_eh = Anorm * tk_eh**2 * kk**ns
            return [pk_eh, tk_eh]
        
        # Define integration bounds and number of sub-intervals
        b0 = np.log(1e-7) # ln(k_min)
        b1 = np.log(1e5)  # ln(k_max)
        n = 1000      # Number of sub-intervals (make sure it's even for Simpson's Rule)

        # Find normalisation
        R = 8.0
        kk = np.exp(np.linspace(b0, b1, n))
        x = kk * R
        W = np.zeros(x.shape)
        m = x < 1.e-3
        W[m] = 1.0
        W[~m] =3.0 / x[~m]**3 * (np.sin(x[~m]) - x[~m] * np.cos(x[~m]))
        y = get_pk_tk(kk, 1.0)[0] * W**2 * kk**3
        sigma2 = scipy.integrate.simpson(y, x=np.log(x))
        
        sigmaExact = np.sqrt(sigma2 / (2.0 * np.pi**2))
        Anorm = (sigma8 / sigmaExact)**2
        
        pk_tk_eh = get_pk_tk(k, Anorm)
        
    return pk_tk_eh




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Pedro code for baryon power spectrum

def pk_EisensteinHu_b(k, sigma8, Om, Ob, h, ns):
    """
    Compute the Eisentein & Hu 1998 baryon approximation to P(k) at z=0
    
    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        
    Returns:
        :pk_eh (np.ndarray): The Eisenstein & Hu 1998 baryon P(k) [(Mpc/h)^3]
    """

    cosmo_params = {
        'flat':True,
        'sigma8':sigma8,
        'Om0':Om,
        'Ob0':Ob,
        'H0':h*100.,
        'ns':ns,
    }
    cosmo = cosmology.setCosmology('myCosmo', **cosmo_params)
    pk_eh = cosmo.matterPowerSpectrum(k, z = 0.0, model='eisenstein98')
        
    return pk_eh




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Pedro code for logF

def logF_fiducial(k, sigma8, Om, Ob, h, ns, extrapolate=False):
    """
    Compute the emulated logarithm of the ratio between the true linear
    power spectrum and the Eisenstein & Hu 1998 fit. Here we use the fiducial exprssion
    given in Bartlett et al. 2023.
    
    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :extrapolate (bool, default=False): If True, then extrapolates the Bartlett
            et al. 2023 fit outside range tested in paper. Otherwise, uses E&H with
            baryons for this regime
        
    Returns:
        :logF (np.ndarray): The logarithm of the ratio between the linear P(k) and the
            Eisenstein & Hu 1998 zero-baryon fit
    """
    
    b = [0.05448654, 0.00379, 0.0396711937097927, 0.127733431568858, 1.35,
        4.053543862744234, 0.0008084539054750851, 1.8852431049189666,
        0.11418372931475675, 3.798, 14.909, 5.56, 15.8274343004709, 0.0230755621512691,
        0.86531976, 0.8425442636372944, 4.553956000000005, 5.116999999999995,
        70.0234239999998, 0.01107, 5.35, 6.421, 134.309, 5.324, 21.532,
        4.741999999999985, 16.68722499999999, 3.078, 16.987, 0.05881491,
        0.0006864690561825617, 195.498, 0.0038454457516892, 0.276696018851544,
        7.385, 12.3960625361899, 0.0134114370723638]
        
    line1 = b[0] * h - b[1]
    
    line2 = (
        ((Ob * b[2]) / np.sqrt(h ** 2 + b[3])) ** (b[4] * Om) *
        (
            (b[5] * k - Ob) / np.sqrt(b[6] + (Ob - b[7] * k) ** 2)
            * b[8] * (b[9] * k) ** (-b[10] * k) * np.cos(Om * b[11]
            - (b[12] * k) / np.sqrt(b[13] + Ob ** 2))
            - b[14] * ((b[15] * k) / np.sqrt(1 + b[16] * k ** 2) - Om)
            * np.cos(b[17] * h / np.sqrt(1 + b[18] * k ** 2))
        )
    )
    
    line3 = (
        b[19] *  (b[20] * Om + b[21] * h - np.log(b[22] * k)
        + (b[23] * k) ** (- b[24] * k)) * np.cos(b[25] / np.sqrt(1 + b[26] * k ** 2))
    )
    
    line4 = (
        (b[27] * k) ** (-b[28] * k) * (b[29] * k - (b[30] * np.log(b[31] * k))
        / np.sqrt(b[32] + (Om - b[33] * h) ** 2))
        * np.cos(Om * b[34] - (b[35] * k) / np.sqrt(Ob ** 2 + b[36]))
    )
    
    logF = line1 + line2 + line3 + line4
    
    # Use Bartlett et al. 2023 P(k) only in tested regime
    m = ~((k >= 9.e-3) & (k <= 9))
    if (not extrapolate) and m.sum() > 0:
        warnings.warn("Not using Bartlett et al. formula outside tested regime")
        logF[m] = np.log(
            pk_EisensteinHu_b(k[m], sigma8, Om, Ob, h, ns) /
            pk_EisensteinHu_zb(k[m], sigma8, Om, Ob, h, ns)[0]
        )

    return logF




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# transfer function from Pedro code (transfer(Eisensten e Hu)**2) * exp(log(F_fiducial))

def Transfer2(kk, On, Ob, Oc, h) :

    Om = On + Ob + Oc
    sigma8 = 0.824398
    ns = 0.9660499

    # transfer function Eisensten - Hu
    tf_eh2 = pk_EisensteinHu_zb(kk, sigma8, Om, Ob, h, ns)[1] ** 2
    F = np.exp(logF_fiducial(kk, sigma8, Om, Ob, h, ns))

    return tf_eh2*F




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# funzione che calcola il power spectrum primordiale

def Prim(kk) :
    As = 2.100549e-09
    ns = 0.9660499
    k_piv = 0.05

    return ((2*np.pi*As) / (kk**3)) * ((kk/k_piv)**(ns-1))




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# funzione che calcola mu secondo il modello 'mu_VBCh(4000,40,120)'

def Mu_VBCh (k, z, On, Ob, Oc, h) :

    # modello 'mu_VBCh(4000,40,120)'
    mu = np.e ** ( (On*z) / ( -1.13 ** (z + k * (On ** (-0.532/h))) - np.log(Ob+Oc) - 1.08 ) )

    return mu




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# parametri cosmologici: training dataset di input

print('calcolo l\'input del training dataset:')
# ipercubo con nc combinazioni
sampler = qmc.LatinHypercube(d=4)
sample = sampler.random(n=nc)

# ipercubo: [Omega_nu,                  Omega_b,    Omega_c,   h]
inf =       [0.06/(93.14*(0.8**2)),     0.04,       0.23,      0.6]
sup =       [1/(93.14*(0.6**2)),        0.06,       0.29,      0.8]
input = qmc.scale(sample, inf, sup).T
np.save('../files/train_in_emul_' + str(nc), input)
print('input training salvato\n')




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# parametri cosmologici: validation dataset di input

print('calcolo l\'input del validation dataset:')
# ipercubo con nc combinazioni
sampler_v = qmc.LatinHypercube(d=4)
sample_v = sampler_v.random(n=ncv)

# ipercubo: [Omega_nu,                  Omega_b,    Omega_c,   h]
inf =       [0.06/(93.14*(0.8**2)),     0.04,       0.23,      0.6]
sup =       [1/(93.14*(0.6**2)),        0.06,       0.29,      0.8]
input_v = qmc.scale(sample_v, inf, sup).T
np.save('../files/val_in_emul_' + str(ncv), input_v)
print('input validation salvato\n')

#input = np.load('../files/train_in_emul_' + str(nc) + '.npy')
#input_v = np.load('../files/val_in_emul_' + str(ncv) + '.npy')




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# calcolo l'output del training dataset

print('calcolo l\'output del training dataset:')
output = np.zeros([nc,nk,nz])
pkz = np.zeros([nc,nk,nz])
mu2 = np.zeros([nc,nk,nz])
prim = np.zeros([nc,nk])
tf2 = np.zeros([nc,nk])

for c in range(nc) :
    LCDM = Class()

    LCDM.set({
        'Omega_ncdm': input[0,c],
        'Omega_b': input[1,c],
        'Omega_cdm': input[2,c],
        'h': input[3,c],
        'tau_reio': 0.05430842,
        'z_max_pk': 10,
        'N_ncdm': 1
        })
    LCDM.set({
        'output': 'tCl,pCl,lCl,mPk',
        'lensing':'yes',
        'P_k_max_h/Mpc': 10, 
        'z_max_pk': 10
        })
    LCDM.compute()

    h = LCDM.h()
    zz = np.linspace(0, 5, nz)
    kk = np.logspace(-4, np.log10(3), nk)
    kkh = kk*h

    prim[c,:] = Prim(kk)
    tf2[c,:] = Transfer2(kkh, input[0,c], input[1,c], input[2,c], h)

    for k in range(nk) :
        for z in range(nz) :
            pkz[c,k,z] = LCDM.pk_lin(kkh[k], zz[z])*(h**3)
            mu2[c,k,z] = (Mu_VBCh(kkh[k], zz[z], input[0,c], input[1,c], input[2,c], input[3,c])) **2

            output[c,k,z] = pkz[c,k,z] / (prim[c,k] * tf2[c,k] * mu2[c,k,z])

    print(c+1, '/', nc)

output = np.reshape(output, [nc,nk*nz])

np.save('../files/train_out_emul_' + str(nc), output)
np.save('../files/train_pkz_emul_' + str(nc), pkz)
np.save('../files/train_mu2_emul_' + str(nc), mu2)
np.save('../files/train_prim_emul_' + str(nc), prim)
np.save('../files/train_tf2_emul_' + str(nc), tf2)

print('output training salvato\n')




# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# calcolo l'output del validation dataset

print('calcolo l\'output del validation dataset:')
output_v = np.zeros([ncv,nk,nz])
pkz_v = np.zeros([ncv,nk,nz])
mu2_v = np.zeros([ncv,nk,nz])
prim_v = np.zeros([ncv,nk])
tf2_v = np.zeros([ncv,nk])

for c in range(ncv) :
    LCDM = Class()

    LCDM.set({
        'Omega_ncdm': input_v[0,c],
        'Omega_b': input_v[1,c],
        'Omega_cdm': input_v[2,c],
        'h': input_v[3,c],
        'tau_reio': 0.05430842,
        'z_max_pk': 10,
        'N_ncdm': 1
        })
    LCDM.set({
        'output': 'tCl,pCl,lCl,mPk',
        'lensing':'yes',
        'P_k_max_h/Mpc': 10, 
        'z_max_pk': 10
        })
    LCDM.compute()

    h = LCDM.h()
    zz = np.linspace(0, 5, nz)
    kk = np.logspace(-4, np.log10(3), nk)
    kkh = kk*h

    prim_v[c,:] = Prim(kk)
    tf2_v[c,:] = Transfer2(kkh, input_v[0,c], input_v[1,c], input_v[2,c], h)

    for k in range(nk) :
        for z in range(nz) :
            pkz_v[c,k,z] = LCDM.pk_lin(kkh[k], zz[z])*(h**3)
            mu2_v[c,k,z] = (Mu_VBCh(kkh[k], zz[z], input_v[0,c], input_v[1,c], input_v[2,c], input_v[3,c])) **2

            output_v[c,k,z] = pkz_v[c,k,z] / (prim_v[c,k] * tf2_v[c,k] * mu2_v[c,k,z])

    print(c+1, '/', ncv)

output_v = np.reshape(output_v, [ncv,nk*nz])

np.save('../files/val_out_emul_' + str(ncv), output_v)
np.save('../files/val_pkz_emul_' + str(ncv), pkz_v)
np.save('../files/val_mu2_emul_' + str(ncv), mu2_v)
np.save('../files/val_prim_emul_' + str(ncv), prim_v)
np.save('../files/val_tf2_emul_' + str(ncv), tf2_v)

print('output validation salvato\n')
