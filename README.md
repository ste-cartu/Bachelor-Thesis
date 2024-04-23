# Symbolic regression application to massive neutrinos in cosmology
## Applicazioni della regressione simbolica ai neutrini massivi in cosmologia

_WORK IN PROGRESS_

In this repository it is stored the code I'm writing for my bachelor thesis project. The aim is to find an efficient way to compute the **Growth Factor** $D(k,z)$ combining the effects of symbolic regression and neural networks.
In particular, given the Matter Power Density Spectrum:

$$ P(k,z) = \mathcal{P}(k) \, T^2(k) \, D^2(k,z) = \mathcal{P}(k) \, T^2(k) \, \bar{D}^2(z) \, \mu^2(k,z) \ . $$

The goal was to obtain a symbolic espression for $\mu(k,z)$ using **Symbolic Regression**, from:

$$ \sqrt{\dfrac{P(k,z)}{P(k,0)}} \, \dfrac{\bar{D}(0)}{\bar{D}(z)}  = \dfrac{D(k,z)}{D(k,0)} \, \dfrac{\bar{D}(0)}{\bar{D}(z)} = \dfrac{\mu(k,z)}{\mu(k,0)} \ , $$

and to use this expression to train a **Neural Network** which will rebuild the complete Power Spectrum:

$$ NN(k,z) \equiv \frac{P_{\texttt{CLASS}}(k,z)}{P_\text{prim}(k) \, T_\text{SR}^{\, 2}(k) \, \mu_\text{SR}^{\, 2}(k,z)} \ . $$

The neuro-symbolic emulator built in this way is also compared to a simpler emulator, composed only by a Neural Network.


### Code
All the code is stored in the 'code' folder, that contains some scripts and a few jupyter notebooks for data visualization.
- **sim_mu_*.py**: these are Python scripts that compute symbolic regression to find $\mu(k,z)$, varying different cosmological parameters.
- **emul_*.jl**: these files create the neural networks that rebuild the Power Spectrum.
- **gen_data_*.py**: these scripts are used to generate the datasets needed by the emulators if they are not yet in the 'files' directory.

### Datasets
The datasets are avaliable in the Zenodo release.

### Models
The trained PySR models, containing the best expressions found by symbolic regressions are stored in the 'models' directosy. Each model is identified as follows: **mu_parameters(iterations,complexity,populations)**.
The 'models' directory also contains the trained parameters of the neural networks, in files called **nn+sr_cycles-epochs_divider.jld2** for the neuro-symbolic emulator and **nn_cycles-epochs_divider.jld2** for the neural emulators.

### Data visualization
In the 'plots' directory there are all the relevant graphics produced during data analisys.

### Reproducibility
In order to reproduce the analisys of the project, you should mantain the structure of the repository and add a 'files' folder which contains the datasets from the Zenodo relase.