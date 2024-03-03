# Symbolic regression application to massive neutrinos in cosmology
## Applicazioni della regressione simbolica ai neutrini massivi in cosmologia

_WORK IN PROGRESS_

In this repository it is stored the code I'm writing for my bachelor thesis project. The aim is to find an efficient way to compute the **Growth factor** $D(k,z)$ combining the effects of symbolic regression and neural networks.
In particular, given the Matter Power Density Spectrum:

$$ P(k,z) = \mathcal{P}(k) \, T^2(k) \, D^2(k,z) = \mathcal{P}(k) \, T^2(k) \, \bar{D}^2(z) \, \mu^2(k,z) $$

the goal was to obtain a symbolic espression for $\mu(k,z)$, from:

$$ \sqrt{\dfrac{P(k,z)}{P(k,0)}} \, \dfrac{\bar{D}(0)}{\bar{D}(z)}  = \dfrac{D(k,z)}{D(k,0)} \, \dfrac{\bar{D}(0)}{\bar{D}(z)} = \dfrac{\mu(k,z)}{\mu(k,0)} $$


### Code
All the code is stored in the 'code' folder, that contains some scripts and a few jupyter notebooks for data visualization.
- **sim_mu_.py**: these are Python scripts that compute symbolic regression to find $\mu(k,z)$, varying different cosmological parameters
- **gen_data_.py**: these scripts are called by the 'sim_mu_.py' if the needed datasets are not present iìn the 'data' directory

### Datasets
The datasets are avaliable in the Zenodo release

### Models
The trained PySR models, containing the best expressions found by symbolic regressions are stored in the 'models' directosy. Each model is identified as follows: **mu_parameters(iterations,complexity,populations)**

### Data visualization
In the 'plots' directory there are all the relevant graphics produced during data analisys

### Reproducibility
In order to reproduce the analisys of the project, you should mantain the structure of the repository and add a 'data' folder which contains the datasets from the Zenodo relase