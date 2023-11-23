import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# Generăm date de exemplu
np.random.seed(42)
data = np.random.normal(5, 2, 100)

# Modelăm distribuția datelor folosind PyMC
with pm.Model() as model:
    # Definim distribuția prior pentru medie
    mean = pm.Normal('mean', mu=0, tau=1.0 / 10 ** 2)  # Tau = 1/sigma^2

    # Definim distribuția likelihood a datelor
    likelihood = pm.Normal('likelihood', mu=mean, tau=1.0 / 2 ** 2, observed=True, value=data)  # Tau = 1/sigma^2

    # Realizăm inferența folosind Metoda Lantului Markov Monte Carlo
    mcmc = pm.MCMC(model)
    mcmc.sample(1000, burn=500)

# Vizualizăm rezultatele
pm.Matplot.plot(mcmc)
plt.show()
