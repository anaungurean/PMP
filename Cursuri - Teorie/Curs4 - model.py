import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# Generăm date de exemplu
np.random.seed(42)
data = np.random.normal(5, 2, 100)

# Modelăm distribuția datelor folosind PyMC
with pm.Model() as model:
    # Definim distribuția prior pentru medie
    mean = pm.Normal('mean', mu=0, sigma=10)

    # Definim distribuția likelihood a datelor
    likelihood = pm.Normal('likelihood', mu=mean, sigma=2, observed=data)

    # Realizăm inferența folosind Metoda Lantului Markov Monte Carlo
    trace = pm.sample(1000, tune=1000)

# Vizualizăm rezultatele
pm.traceplot(trace)
plt.show()
