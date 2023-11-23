import pymc as pm
import numpy as np


def gaussian_inference():
    # Generăm date de exemplu
    np.random.seed(42)
    data = np.random.normal(5, 2, 100)

    # Modelăm distribuția datelor folosind PyMC
    with pm.Model() as model:
        # Definim distribuția prior pentru medie
        mean = pm.Normal('mean', mu=0, tau=1.0 / (10.0 ** 2))

        # Definim distribuția likelihood a datelor
        likelihood = pm.Normal('likelihood', mu=mean, tau=1.0 / (2.0 ** 2), observed=data)

        # Realizăm inferența folosind Metoda Lantului Markov Monte Carlo
        trace = pm.sample(1000, tune=1000)
        print(trace)
  

if __name__ == '__main__':
    gaussian_inference()
