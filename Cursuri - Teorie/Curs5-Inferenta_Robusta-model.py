import pymc as pm
import numpy as np


def robust_inference():
    # Generăm date de exemplu cu outliere
    np.random.seed(42)
    data = np.concatenate([np.random.normal(5, 2, 90), np.random.normal(15, 2, 10)])

    # Modelăm distribuția datelor folosind distribuție t-student (robustă la outliere)
    with pm.Model() as model:
        # Definim distribuția prior pentru medie
        mean = pm.Normal('mean', mu=0, tau=1.0 / (10.0 ** 2))

        # Definim distribuția likelihood a datelor
        likelihood = pm.StudentT('likelihood', nu=3, mu=mean, lam=1.0 / (2.0 ** 2), observed=data)

        # Realizăm inferența folosind Metoda Lantului Markov Monte Carlo
        trace = pm.sample(1000, tune=1000)


if __name__ == '__main__':
    robust_inference()
