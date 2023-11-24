import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def main():
    np.random.seed(42) # pentru a genera aceleași numere aleatoare
    # Generăm o distribuție normală a timpilor de așteptare, cu o medie de 10 și o deviație standard de 2, pentru 100 de eșantioane.
    timp_mediu_asteptare = np.random.normal(loc=10, scale=2, size=100)

    with pm.Model() as model:
        miu = pm.Normal('miu', mu=10) # pentru a defini o distribuție normală în cadrul unui model Bayesian
        #mu este media distribuției normale
        #sigma este deviația standard a distribuției normale
        sigma = pm.HalfNormal('sigma', sigma=2) #Pentru a defini o variabilă aleatoare distribuită normal (sau gaussian) în cadrul unui model probabilistic, dar limitată la valorile pozitive
        likelihood = pm.Normal('likelihood', mu=miu, sigma=sigma, observed=timp_mediu_asteptare) #probabilitatea de a observa datele observate, dată o anumită configurație a parametrilor modelului
        trace = pm.sample(100, tune=100) #pentru a efectua eșantionarea din distribuția posterioară a unui model Bayesian

    az.plot_posterior(trace, var_names=['miu'])
    #pentru a vizualiza distribuția posterioară a parametrilor modelului
    plt.title('Distribuția a posteriori pentru miu')
    plt.xlabel('Miu')
    plt.ylabel('Densitatea de probabilitate')
    plt.show()


if __name__ == "__main__":
    main()