import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az

rng = np.random.default_rng(100)

def plot_gre_gpa(idata, gre, gpa):
    beta0 = idata.posterior['beta_0'].mean(dim=('chain', 'draw'))
    beta = idata.posterior['beta'].mean(dim=('chain', 'draw'))

    probabilities = 1 / (1 + np.exp(-(beta0 + gre * beta[0] + gpa * beta[1])))

    hdi = pm.stats.hdi(probabilities, hdi_prob=0.9)
    hdi_probabilities = probabilities[(probabilities >= hdi[0]) & (probabilities <= hdi[1])]

    plt.plot(np.sort(hdi_probabilities))
    plt.title(f'GRE = {gre} GPA = {gpa}')
    plt.show()

def main():
    data = pd.read_csv("Admission.csv")
    scores = data[['GRE', 'GPA']].values
    admission_status = data['Admission'].values
    scores = np.array(scores)

    with pm.Model() as logistic_model:
        beta_0 = pm.Normal('beta_0', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=2, shape=2)

        mu = beta_0 + pm.math.dot(scores, beta)

        theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-mu)))
        bd = pm.Deterministic('bd', -beta_0 / beta[1] - beta[0] / beta[1] * scores[:, 0])
        yl = pm.Bernoulli('yl', p=theta, observed=admission_status)
        idata = pm.sample(2000, random_seed=100, return_inferencedata=True)

    idx = np.argsort(scores[:, 0])
    bd = idata.posterior['bd'].mean(dim=('chain', 'draw'))[idx]
    print(bd.mean())

    plt.scatter(scores[:, 0], scores[:, 1], c=admission_status)
    plt.plot(scores[:, 0][idx], bd, color='k')
    az.plot_hdi(scores[:, 0], idata.posterior['bd'], color='k')
    plt.xlabel('GRE')
    plt.ylabel('GPA')
    plt.title('Admission')
    plt.show()

    plot_gre_gpa(idata, 3.5, 550)
    plot_gre_gpa(idata, 3.2, 500)
    plot_gre_gpa(idata, 4.5, 1000)

if __name__ == "__main__":
    main()
