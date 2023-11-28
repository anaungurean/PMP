import pymc3 as pm
import numpy as np
import pandas as pd


def main():
    data = pd.read_csv('Prices.csv')
    model_regression = pm.Model()
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)
        sigma = pm.HalfCauchy('sigma', 5)
        mu = pm.Deterministic('mu', alpha + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive']))
        y = pm.Normal('y', mu=mu, sd=sigma, observed=data['Price'])
        trace = pm.sample(10, tune=10)

    hdi_beta1 = pm.stats.hdi(trace['beta1'], hdi_prob=0.95)
    hdi_beta2 = pm.stats.hdi(trace['beta2'], hdi_prob=0.95)

    print(f"Estimarea HDI pentru beta1: {hdi_beta1}")
    print(f"Estimarea HDI pentru beta2: {hdi_beta2}")


if __name__ == "__main__":
    main()
