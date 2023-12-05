import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv("Admission.csv")

    gre_scores = data["GRE"].values
    gpa_scores = data["GPA"].values
    admission_status = data['Admission'].values

    with pm.Model() as logistic_model:
        beta_0 = pm.Normal('beta_0', mu=0, sigma=10)
        beta_1 = pm.Normal('beta_1', mu=0, sigma=10)
        beta_2 = pm.Normal('beta_2', mu=0, sigma=10)

        pi = pm.math.invlogit(beta_0 + beta_1 * gre_scores + beta_2 * gpa_scores)
        admission_likelihood = pm.Bernoulli('admission_likelihood', p=pi, observed=admission_status)

    with logistic_model:
        trace = pm.sample(1000, tune=1000, cores=2)

    beta_0_samples = trace['beta_0']
    beta_1_samples = trace['beta_1']
    beta_2_samples = trace['beta_2']

    decision_boundary = np.median(pm.math.invlogit(beta_0_samples + beta_1_samples * gre_scores + beta_2_samples * gpa_scores))


if "__name__" == "__main__":
    main()
