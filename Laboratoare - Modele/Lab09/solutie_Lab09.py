import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

if __name__ == "__main__":
    admissions = pd.read_csv("Laboratoare - Modele\Lab09\Admission.csv")

    y = admissions["Admission"]
    print(len(y[y == 1]), len(y[y == 0]))  # date nebalansate
    Index = np.random.choice(
        np.flatnonzero(y == 0), size=len(y[y == 0]) - len(y[y == 1]), replace=False
    )  # pentru a balansa datele, alegem la intamplare indici pentru a fi stersi
    admissions = admissions.drop(labels=Index)
    y = admissions["Admission"]
    x_GRE = admissions["GRE"].values
    x_GPA = admissions["GPA"].values
    x_GRE_mean = x_GRE.mean()
    x_GRE_std = x_GRE.std()
    x_GPA_mean = x_GPA.mean()
    x_GPA_std = x_GPA.std()
    # standardizam datele:
    x_GRE = (x_GRE - x_GRE_mean) / x_GRE_std
    x_GPA = (x_GPA - x_GPA_mean) / x_GPA_std
    X = np.column_stack((x_GRE, x_GPA))

    # Exercitiul 1
    with pm.Model() as adm_model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=2)
        X_shared = pm.MutableData("x_shared", X)  # pentru pct. 3 si 4
        mu = pm.Deterministic("μ", alpha + pm.math.dot(X_shared, beta))
        theta = pm.Deterministic("theta", pm.math.sigmoid(mu))
        bd = pm.Deterministic("bd", -alpha / beta[1] - beta[0] / beta[1] * x_GRE)
        y_pred = pm.Bernoulli("y_pred", p=theta, observed=y)
        idata = pm.sample(2000, return_inferencedata=True)

    # Exercitiul 2
    idx = np.argsort(x_GRE)
    bd = idata.posterior["bd"].mean(("chain", "draw"))[idx]
    plt.scatter(x_GRE, x_GPA, c=[f"C{x}" for x in y])
    plt.xlabel("GRE")
    plt.ylabel("GPA")
    plt.show()

    idx = np.argsort(x_GRE)
    bd = idata.posterior["bd"].mean(("chain", "draw"))[idx]
    plt.scatter(x_GRE, x_GPA, c=[f"C{x}" for x in y])
    plt.plot(x_GRE[idx], bd, color="k")
    az.plot_hdi(x_GRE, idata.posterior["bd"], color="k")
    plt.xlabel("GRE")
    plt.ylabel("GPA")
    plt.show()

    # Exercitiul 3 -- var 1
    obs_std1 = [(550 - x_GRE_mean) / x_GRE_std, (3.5 - x_GPA_mean) / x_GPA_std]
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    posterior_g = idata.posterior.stack(samples={"chain", "draw"})
    mu = (
        posterior_g["alpha"]
        + posterior_g["beta"][0] * obs_std1[0]
        + posterior_g["beta"][1] * obs_std1[1]
    )
    theta = sigmoid(mu)
    az.plot_posterior(theta.values, hdi_prob=0.9)
    plt.show()

    # Exercitiul 3 -- var 2
    pm.set_data({"x_shared": [obs_std1]}, model=adm_model)
    ppc = pm.sample_posterior_predictive(idata, model=adm_model, var_names=["theta"])
    y_ppc = ppc.posterior_predictive["theta"].stack(sample=("chain", "draw")).values
    az.plot_posterior(y_ppc, hdi_prob=0.9)
    plt.show()

    # Exercitiul 4
    obs_std2 = [(500 - x_GRE_mean) / x_GRE_std, (3.2 - x_GPA_mean) / x_GPA_std]
    pm.set_data({"x_shared": [obs_std2]}, model=adm_model)
    ppc = pm.sample_posterior_predictive(idata, model=adm_model, var_names=["theta"])
    y_ppc = ppc.posterior_predictive["theta"].stack(sample=("chain", "draw")).values
    az.plot_posterior(y_ppc, hdi_prob=0.9)
    plt.show()

    print(obs_std1)
    print(obs_std2)
    idx = np.argsort(x_GRE)
    bd = idata.posterior["bd"].mean(("chain", "draw"))[idx]
    plt.scatter(x_GRE, x_GPA, c=[f"C{x}" for x in y])
    plt.plot(x_GRE[idx], bd, color="k")
    plt.scatter(obs_std1[0], obs_std1[1], color="g", label="obs_std1")
    plt.scatter(obs_std2[0], obs_std2[1], color="m", label="obs_std2")
    plt.legend()
    # az.plot_hdi(x_GRE, idata.posterior["bd"], color ='k')
    plt.xlabel("GRE")
    plt.ylabel("GPA")
    plt.show()
