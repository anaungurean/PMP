import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Exercitiul 1
    clusters = 2
    n_cluster = [250, 150, 100]
    n_total = sum(n_cluster)
    means = [-2, 2, 5]
    std_devs = [1, 1, 1]
    mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix))
    plt.show()

    # Exercitiul 2
    clusters = [2, 3, 4]
    models = []
    idatas = []
    for cluster in clusters:
        with pm.Model() as model:
            p = pm.Dirichlet("p", a=np.ones(cluster))
            means = pm.Normal(
                "means",
                mu=np.linspace(mix.min(), mix.max(), cluster),
                sigma=10,
                shape=cluster,
                transform=pm.distributions.transforms.ordered,
            )
            sd = pm.HalfNormal("sd", sigma=10)
            y = pm.NormalMixture("y", w=p, mu=means, sigma=sd, observed=mix)
            idata = pm.sample(
                1000,
                tune=2000,
                target_accept=0.9,
                random_seed=123,
                return_inferencedata=True,
            )
        idatas.append(idata)
        models.append(model)

    # Exercitiul 3
    comp_waic = az.compare(dict(zip([str(c) for c in clusters], idatas)),method='BB-pseudo-BMA', ic="waic", scale="deviance")
    print(comp_waic)
    az.plot_compare(comp_waic)

    comp_loo = az.compare(dict(zip([str(c) for c in clusters], idatas)),method='BB-pseudo-BMA', ic="loo", scale="deviance")
    print(comp_loo)
    az.plot_compare(comp_loo)

    