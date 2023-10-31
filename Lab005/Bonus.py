import scipy.stats as stats
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

numar_simulari = 10_000
lambda_poisson = 20
mu_normal = 2
sigma_normal = 0.5
alpha = 3

def genereaza_timp_asteptare_mediu(alpha, lambda_poisson, mu_normal, sigma_normal):
    numar_clienti = stats.poisson.rvs(mu=lambda_poisson)
    timp_plasare_plata = stats.norm.rvs(loc=mu_normal, scale=sigma_normal, size=numar_clienti)
    timp_gatire = stats.expon.rvs(scale=alpha, size=numar_clienti)
    timp_total = timp_plasare_plata + timp_gatire
    return np.mean(timp_total)

def run_model():
    alpha = 3
    timpi_asteptare_medii = [genereaza_timp_asteptare_mediu(alpha, lambda_poisson, mu_normal, sigma_normal) for _ in
                             range(100)]
    print("Timpi de așteptare medii:", timpi_asteptare_medii)
    print("Media timpi de așteptare medii:", np.mean(timpi_asteptare_medii))
    print("Deviația standard a timpilor de așteptare medii:", np.std(timpi_asteptare_medii))

    with pm.Model() as model:
        alpha = pm.Exponential('alpha', lam=1/3)
        timp_asteptare_observat = pm.Normal('timp_asteptare_observat', mu=2+alpha, sigma=0.5, observed=timpi_asteptare_medii)
        trace = pm.sample(2000, tune=1000)

    az.plot_trace(trace, var_names=['alpha'])
    plt.show()

    az.plot_kde(trace['alpha'], fill_kwargs={'alpha': 0.5})
    plt.axvline(x=3, color='r', linestyle='--')
    plt.title('Distribuția a Posteriori a lui α')
    plt.show()

    summary = az.summary(trace, var_names=['alpha'])
    print(summary)

if __name__ == '__main__':
    run_model()


