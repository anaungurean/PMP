import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

data = np.loadtxt('date.csv')
x_1 = data[:, 0]
y_1 = data[:, 1]

# Modificarea ordinului la 5
order = 5
x_1p = np.vstack([x_1**i for i in range(1, order + 1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()

# Modelul liniar
with pm.Model() as model_l:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10,shape=order)
    eps = pm.HalfNormal('eps', 5)
    mu = alpha + beta * x_1s[0]
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_1s)
    idata_l = pm.sample(2000, return_inferencedata=True)

# Modelul polinomial
with pm.Model() as model_p:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
    eps = pm.HalfNormal('eps', 5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

# Repetare cu distribuție pentru beta cu sd=100
with pm.Model() as model_p_sd100:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=100, shape=order)
    eps = pm.HalfNormal('eps', 5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_1s)
    idata_p_sd100 = pm.sample(2000, return_inferencedata=True)

# Repetare cu distribuție pentru beta cu sd=np.array([10, 0.1, 0.1, 0.1, 0.1])
with pm.Model() as model_p_custom_sd:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
    eps = pm.HalfNormal('eps', 5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_1s)
    idata_p_custom_sd = pm.sample(2000, return_inferencedata=True)

# Reprezentarea grafică a curbelor pentru modelul polinomial
plt.figure(figsize=(12, 8))
plt.scatter(x_1, y_1, label='Date observate', alpha=0.5)

# Grafic pentru modelul liniar
az.plot_hdi(x_1, idata_l.posterior_predictive['y_pred'], label='Model liniar')

# Grafic pentru modelul polinomial cu sd=10
az.plot_hdi(x_1, idata_p.posterior_predictive['y_pred'], label='Model polinomial (sd=10)')

# Grafic pentru modelul polinomial cu sd=100
az.plot_hdi(x_1, idata_p_sd100.posterior_predictive['y_pred'], label='Model polinomial (sd=100)')

# Grafic pentru modelul polinomial cu sd=np.array([10, 0.1, 0.1, 0.1, 0.1])
az.plot_hdi(x_1, idata_p_custom_sd.posterior_predictive['y_pred'], label='Model polinomial (custom sd)')

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Inferență bayesiană - Modele polinomiale')
plt.show()
