import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(42)
x_500 = np.sort(5 * np.random.rand(500))
y_500 = np.sin(x_500) + 0.1 * np.random.randn(500)

order = 5
x_500p = np.vstack([x_500**i for i in range(1, order + 1)])
x_500s = (x_500p - x_500p.mean(axis=1, keepdims=True)) / x_500p.std(axis=1, keepdims=True)
y_500s = (y_500 - y_500.mean()) / y_500.std()

# Modelul liniar
with pm.Model() as model_l_500:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10)
    eps = pm.HalfNormal('eps', 5)
    mu = alpha + beta * x_500s[0]
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_500s)
    idata_l_500 = pm.sample(2000, return_inferencedata=True)

# Modelul polinomial
with pm.Model() as model_p_500:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
    eps = pm.HalfNormal('eps', 5)
    mu = alpha + pm.math.dot(beta, x_500s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_500s)
    idata_p_500 = pm.sample(2000, return_inferencedata=True)

# Reprezentarea grafică a curbelor pentru modelul polinomial cu 500 de puncte de date
plt.figure(figsize=(12, 8))
plt.scatter(x_500, y_500, label='Date observate', alpha=0.5)

# Grafic pentru modelul liniar
az.plot_hdi(x_500, idata_l_500.posterior_predictive['y_pred'], label='Model liniar')

# Grafic pentru modelul polinomial cu 500 de puncte
az.plot_hdi(x_500, idata_p_500.posterior_predictive['y_pred'], label='Model polinomial (500 de puncte)')

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Inferență bayesiană - Model polinomial cu 500 de puncte')
plt.show()
