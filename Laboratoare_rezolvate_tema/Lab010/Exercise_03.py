import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(42)
x_500 = np.sort(5 * np.random.rand(500))
y_500 = np.sin(x_500) + 0.1 * np.random.randn(500)

# Modificarea ordinului la 3 pentru modelul cubic
order_cubic = 3
x_500p_cubic = np.vstack([x_500**i for i in range(1, order_cubic + 1)])
x_500s_cubic = (x_500p_cubic - x_500p_cubic.mean(axis=1, keepdims=True)) / x_500p_cubic.std(axis=1, keepdims=True)
y_500s_cubic = (y_500 - y_500.mean()) / y_500.std()

# Modelul cubic
with pm.Model() as model_cubic:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=order_cubic)
    eps = pm.HalfNormal('eps', 5)
    mu =alpha + pm.math.dot(beta, x_500s_cubic)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_500s_cubic)
    idata_cubic = pm.sample(2000, return_inferencedata=True)

# Modelul liniar
with pm.Model() as model_linear:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10)
    eps = pm.HalfNormal('eps', 5)
    mu =alpha + beta * x_500s_cubic[0]
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_500s_cubic)
    idata_linear = pm.sample(2000, return_inferencedata=True)

# Modelul pătratic
with pm.Model() as model_quadratic:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)  # 2 parameters for quadratic
    eps = pm.HalfNormal('eps', 5)
    mu =alpha + pm.math.dot(beta, x_500s_cubic[:2])
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_500s_cubic)
    idata_quadratic = pm.sample(2000, return_inferencedata=True)

# Calcularea WAIC pentru modelele liniare, pătratice și cubice
waic_linear = az.waic(idata_linear, scale='deviance')
waic_quadratic = az.waic(idata_quadratic, scale='deviance')
waic_cubic = az.waic(idata_cubic, scale='deviance')

# Compararea WAIC între modele
print("WAIC - Linear: {:.2f}".format(waic_linear.waic.item()))
print("WAIC - Quadratic: {:.2f}".format(waic_quadratic.waic.item()))
print("WAIC - Cubic: {:.2f}".format(waic_cubic.waic.item()))

# Reprezentarea grafică a curbelor pentru modelele liniare, pătratice și cubice
plt.figure(figsize=(12, 8))
plt.scatter(x_500, y_500, label='Date observate', alpha=0.5)

# Grafic pentru modelul liniar
az.plot_hdi(x_500, idata_linear.posterior_predictive['y_pred'], label='Model liniar')

# Grafic pentru modelul pătratic
az.plot_hdi(x_500, idata_quadratic.posterior_predictive['y_pred'], label='Model pătratic')

# Grafic pentru modelul cubic
az.plot_hdi(x_500, idata_cubic.posterior_predictive['y_pred'], label='Model cubic')

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Inferență bayesiană - Modele liniare, pătratice și cubice')
plt.show()
