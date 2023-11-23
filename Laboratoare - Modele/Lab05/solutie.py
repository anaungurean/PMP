import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import arviz as az

count_data = np.loadtxt("trafic.csv", delimiter=',', dtype=int)
n_count_data = len(count_data)
with pm.Model() as model:
    alpha = 1.0 / count_data[:, 1].mean()

    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    lambda_3 = pm.Exponential("lambda_3", alpha)
    lambda_4 = pm.Exponential("lambda_4", alpha)
    lambda_5 = pm.Exponential("lambda_5", alpha)

    interval1 = pm.Normal("interval1", 60 * 3)
    interval2 = pm.Normal("interval2", 60 * 12)
    interval3 = pm.Normal("interval3", 60 * 15)
    interval4 = pm.Normal("interval4", 60 * 20)

    tau1 = pm.DiscreteUniform("tau1", lower=1, upper=interval1)
    tau2 = pm.DiscreteUniform("tau2", lower=tau1, upper=interval2)
    tau3 = pm.DiscreteUniform("tau3", lower=tau2, upper=interval3)
    tau4 = pm.DiscreteUniform("tau4", lower=tau3, upper=interval4)

    idx = np.arange(n_count_data)
    lmbd1 = pm.math.switch(tau1 > idx, lambda_1, lambda_2)
    lmbd2 = pm.math.switch(tau2 > idx, lmbd1, lambda_3)
    lmbd3 = pm.math.switch(tau3 > idx, lmbd2, lambda_4)
    lmbd4 = pm.math.switch(tau4 > idx, lmbd3, lambda_5)
    observation = pm.Poisson("obs", lmbd4, observed=count_data[:, 1])
    trace = pm.sample(10,tune = 10, cores=1)
    az.plot_posterior(trace)
    plt.show()

# Extract the most probable intervals and parameter values
map_estimate = pm.find_MAP(model=model)

# Extrageți valorile MAP pentru parametrii și taus
lambda_1_map = map_estimate['lambda_1']
lambda_2_map = map_estimate['lambda_2']
lambda_3_map = map_estimate['lambda_3']
lambda_4_map = map_estimate['lambda_4']
lambda_5_map = map_estimate['lambda_5']

tau1_map = map_estimate['tau1']
tau2_map = map_estimate['tau2']
tau3_map = map_estimate['tau3']
tau4_map = map_estimate['tau4']

# Calculați capetele intervalelor și valorile asociate
interval_1 = (0, tau1_map)
interval_2 = (tau1_map, tau2_map)
interval_3 = (tau2_map, tau3_map)
interval_4 = (tau3_map, tau4_map)
interval_5 = (tau4_map, n_count_data)

lambda_values = [lambda_1_map, lambda_2_map, lambda_3_map, lambda_4_map, lambda_5_map]

# Afișați rezultatele
print("Intervalele probabile și valorile asociate:")
for i in range(5):
    print(f"Interval {i + 1}: {eval('interval_' + str(i + 1))}, Lambda: {lambda_values[i]}")

# Afișați histograma valorilor λ
az.plot_posterior(map_estimate)
plt.show()