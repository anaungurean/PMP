import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

possible_Y = [0, 5, 10]
possible_theta = [0.2, 0.5]

n = 10

fig, axes = plt.subplots(len(possible_Y), len(possible_theta), figsize=(12, 8), squeeze=False)

for i, y in enumerate(possible_Y):
    for j, theta in enumerate(possible_theta):
        with pm.Model() as model:
            n = pm.Poisson("n", mu=n)
            Y_obs = pm.Binomial("Y_obs", n=n, p=theta, observed=y)
            trace = pm.sample(1000, tune=1000, cores=1, return_inferencedata=True)
            az.plot_posterior(trace, var_names=['n'], ax=axes[i][j])
            axes[i][j].set_title(f"Y={y}, θ={theta}")

plt.tight_layout()
plt.show()
