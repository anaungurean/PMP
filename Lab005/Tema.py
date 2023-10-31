import numpy as np
import pandas as pd
import pymc3 as pm

data = pd.read_csv('trafic.csv', usecols=['minut', 'nr. masini'])

average_changes = [7 * 60, 8 * 60, 16 * 60, 19 * 60]

with pm.Model() as model:
    lambda_prior = pm.Gamma("lambda", alpha=1, beta=0.1)

    intervals = []
    lambda_values = []

    for i in range(1, len(average_changes)):
        start = average_changes[i - 1]
        end = average_changes[i]
        lambda_value = pm.Deterministic(f"lambda_{i}", lambda_prior)
        intervals.append((start, end))
        lambda_values.append(lambda_value)

    observations = []
    for i in range(1, len(intervals) + 1):
        obs = pm.Poisson(f"observations_{i}", mu=lambda_values[i - 1], observed=data['nr. masini'][(data['minut'] >= intervals[i - 1][0]) & (data['minut'] < intervals[i - 1][1])].values)
        observations.append(obs)

    trace = pm.sample(100, cores=1)

probable_intervals = []
probable_lambda_values = []

for i in range(1, len(average_changes)):
    start = average_changes[i - 1]
    end = average_changes[i]
    lambda_samples = trace[f"lambda_{i}"]
    probable_interval = (start, end)
    probable_lambda_value = lambda_samples.mean()
    probable_intervals.append(probable_interval)
    probable_lambda_values.append(probable_lambda_value)

print("The probable time intervals are:", probable_intervals)
print("The probable values of lambda are:", probable_lambda_values)
