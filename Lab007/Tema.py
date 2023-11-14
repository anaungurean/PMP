import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm

# Încărcarea și curățarea datelor
df = pd.read_csv('auto-mpg.csv')
df = df[df['horsepower'] != '?']
df['horsepower'] = pd.to_numeric(df['horsepower'])
df['mpg'] = pd.to_numeric(df['mpg'], errors='coerce').dropna()

# Prepararea datelor pentru model
X = df['horsepower'].values
y = df['mpg'].values

# Construirea modelului liniar în PyMC3
with pm.Model() as linear_model:
    # Parametrii modelului
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)
    a = np.array(df['horsepower'])
    mu = alpha + beta * df['horsepower']
    sigma = pm.HalfNormal('sigma', sd=1)
    mpg = pm.Normal('mpg', mu=mu, sd=sigma, observed=df['mpg'])

# Estimam și afișam parametrii unei drepte de regresie liniară
with linear_model:
    map_estimate = pm.find_MAP()
    alpha_map = map_estimate['alpha']
    beta_map = map_estimate['beta']
    print(f'Dreapta de regresie: y = {alpha_map:.2f} + {beta_map:.2f} * CP')

sns.lmplot(x='horsepower', y='mpg', data=df, ci=None, line_kws={'color': 'red'})
plt.title('Regresia liniară')
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Mile pe Galon (mpg)')
plt.show()

#Observam ca cu cat numarul de cai putere este mai mare, cu atat consumul scade