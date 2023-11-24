import pandas as pd
import pymc3 as pm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import theano.tensor as tt
import arviz as az
import matplotlib.pyplot as plt
#
# def lab7():
#     file_path = 'auto-mpg.csv'
#     df = pd.read_csv(file_path)
#     df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
#     df = df.dropna(subset=['horsepower'])
#     horsepower = df['horsepower'].values.astype(float)
#     mpg = df['mpg'].values.astype(float)
#
#     plt.scatter(horsepower, mpg, marker='o')
#     plt.xlabel('horsepower')
#     plt.ylabel('mpg')
#     plt.title('my_data')
#     plt.show()
#
#     with pm.Model() as model:
#         alpha = pm.Normal('alpha', mu=0, sd=1)
#         beta = pm.Normal('beta', mu=0, sd=10)
#         eps = pm.HalfCauchy('eps', 5)
#         niu = pm.Deterministic('niu', horsepower * beta + alpha)
#         mpg_pred = pm.Normal('mpg_pred', mu=niu, sigma=eps, observed=mpg)
#
#     with model:
#         map_estimate = pm.find_MAP()
#         alpha_map = map_estimate['alpha']
#         beta_map = map_estimate['beta']
#         print(f'Dreapta de regresie: y = {alpha_map:.2f} + {beta_map:.2f} * CP')
#
#     sns.lmplot(x='horsepower', y='mpg', data=df, ci=None, line_kws={'color': 'red'})
#     plt.title('Regresia liniară și regiunea de încredere')
#     plt.xlabel('Cai Putere (CP)')
#     plt.ylabel('Mile pe Galon (mpg)')
#     plt.show()
#
# if __name__ == '__main__':
#     np.random.seed(1)
#     lab7()
#
#
#
# idata_g = pm.sample(2000, tune=2000, return_inferencedata=True)
def lab7():

    file_path = 'auto-mpg.csv'
    df = pd.read_csv(file_path)
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df = df.dropna(subset=['horsepower'])
    horsepower = df['horsepower'].values.astype(float)
    mpg = df['mpg'].values.astype(float)
    plt.scatter(horsepower, mpg, marker='o')
    plt.xlabel('horsepower')
    plt.ylabel('mpg')
    plt.title('my_data')
    plt.show()

    with pm.Model() as model_regression:        # b
        alfa = pm.Normal('alfa', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=1)
        eps = pm.HalfCauchy('eps', 5)
        niu = pm.Deterministic('niu', horsepower * beta + alfa)
        mpg_pred = pm.Normal('mpg_pred', mu=niu, sigma=eps, observed=mpg)
        idata = pm.sample(10, tune=10, cores=1)

    az.plot_trace(idata, var_names=['alfa', 'beta', 'eps'])
    plt.show()

    with model_regression:
        map_estimate = pm.find_MAP()
        alpha_map = map_estimate['alfa']
        beta_map = map_estimate['beta']
        print(f'Dreapta de regresie: y = {alpha_map:.2f} + {beta_map:.2f} * CP')

    sns.lmplot(x='horsepower', y='mpg', data=df, ci=None, line_kws={'color': 'red'})
    plt.title('Regresia liniară și regiunea de încredere')
    plt.xlabel('Cai Putere (CP)')
    plt.ylabel('Mile pe Galon (mpg)')
    plt.show()




if __name__ == '__main__':
    np.random.seed(1)
    lab7()