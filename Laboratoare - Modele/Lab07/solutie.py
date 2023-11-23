import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az


def read_data():
    file_path = 'auto-mpg.csv'
    df = pd.read_csv(file_path)
    df = df[df['horsepower'] != '?']

    horsepower = df['horsepower'].values.astype(float)
    mpg = df['mpg'].values.astype(float)

    return np.array(horsepower), np.array(mpg)


def plot_data(horsepower, mpg):
    plt.scatter(horsepower, mpg, marker='o')
    plt.xlabel('horsepower')
    plt.ylabel('mpg')
    plt.title('my_data')
    plt.show()


def main():
    horsepower, mpg = read_data()
    plot_data(horsepower, mpg)

    with pm.Model() as model_regression:
        alfa = pm.Normal('alfa', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=1)
        eps = pm.HalfCauchy('eps', 5)
        niu = pm.Deterministic('niu', horsepower[:, np.newaxis] * beta + alfa)
        mpg_pred = pm.Normal('mpg_pred', mu=niu, sigma=eps, observed=mpg)

        # Utilizarea modelului MAP pentru inferență
        map_estimate = pm.find_MAP()

    # Afișarea valorilor MAP
    alpha_map = map_estimate['alfa']
    beta_map = map_estimate['beta']
    print("MAP estimates: alpha =", alpha_map, ", beta =", beta_map)

    # Diagramele pentru urmărirea modelului
    plt.scatter(horsepower, mpg, marker='o')
    plt.xlabel('horsepower')
    plt.ylabel('mpg')
    plt.plot(horsepower, alpha_map + beta_map * horsepower, c='k', label='MAP Fit')
    plt.legend()
    plt.show()

    # Diagrame pentru intervalul de încredere HDI
    hdi_values = np.column_stack([pm.hpd(np.dot(horsepower[:, np.newaxis], beta_map) + alpha_map, alpha=0.05)[0],
                                  pm.hpd(np.dot(horsepower[:, np.newaxis], beta_map) + alpha_map, alpha=0.05)[1]])
    plt.fill_between(horsepower, hdi_values[:, 0], hdi_values[:, 1], color='gray', alpha=0.5, label='HDI')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.random.seed(1)
    main()
