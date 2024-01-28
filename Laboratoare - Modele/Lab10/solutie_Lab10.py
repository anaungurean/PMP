import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt


def get_data(order, x_1, y_1):
    x_1p = np.vstack([x_1**i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()

    return (x_1s, y_1s)


if __name__ == "__main__":
    dummy_data = np.loadtxt("Laboratoare - Modele\Lab10\dummy.csv")
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    x_1s, y_1s = get_data(5, x_1, y_1)

    order = 5
    model_p1 = pm.Model()
    model_p2 = pm.Model()
    model_p3 = pm.Model()

    # Exercitiul 1
    # a)
    with model_p1:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=order)
        epsilon = pm.HalfNormal("epsilon", 5)
        miu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal("y_pred", mu=miu, sigma=epsilon, observed=y_1s)
        idata_p1 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)

    # b.1
    with model_p2:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=100, shape=order)
        epsilon = pm.HalfNormal("epsilon", 5)
        miu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal("y_pred", mu=miu, sigma=epsilon, observed=y_1s)
        idata_p2 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)

    # b.2)
    with model_p3:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal(
            "beta", mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order
        )
        epsilon = pm.HalfNormal("epsilon", 5)
        miu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal("y_pred", mu=miu, sigma=epsilon, observed=y_1s)
        idata_p3 = pm.sample(2000, return_inferencedata=True)

    # Exercitiul 2
    # adăugăm alte date pentru a ajunge la 500, la o scară asemănătoare:
    x_1_add = np.random.normal(np.mean(x_1), np.std(x_1), size=500 - len(x_1))
    y_1_add = np.random.normal(np.mean(y_1), np.std(y_1), size=500 - len(y_1))
    x_1_500 = np.concatenate((x_1, x_1_add))
    y_1_500 = np.concatenate((y_1, y_1_add))

    # vizualizarea datelor:
    plt.scatter(x_1, y_1)
    plt.scatter(x_1_add, y_1_add, color="m", alpha=0.5)
    plt.show()

    # Exercitiul 3
    # modelul liniar (din curs):
    x_1s, y_1s = get_data(1, x_1, y_1)
    with pm.Model() as model_l:
        α = pm.Normal("α", mu=0, sigma=1)
        β = pm.Normal("β", mu=0, sigma=10)
        ε = pm.HalfNormal("ε", 5)
        μ = α + β * x_1s[0]
        y_pred = pm.Normal("y_pred", mu=μ, sigma=ε, observed=y_1s)
        idata_l = pm.sample(2000, return_inferencedata=True)

    model_p_ord2 = pm.Model()
    model_p_ord3 = pm.Model()

    order = 2
    x_1s, y_1s = get_data(order, x_1, y_1)
    with model_p_ord2:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=order)
        epsilon = pm.HalfNormal("epsilon", 5)
        miu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal("y_pred", mu=miu, sigma=epsilon, observed=y_1s)
        idata_p_ord2 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)

    order = 3
    x_1s, y_1s = get_data(order, x_1, y_1)
    with model_p_ord3:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=order)
        epsilon = pm.HalfNormal("epsilon", 5)
        miu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal("y_pred", mu=miu, sigma=epsilon, observed=y_1s)
        idata_p_ord3 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)

    pm.compute_log_likelihood(idata_l, model=model_l)
    pm.compute_log_likelihood(idata_p_ord2, model=model_p_ord2)
    pm.compute_log_likelihood(idata_p_ord3, model=model_p_ord3)

    cmp_waic = az.compare({'model_l':idata_l, 'model_p_ord2':idata_p_ord2, 'model_p_ord3':idata_p_ord3},method='BB-pseudo-BMA', ic="waic", scale="deviance")
    print(cmp_waic)
    az.plot_compare(cmp_waic)
    plt.show()

    cmp_loo = az.compare({'model_l':idata_l, 'model_p_ord2':idata_p_ord2, 'model_p_ord3':idata_p_ord3},method='BB-pseudo-BMA', ic="loo", scale="deviance")
    print(cmp_loo)
    az.plot_compare(cmp_loo)
    plt.show()

