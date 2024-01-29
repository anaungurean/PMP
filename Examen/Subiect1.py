import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pytensor as pt

if __name__ == "__main__":

# Exercitiul 1 a)
    # Citim datele din fisier in dataframe
    df = pd.read_csv("Examen\Titanic.csv")
    # Eliminam coloanele care nu sunt semnificative pentru modelul nostru
    df = df.drop(
        [
            "PassengerId",
            "Name",
            "Sex",
            "SibSp",
            "Parch",
            "Ticket",
            "Fare",
            "Cabin",
            "Embarked",
        ],
        axis=1,
    )
    # Eliminam randurile care au valori lipsa
    df = df.dropna()
    survived = df["Survived"].values
    # Generare medie si deviatie standard
    print(len(survived[survived == 1]), len(survived[survived == 0]))
    # Rezultat 127 - 205 => datele sunt neechilibrate => e nevoie de balansare
    # Pentru a balansa datele, alegem la intamplare indici pentru a fi stersi
    Index = np.random.choice(
        np.flatnonzero(survived == 0),
        size=len(survived[survived == 0]) - len(survived[survived == 1]),
        replace=False,
    )

    # Punem coloanele Age, Pclass in variabile
    col_age = df["Age"].values
    col_class = df["Pclass"].values

    # Calculam media si deviatia standard pentru age pentrul a standardiza datele si pentru punctul d
    age_mean = col_age.mean()
    age_std = col_age.std()
    # Standardizam datele pentru age deoarece acestea nu este un atribut discret
    col_age = (col_age - age_mean) / age_std

    X = np.column_stack((col_age, col_class))
    X_mean = X.mean(axis=0, keepdims=True)


# Exercitiul 1 b)
    with pm.Model() as model_mlr:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=2)
        eps = pm.HalfCauchy("ϵ", 5000)
        ν = pm.Exponential("ν", 1 / 30)
        X_shared = pm.MutableData("x_shared", X)
        miu = pm.Deterministic("miu", alpha + pm.math.dot(X_shared, beta))
        theta = pm.Deterministic("theta", pm.math.sigmoid(miu))
        bd = pm.Deterministic(
            "bd", pm.math.sigmoid(ν * (X_shared - alpha) / pm.math.abs(beta))
        )
        y_pred = pm.Bernoulli("y_pred", p=theta, observed=survived) # Avem distributie Bernoulli deoarece eticheta este una binara 
        idata = pm.sample(1250, return_inferencedata=True)

# Exercitiul 1 c)
    az.plot_forest(idata, hdi_prob=0.95, var_names=["beta"])
    plt.show()
    print(az.summary(idata, hdi_prob=0.95, var_names=["beta"]))

    # Se poate observa că coeficientul beta[0] (age) nu are o influență semnificativă
    # asupra output-ului în comparație cu coeficientul beta[1] (class),
    # care are o influență mai mare. De asemenea, se poate observa că coeficientul beta[0] conține valoarea 0 în intervalul său,
    # ceea ce indică că variabila age nu contribuie semnificativ la predicția output-ului.
    # Astfel, putem spune că variabila class este cea care influențează cel mai mult output-ul.

# Exercitiul 1 d)
    obs_std2 = [(30 - age_mean) / age_mean, 2]
    pm.set_data({"x_shared": [obs_std2]}, model=model_mlr)
    ppc = pm.sample_posterior_predictive(idata, model=model_mlr, var_names=["theta"])
    y_ppc = ppc.posterior_predictive["theta"].stack(sample=("chain", "draw")).values
    # Construim interval de incredere daca persoana are 30 de ani si este de la clasa a 2-a va supravietui sau nu (39% din grafic)
    az.plot_posterior(y_ppc, hdi_prob=0.9)
    plt.show()
