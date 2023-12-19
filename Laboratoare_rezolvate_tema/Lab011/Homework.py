import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import seaborn as sns

#Exercițiul 1
clusters = 3
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)
means = [5, 0, -5]
std_devs = [2, 2, 2]

mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))

plt.hist(mix, bins=30, density=True, alpha=0.5, color='b')
az.plot_kde(np.array(mix))

plt.title('Mitură de trei distribuții gaussiene')
plt.xlabel('Valoare')
plt.ylabel('Densitate')
plt.show()

#Exercițiul 2
for num_components in [2, 3, 4]:
    model = GaussianMixture(n_components=num_components, random_state=42)
    model.fit(mix)
    plt.figure()
    sns.histplot(mix, bins=30, kde=True, color='blue', alpha=0.5, label='Date generate')
    sns.histplot(model.sample(n_total)[0], bins=30, kde=True, color='red', alpha=0.5, label='Model')
    plt.title(f'Model cu {num_components} componente')
    plt.xlabel('Valoare')
    plt.ylabel('Densitate')
    plt.legend()
    plt.show()

#Exercițiul 3