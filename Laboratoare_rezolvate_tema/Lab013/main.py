import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

#Exercitiul 1
centered_data = az.load_arviz_data('centered_eight')
non_centered_data = az.load_arviz_data('non_centered_eight')

num_chains_centered = centered_data.posterior.chain.size
num_chains_non_centered = non_centered_data.posterior.chain.size


print(f'Numărul de lanțuri pentru modelul centrat: {num_chains_centered}')
print(f'Numarul de lanturi pentru modelul necentrat: {num_chains_non_centered}')

total_samples_centered = centered_data.posterior.draw.size
total_samples_non_centered = non_centered_data.posterior.draw.size

print(f'Numărul de eșantioane pentru modelul centrat: {total_samples_centered}')
print(f'Numărul de eșantioane pentru modelul necentrat: {total_samples_non_centered}')


az.plot_posterior(centered_data, hdi_prob=0.95)
plt.title("Distribuție a Posteriori - Model Centrat")
# plt.show()

az.plot_posterior(non_centered_data, hdi_prob=0.95)
plt.title("Distribuție a Posteriori - Model Non-Centrat")
# plt.show()


rhats_centered = az.rhat(centered_data, var_names=['mu', 'tau'])
autocorr_mu_centered = az.autocorr(centered_data.posterior["mu"].values)
autocorr_tau_centered = az.autocorr(centered_data.posterior["tau"].values)

rhats_non_centered = az.rhat(non_centered_data, var_names=['mu', 'tau'])
autocorr_mu_non_centered = az.autocorr(non_centered_data.posterior["mu"].values)
autocorr_tau_non_centered = az.autocorr(non_centered_data.posterior["tau"].values)

data = {
    'Model': ['Centrat', 'Necentrat'],
    'Rhat_mu': [rhats_centered['mu'].item(), rhats_non_centered['mu'].item()],
    'Rhat_tau': [rhats_centered['tau'].item(), rhats_non_centered['tau'].item()],
    'Autocorr_mu': [autocorr_mu_centered.mean().item(), autocorr_mu_non_centered.mean().item()],
    'Autocorr_tau': [autocorr_tau_centered.mean().item(), autocorr_tau_non_centered.mean().item()]
}

df = pd.DataFrame(data)
print(df)


# Exercitiul3
divergences_centered = centered_data.sample_stats.diverging.sum()
divergences_non_centered = non_centered_data.sample_stats.diverging.sum()

print(f'Numărul de divergențe pentru modelul centrat: {divergences_centered}')
print(f'Numărul de divergențe pentru modelul necentrat: {divergences_non_centered}')

az.plot_pair(centered_data, var_names=['mu', 'tau'], divergences=True, kind='scatter')
plt.title("Divergențe - Modelul Centrat")
plt.show()

az.plot_pair(non_centered_data, var_names=['mu', 'tau'], divergences=True, kind='scatter')
plt.title("Divergențe - Modelul Non-Centrat")
plt.show()

