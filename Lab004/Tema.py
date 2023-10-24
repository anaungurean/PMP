import numpy as np
import scipy.stats as stats

numar_simulari = 1_000
lambda_poisson = 20
numar_clienti = stats.poisson.rvs(mu=lambda_poisson, size=numar_simulari)

mu_normal = 2  # media
sigma_normal = 0.5  # deviația standard
timp_plasare_plata = stats.norm.rvs(loc=mu_normal, scale=sigma_normal, size=numar_simulari)

alpha_exponential = 4   
timp_gatire = stats.expon.rvs(scale=alpha_exponential, size=numar_simulari)

print("Rezultate simulare:")
print("-------------------")
print(f"Numărul mediu de clienți intră în restaurant într-o oră: {np.mean(numar_clienti):.2f}")
print(f"Deviația standard a numărului de clienți: {np.std(numar_clienti):.2f}")
print()
print(f"Timpul mediu de plasare și plată a unei comenzi: {np.mean(timp_plasare_plata):.2f} minute")
print(f"Deviația standard a timpului de plasare și plată: {np.std(timp_plasare_plata):.2f} minute")
print()
print(f"Timpul mediu de gătire a unei comenzi: {np.mean(timp_gatire):.2f} minute")
print(f"Deviația standard a timpului de gătire: {np.std(timp_gatire):.2f} minute")