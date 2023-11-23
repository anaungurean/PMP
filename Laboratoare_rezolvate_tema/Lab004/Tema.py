import numpy as np
import scipy.stats as stats

numar_simulari = 10_000
lambda_poisson = 20
numar_clienti = stats.poisson.rvs(mu=lambda_poisson, size=numar_simulari)

mu_normal = 2  # media
sigma_normal = 0.5  # deviația standard
timp_plasare_plata = stats.norm.rvs(loc=mu_normal, scale=sigma_normal, size=numar_simulari)

# alpha_exponential = 400
# timp_gatire = stats.expon.rvs(scale=alpha_exponential, size=numar_simulari)
#
# print("Rezultate simulare:")
# print("-------------------")
# print(f"Numărul mediu de clienți intră în restaurant într-o oră: {np.mean(numar_clienti):.2f}")
# print(f"Deviația standard a numărului de clienți: {np.std(numar_clienti):.2f}")
# print()
# print(f"Timpul mediu de plasare și plată a unei comenzi: {np.mean(timp_plasare_plata):.2f} minute")
# print(f"Deviația standard a timpului de plasare și plată: {np.std(timp_plasare_plata):.2f} minute")
# print()
# print(f"Timpul mediu de gătire a unei comenzi: {np.mean(timp_gatire):.2f} minute")
# print(f"Deviația standard a timpului de gătire: {np.std(timp_gatire):.2f} minute")


def timp_servire_sub_15_minute(alpha, numar_simulari, lambda_poisson, mu_normal, sigma_normal):
    numar_clienti = stats.poisson.rvs(mu=lambda_poisson, size=numar_simulari)
    timp_plasare_plata = stats.norm.rvs(loc=mu_normal, scale=sigma_normal, size=numar_simulari)
    timp_gatire = stats.expon.rvs(scale=alpha, size=numar_simulari)

    timp_total = timp_plasare_plata + timp_gatire
    timp_servire_95 = np.percentile(timp_total, 95)
    return timp_servire_95

alpha_max = 0
alpha_values = np.linspace(10, 0, 1000)   
for alpha in alpha_values:
    timp_servire_95 = timp_servire_sub_15_minute(alpha, numar_simulari, lambda_poisson, mu_normal, sigma_normal)
    print(f"Testăm α = {alpha:.3f}, timpul de servire pentru 95% dintre clienți = {timp_servire_95:.3f} minute")
    if timp_servire_95 <= 15:
        alpha_max = alpha
        break

if alpha_max > 0:
    print("Valoarea maximă a lui α pentru care timpul total de servire este sub 15 minute pentru 95% dintre clienți este:", alpha_max)
else:
    print("Nu s-a găsit o valoare a lui α care să îndeplinească condiția.")


timp_plasare_plata = stats.norm.rvs(loc=mu_normal, scale=sigma_normal, size=numar_simulari)
timp_gatire = stats.expon.rvs(scale=alpha_max, size=numar_simulari)

timp_asteptare_mediu = np.mean(timp_plasare_plata + timp_gatire)
print(f"Timpul mediu de așteptare pentru a fi servit este: {timp_asteptare_mediu:.2f} minute")
