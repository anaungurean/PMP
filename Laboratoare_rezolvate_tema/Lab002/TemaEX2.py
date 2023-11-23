# Patru servere web oferă acelaşi serviciu (web) clienţilor . Timpul necesar procesării unei cereri (request)
# HTTP este distribuit
# Γ(4, 3) pe primul server,
# Γ(4, 2) pe cel de-al doilea,
# Γ(5, 2) pe cel de-al treilea,
# Γ(5, 3) pe cel de-al patrulea (în milisecunde).
# La această durată se adaugă latenţa dintre client şi serverele pe Internet, care are o distribuţie exponenţială cu
# λ = 4 (în miliseconde−1).
# Se ştie că un client este direcţionat către
# primul server cu probabilitatea 0.25,
# către al doilea cu probabilitatea 0.25,
#  iar către al treilea server cu probabilitatea 0.30.
# Estimaţi probabilitatea ca timpul necesar servirii unui client, notat cu X, (de la lansarea cererii până la primirea răspunsului) să fie mai mare decât 3 milisecunde.
# Realizaţi un grafic al densităţii distribuţiei lui X.
# Notă: Distribuţia Γ(α, λ) se poate apela cu stats.gamma(α,0,1/λ) sau stats.gamma(α,scale=1/λ).

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

alpha1, lambda1 = 4, 1 / 3  # Server 1
alpha2, lambda2 = 4, 1 / 2  # Server 2
alpha3, lambda3 = 5, 1 / 2  # Server 3
alpha4, lambda4 = 5, 1 / 3  # Server 4

lambda_latenta = 4  # Distribuitia exponentiala

prob_server1 = 0.25
prob_server2 = 0.25
prob_server3 = 0.30

valori_X = 10000

# Timp servire pt fiecare server
timp_server1 = np.random.gamma(alpha1, scale=1 / lambda1, size=valori_X)
timp_server2 = np.random.gamma(alpha2, scale=1 / lambda2, size=valori_X)
timp_server3 = np.random.gamma(alpha3, scale=1 / lambda3, size=valori_X)
timp_server4 = np.random.gamma(alpha4, scale=1 / lambda4, size=valori_X)

# Timp de latență dintre client și servere
latenta = np.random.exponential(scale=1 / lambda_latenta, size=valori_X)

# Servire + latenta
timp_final_server1 = timp_server1 + latenta
timp_final_server2 = timp_server2 + latenta
timp_final_server3 = timp_server3 + latenta
timp_final_server4 = timp_server4 + latenta

probabilitate_X_greater_than_3 = np.mean(
    np.any([timp_final_server1 > 3, timp_final_server2 > 3, timp_final_server3 > 3, timp_final_server4 > 3], axis=0)
)

print("Estimre probabilitatea ca timpul necesar servirii unui cliente este mai mare decat 3: ", probabilitate_X_greater_than_3)

plt.hist(timp_final_server1, label='Distribuție de probabilitate')
x = np.linspace(0, max(timp_final_server1), 1000)
pdf = stats.gamma.pdf(x, alpha1, scale=1/lambda1) * prob_server1

plt.plot(x, pdf, 'r', lw=2, label='Distribuție Gamma')
plt.xlabel('Timp de servire (milisecunde)')
plt.ylabel('Densitatea probabilității')
plt.legend()
plt.title('Densitatea distribuției lui X pentru Serverul 1')
plt.show()