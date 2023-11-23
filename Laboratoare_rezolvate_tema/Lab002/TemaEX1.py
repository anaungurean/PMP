# Ex. 1 (1pct.) Doi mecanici schimbă filtrele de ulei pentru autoturisme într-un service. 
# Timpul de servire este exponenţial cu parametrul
# λ1 = 4 hrs−1 în cazul primului mecanic si
# λ2 = 6 hrs−1 în cazul celui de al doilea. 
# Deoarece al doilea mecanic este mai rapid, el serveşte de 1.5 ori mai mulţi clienţi decât partenerul său.
# Astfel când un client ajunge la rând, probabilitatea de a fi servit de primul mecanic este 40%. Fie X timpul de servire pentru un client.
# Realizaţi un grafic al densităţii distribuţiei lui X.
# Notă: Distribuţia Exp(λ) se poate apela cu stats.expon(0,1/λ) sau stats.expon(scale=1/λ).

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

lambda1 = 4 #timp de servire exponential pt. primul mecanic
lambda2 = 6 #timp de servire exponential pt. al doilea mecanic

prob_prim_mecanic = 0.4
valori_X = 10000 
timp_servire = []

for _ in range ( valori_X) :
        
        mecanic = np.random.choice([1, 2], p=[prob_prim_mecanic, 1 - prob_prim_mecanic])
        if mecanic == 1 :
            timp_servire.append(np.random.exponential(1 / lambda1))
        else :
            timp_servire.append(np.random.exponential(1/lambda2))

media = np.mean(timp_servire)  
deviatia_standard = np.std(timp_servire)  

print("Media celor 1000 de valori a lui X: " ,media)
print("Deviatia standard a celor 1000 de valori a lui X: ", deviatia_standard)


plt.hist(timp_servire)
plt.xlabel('Timp de servire')
plt.ylabel('Densitate')
plt.title('Densitatea distribuției timpului de servire')
plt.show()
