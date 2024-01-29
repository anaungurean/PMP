from scipy.stats import geom
import numpy as np
import matplotlib.pyplot as plt

# Exercitiul 2 a)
# Parametri pentru distributia geometrica
theta_X = 0.3
theta_Y = 0.5

# Numarul de realizari
N = 10000

# Generare de variabile aleatoare geometrice
x = geom.rvs(theta_X, size=N)
y = geom.rvs(theta_Y, size=N)

inside = x > y**2
pi = inside.sum() * 4 / N
error = abs((pi - np.pi) / pi) * 100
outside = np.invert(inside)
plt.figure(figsize=(8, 8))
plt.plot(x[inside], y[inside], "b.")
plt.plot(x[outside], y[outside], "r.")
plt.plot(0, 0, label=f"π*= {pi:4.3f}\\nerror = {error:4.3f}", alpha=0)
plt.axis("square")
plt.xticks([])
plt.yticks([])
plt.legend(loc=1, frameon=True, framealpha=0.9)
plt.show()


#Exercitiul 2 b)
k = 30

# Rezultatele aproximării Monte Carlo
results = []

# Realizare k aproximări Monte Carlo
for _ in range(k):
    X = geom.rvs(theta_X)
    Y = geom.rvs(theta_Y)

    if X > Y**2:
        results.append(1)
    else:
        results.append(0)

# Media și deviația standard pentru rezultate
mean_estimate = np.mean(results)
std_dev_estimate = np.std(results)


print("Rezultatele exercitiul 2b)")
print("Aproximarea mediei:", mean_estimate)
print("Aproximarea deviației standard:", std_dev_estimate)




