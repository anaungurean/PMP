# Se consideră un experiment aleator prin aruncarea de 10 ori a două monezi, una nemăsluită, cealaltă cu
# probabilitatea de 0.3 de a obţine stemă. 
# Să se genereze 100 de rezultate independente ale acestui experiment şi astfel
# să se determine grafic distribuţiile variabilelor aleatoare 
# care numără rezultatele posibile în cele 10 aruncări (câte una pentru fiecare rezultat posibil: ss, sb, bs, bb).


import random
import matplotlib.pyplot as plt

results = []

for _ in range(100):
    result_experiment = ""
    for _ in range(10):
        moneda1 = random.choice(["S", "B"])  # Moneda buna
        moneda2 = random.choices(["S", "B"], weights=[0.3, 0.7])[0]  # Moneda rea
        result_experiment += moneda1 + moneda2
    results.append(result_experiment)

# Numaram aparitia a fiecare tip 
count_ss = results.count("SS")
count_sb = results.count("SB")
count_bs = results.count("BS")
count_bb = results.count("BB")

rezultat = ["SS", "SB", "BS", "BB"]
counts = [count_ss, count_sb, count_bs, count_bb]

print(results)
plt.bar(rezultat, counts)
plt.xlabel("Rezultat")
plt.ylabel("Numarari")
plt.title("Distribuire ")
plt.show()
