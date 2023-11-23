import numpy as np
from scipy import stats

n_value = 100
np.random.seed(1)

alfa = 0.0
while True:
    client_wait_time = []
    good = 0
    clients_number = stats.poisson.rvs(20.0, size=n_value)
    for c_nr in clients_number:
        order_time = stats.norm.rvs(2.0, 0.5, size=c_nr)
        preparation_time = stats.expon.rvs(loc=alfa, size=c_nr)
        wait = order_time + preparation_time

        ok = True
        for w in wait:
            if w >= 15:
                ok = False
            client_wait_time.append(w)

        if ok:
            good += 1

    if good/n_value <= 0.95:
        break

    mean = sum(client_wait_time) / len(client_wait_time)
    alfa += 0.001

print(f"alfa is {alfa}")
print(f"mean in {mean}")