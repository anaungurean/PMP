import random
from scipy import stats
from pgmpy.models import BayesianNetwork, BayesianModel
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination

jucator0_castiga = 0  # Variabila pentru numărul de jocuri câștigate de jucatorul 0
jucator1_castiga = 0  # Variabila pentru numărul de jocuri câștigate de jucatorul 1

for i in range(10000):  # Se simulează 10000 de jocuri
    p0 = 0
    p1 = 0
    moneda = random.random()  # Moneda este aruncată și se decide cine începe, jucatorul 0 sau jucatorul 1
    if moneda < 0.5:
        p0 = 1  # Dacă valoarea generată este mai mică de 0.5, jucatorul 0 începe
    else:
        p1 = 1  # Altfel, jucatorul 1 începe

    if p1 == 1:
        stema_moneda1 = stats.binom.rvs(1, 2 / 3)  # Se simulează aruncarea primei monede pentru jucatorul 1
    elif p0 == 1:
        stema_moneda1 = stats.binom.rvs(1, 0.5)  # Se simulează aruncarea primei monede pentru jucatorul 0

    if stema_moneda1 == 1:
        n = 1  # Dacă a ieșit stema 1, n = 1, altfel n = 0
    else:
        n = 0

    m = 0
    if p1 == 1:
        stema_moneda2 = stats.binom.rvs(1, 2 / 3,
                                        size=n + 1)  # Se simulează aruncarea celei de-a doua monede pentru jucatorul 1
    elif p0 == 1:
        stema_moneda2 = stats.binom.rvs(1, 0.5,
                                        size=n + 1)  # Se simulează aruncarea celei de-a doua monede pentru jucatorul 0

    for i in range(n + 1):
        if stema_moneda2[i] == 1:
            m += 1  # Se numără câte steme au ieșit din cele simulate

    if n >= m:  # Condiția de câștig: dacă numărul de steme estimate (n) este mai mare sau egal cu numărul real de steme (m)
        if p0 == 1:
            jucator0_castiga += 1  # Jucatorul 0 câștigă
    else:
        if p1 == 1:
            jucator1_castiga += 1  # Jucatorul 1 câștigă

# Se afișează procentul de jocuri câștigate de fiecare jucător
print("Player J0 poate castiga cu sansele de ", jucator0_castiga / 10000 * 100, "%")
print("Player J1 poate castiga cu sansele de", jucator1_castiga / 10000 * 100, "%")

#Modelul Bayesian
model = BayesianNetwork([('PrimPlayer', 'n'), ('n', 'm'), ('PrimPlayer', 'm')])

# Adăugarea modelelor CPD (Conditional Probability Distribution)
cpd_starting_player = TabularCPD('PrimPlayer', 2, [[0.5], [0.5]])

cpd_n = TabularCPD('n', 2, [[2/3, 0.5], [1/3, 0.5]], evidence=['PrimPlayer'], evidence_card=[2])

cpd_m = TabularCPD('m', 2, [[2/3, 1/3, 0.5, 0.5], [1/3, 2/3, 0.5, 0.5]], evidence=['n', 'PrimPlayer'], evidence_card=[2, 2])

# Adăugarea modelelor CPD la modelul Bayesian
model.add_cpds(cpd_starting_player, cpd_n, cpd_m)

# Verificarea modelului
model.check_model()

# Desenarea modelului
pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, node_color='skyblue')
plt.show()

# Inferența
infer = VariableElimination(model) #Crearea unui obiect VariableElimination cu modelul specificat

prob_jucator0_stiind_m = infer.query(variables=['PrimPlayer'], evidence={'m': 1}) #Se calculează probabilitatea ca jucatorul 0 să înceapă jocul
print(prob_jucator0_stiind_m)

