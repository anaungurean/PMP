import random
from scipy import stats
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD


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

# Creare model Bayesian
model = BayesianModel([('Moneda', 'Jucator0_castiga'),
                       ('Moneda', 'Jucator1_castiga'),
                       ('Moneda', 'Stema_moneda1'),
                       ('Stema_moneda1', 'N'),
                       ('Jucator1_castiga', 'Stema_moneda1'),
                       ('Jucator0_castiga', 'Stema_moneda1'),
                       ('Stema_moneda1', 'Stema_moneda2'),
                       ('Stema_moneda2', 'M'),
                       ('N', 'Jucator0_castiga'),
                       ('M', 'Jucator1_castiga')])

# Definirea distribuțiilor de probabilitate condiționate (TabularCPD)
cpd_moneda = TabularCPD(variable='Moneda', variable_card=2, values=[[0.5, 0.5]])

cpd_stema_moneda1 = TabularCPD(variable='Stema_moneda1', variable_card=2,
                               values=[[2/3, 1/2], [1/3, 1/2]],
                               evidence=['Moneda'], evidence_card=[2])

cpd_stema_moneda2 = TabularCPD(variable='Stema_moneda2', variable_card=2,
                               values=[[2/3, 1/2], [1/3, 1/2]],
                               evidence=['Stema_moneda1'], evidence_card=[2])

cpd_n = TabularCPD(variable='N', variable_card=2,
                   values=[[0, 1], [1, 0]],
                   evidence=['Stema_moneda1'], evidence_card=[2])

cpd_m = TabularCPD(variable='M', variable_card=2,
                   values=[[0, 1, 2], [1, 0, 0]],
                   evidence=['Stema_moneda2'], evidence_card=[2])

cpd_jucator0_castiga = TabularCPD(variable='Jucator0_castiga', variable_card=2,
                                  values=[[0, 1], [1, 0]],
                                  evidence=['N'], evidence_card=[2])

cpd_jucator1_castiga = TabularCPD(variable='Jucator1_castiga', variable_card=2,
                                  values=[[1, 0], [0, 1]],
                                  evidence=['M'], evidence_card=[2])

# Adăugarea distribuțiilor la model
model.add_cpds(cpd_moneda, cpd_stema_moneda1, cpd_stema_moneda2, cpd_n, cpd_m, cpd_jucator0_castiga, cpd_jucator1_castiga)

# Verificarea consistenței modelului
print("Modelul este consistent:", model.check_model())

# Inferență pe baza datelor
inference = VariableElimination(model)
probabilitate_castig_jucator0 = inference.query(variables=['Jucator0_castiga'], evidence={'Moneda': 0.5})['Jucator0_castiga']
probabilitate_castig_jucator1 = inference.query(variables=['Jucator1_castiga'], evidence={'Moneda': 0.5})['Jucator1_castiga']

# Se afișează probabilitatea de câștig pentru fiecare jucător
print("Probabilitatea ca Jucatorul 0 să câștige:", probabilitate_castig_jucator0.values[1])
print("Probabilitatea ca Jucatorul 1 să câștige:", probabilitate_castig_jucator1.values[1])