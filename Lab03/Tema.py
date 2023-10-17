import networkx as nx
from mpmath import plot
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from sympy.physics.control.control_plots import plt

model = BayesianNetwork([('Cutremur', 'Incediu'), ('Cutremur', 'Alarma'), ('Incediu', 'Alarma')])

cpd_c = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])

cpd_i_given_c = TabularCPD(variable='Incediu', variable_card=2,
                          evidence=['Cutremur'], evidence_card=[2],
                          values=[[0.99, 0.97], [0.01, 0.03]])

cpd_a_given_ci = TabularCPD(variable='Alarma', variable_card=2,
                           evidence=['Cutremur', 'Incediu'], evidence_card=[2, 2],
                           values=[[0.9999, 0.05, 0.98, 0.02], [0.0001, 0.95, 0.02, 0.98]])


model.add_cpds(cpd_c, cpd_i_given_c, cpd_a_given_ci)

model.check_model()

print(cpd_a_given_ci)

infer = VariableElimination(model)
prob_E_given_A = infer.query(variables=['Cutremur'], evidence={'Alarma': 1})
print(prob_E_given_A)

prob_F_no_A = infer.query(variables=['Incediu'], evidence={'Alarma': 0})
print(prob_F_no_A)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()