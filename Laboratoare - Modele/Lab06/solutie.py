# ex2
model = BayesianModel([('StartingPlayer', 'n'), ('n', 'm')])

cpd_starting_player = TabularCPD('StartingPlayer', 2, [[0.5], [0.5]])

cpd_n = TabularCPD('n', 2, [[2/3, 0.5], [1/3, 0.5]], evidence=['StartingPlayer'], evidence_card=[2])

cpd_m = TabularCPD('m', 2, [[2/3, 0.5], [1/3, 0.5]], evidence=['n'], evidence_card=[2])

model.add_cpds(cpd_starting_player, cpd_n, cpd_m)

model.check_model()

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()

# ex3
infer = VariableElimination(model)

prob_jucator0_stiind_m = infer.query(variables=['StartingPlayer'], evidence={'m': 1})
print(prob_jucator0_stiind_m)