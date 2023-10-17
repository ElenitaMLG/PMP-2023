from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# creare obiect model Bayesian
model = BayesianNetwork()

# adaugam variabilele aleatoare in model
model.add_nodes_from(["C", "A", "I"])

# adaugam arcele ( relatiile dintre variablie )
model.add_edge("C", "A")
model.add_edge("C", "I")
model.add_edge("A", "I")

# definim probabilitatile conditionate
cpd_c = TabularCPD(variable="C", variable_card=2, values=[[0.9995], [0.0005]])
cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.999, 0.0001], [0.001, 0.9999]], evidence=["C"], evidence_card=[2])
cpd_i = TabularCPD(variable="I", variable_card=2, values=[[0.99, 0.01], [0.97, 0.03]], evidence=["C"], evidence_card=[2])

# adaugam probabilitatile condiționate la model
model.add_cpds(cpd_c, cpd_a, cpd_i)

assert model.check_model()

inference = VariableElimination(model)

# calcul P(C=1|A=1) - punctul 2
result_cutremur = inference.query(variables=["C"], evidence={"A": 1})
print("Probabilitatea sa fi avut loc un cutremur, stiind că alarma de incendiu a fost declansata.", result_cutremur.values[1])

# Calcularea P(I=1|A=0) - punctul 3
result_incendiu = inference.query(variables=["I"], evidence={"A": 0})
print("Probabilitatea ca un incendiu sa fi avut loc, fara ca alarma de incendiu sa se activeze:", result_incendiu.values[1])
