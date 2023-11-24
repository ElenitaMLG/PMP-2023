import random
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# Simularea jocului
def simulate_game():

    # am initializat probabilitatea lui P0 de a trisa cu 1/3, cu se cere in exercitiu
    P0_cheating_prob = 1 / 3

    # am initializat probabilitatea lui P1 de a trisa cu 0, deoarece acesta nu are moneda masluita
    P1_cheating_prob = 0

    # daca la aruncarea banului, rezultatul este mai mic de 1/2, inseamna ca avem moneda masluita
    if random.random() < 0.5:
        P0_cheating_prob = 0
        P1_cheating_prob = 1 / 3

    P0_coin = 1 if random.random() < P0_cheating_prob else 0
    P1_coin = 1 if random.random() < P1_cheating_prob else 0

    n = P0_coin
    m = sum([1 if random.random() < 0.5 else 0 for _ in range(n + 1)])

    return n, m


num_simulations = 20000
P0_wins = 0
P1_wins = 0
# aici am facut 2000 de simulari si am numarat cine castiga de cele mai multe ori
for _ in range(num_simulations):
    n, m = simulate_game()
    if n >= m:
        P0_wins += 1
    else:
        P1_wins += 1
# probabilitatea s-a calculat cu formula: nr_cazuri_favorabile/nr_cazuri_totale; aici cazurile favorabile sunt numarul de castiguri pentru fiecare
print("Probabilitatea ca P0 să castige:", P0_wins / num_simulations)
print("Probabilitatea ca P1 să castige:", P1_wins / num_simulations)

# definirea retelei Bayesiane
model = BayesianNetwork([('P0_coin', 'n'), ('P1_coin', 'm'), ('n', 'outcome'), ('m', 'outcome')])

P0_coin_cpd = TabularCPD(variable='P0_coin', variable_card=2, values=[[2 / 3, 1 / 3]])
P1_coin_cpd = TabularCPD(variable='P1_coin', variable_card=2, values=[[2 / 3, 1 / 3]])
n_cpd = TabularCPD(variable='n', variable_card=2, values=[[0.5, 0.5]])
m_cpd = TabularCPD(variable='m', variable_card=2, values=[[0.5, 0.5]])

outcome_cpd = TabularCPD(variable='outcome', variable_card=2,
                         values=[[1, 0, 0, 1],
                                 [0, 1, 1, 0]],
                         evidence=['n', 'm'], evidence_card=[2, 2])

model.add_cpds(P0_coin_cpd, P1_coin_cpd, n_cpd, m_cpd, outcome_cpd)
model.check_model()

# interferenta folosind VariableElimination
inference = VariableElimination(model)
result = inference.query(variables=['P0_coin'], evidence={'m': 0})
print(result)
