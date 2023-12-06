import pymc as pm
import theano.tensor as tt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# incarcam datele
admission_data = pd.read_csv('Admission.csv')

# definim modelul
with pm.Model() as logistic_model:
    # priors pentru parametrii modelului
    beta0 = pm.Normal('beta0', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=10)
    beta2 = pm.Normal('beta2', mu=0, sd=10)

    # modelul logistic
    logits = beta0 + beta1 * admission_data['GRE'] + beta2 * admission_data['GPA']
    p = pm.Deterministic('p', 1 / (1 + tt.exp(-logits)))

    # likelihood-ul observatiilor
    Y_obs = pm.Bernoulli('Y_obs', p=p, observed=admission_data['Admission'])

    # esantionarea din distributia a posteriori
    trace = pm.sample(2000, tune=1000, target_accept=0.95)

summary = pm.summary(trace)
print(summary)

# calculam si reprezentam grafic granita de decizie
beta0_samples = trace['beta0']
beta1_samples = trace['beta1']
beta2_samples = trace['beta2']

# calculam granita de decizie pentru fiecare esantion
decision_boundary = -beta0_samples / beta2_samples - (beta1_samples / beta2_samples) * admission_data['GRE']

# calculam intervalul HDI
hdi_interval = az.hdi(decision_boundary, hdi_prob=0.94)

# reprezentarea grafica
plt.hist(decision_boundary, bins=30, density=True)
plt.axvline(hdi_interval[0], color='red')
plt.axvline(hdi_interval[1], color='red')
plt.title('Distributia granitei de decizie cu intervalul 94% HDI')
plt.xlabel('Granita de decizie')
plt.ylabel('Densitate')
plt.show()


# functie pentru calcularea intervalului HDI pentru un student specific
def compute_hdi_for_student(gre, gpa, trace, hdi_prob=0.9):
    p_student = 1 / (1 + np.exp(-(trace['beta0'] + trace['beta1'] * gre + trace['beta2'] * gpa)))
    return az.hdi(p_student, hdi_prob=hdi_prob)


# exemplu pentru un student cu GRE 550 si GPA 3.5
hdi_student1 = compute_hdi_for_student(550, 3.5, trace)
print(f'Interval HDI pentru studentul cu GRE 550 și GPA 3.5: {hdi_student1}')

# exemplu pentru un student cu GRE 500 si GPA 3.2
hdi_student2 = compute_hdi_for_student(500, 3.2, trace)
print(f'Interval HDI pentru studentul cu GRE 500 și GPA 3.2: {hdi_student2}')
