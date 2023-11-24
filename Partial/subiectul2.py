import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

# generam 200 de timpuri medii de așteptare folosind o distributie normala cu parametrii alesi
np.random.seed(42)
miu_real = 10
sigma_real = 2
timp_mediu_asteptare = np.random.normal(loc=miu_real, scale=sigma_real, size=200)

# Pasul 2: Definim modelul Bayesian în PyMC3
with pm.Model() as model:
    # distributia pentru miu si sigma
    miu = pm.Normal("miu", mu=0, sigma=10)  # alegem o distributie normala cu medie 0 și deviatie standard 10 pentru miu
    sigma = pm.HalfNormal("sigma", sigma=10)  # alegem o distributie Half-Normal pentru sigma (deviatie standard pozitiva)

    # Likelihood (verosimilitatea)
    likelihood = pm.Normal("likelihood", mu=miu, sigma=sigma, observed=timp_mediu_asteptare)

# estimam distributia a posteriori pentru parametrul sigma si facem o vizualizare grafica
with model:
    trace = pm.sample(2000, tune=1000)

# vizualizarea graficului
pm.plot_posterior(trace["sigma"], credible_interval=0.95)
plt.xlabel("Valoarea lui sigma")
plt.ylabel("Densitatea de probabilitate")
plt.title("Distributia a posteriori pentru sigma")
plt.show()
