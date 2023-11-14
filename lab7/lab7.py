import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# Incarcam datele din fisierul csv
df = pd.read_csv('auto-mpg.csv')

# Verificam si curatam datele daca este necesar
df = df.dropna(subset=['horsepower'])  # Eliminam randurile cu valori lipsa in coloana 'horsepower'

# Convertim coloana 'horsepower' la numeric
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df.dropna(subset=['horsepower'])  # Eliminam randurrile unde conversia a esuat

# VariabileleA
X = df['horsepower'].values
Y = df['mpg'].values

with pm.Model() as model:
    # Parametrii modelului
    intercept = pm.Normal('Intercept', mu=0, sigma=20)
    slope = pm.Normal('Slope', mu=0, sigma=20)
    sigma = pm.HalfNormal('sigma', sigma=10)

    # Relatia liniara
    likelihood = pm.Normal('Y', mu=intercept + slope * X, sigma=sigma, observed=Y)

    # Sampling
    trace = pm.sample(1000, return_inferencedata=True)

    # Afisam un rezumat al modelului
    print(pm.model_to_graphviz(model))

# Analizam rezultatele
az.plot_trace(trace)
plt.show()

# Extragem estimarile pentru intercept si slope
intercept_estimate = np.mean(trace.posterior['Intercept'].values)
slope_estimate = np.mean(trace.posterior['Slope'].values)

print("Estimarea Interceptului:", intercept_estimate)
print("Estimarea Pantei:", slope_estimate)

with model:
    # Generam valorile predictive
    ppc = pm.sample_posterior_predictive(trace, var_names=['Y'])

# Calculam intervalul HDI pentru fiecare valoare CP
hdi = az.hdi(ppc['Y'], hdi_prob=0.95)

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, c='blue', label='Date originale')
plt.plot(X, intercept_estimate + slope_estimate * X, c='red', label='Dreapta de regresie')

plt.fill_between(X, hdi[:,0], hdi[:,1], color='gray', alpha=0.5, label='95% HDI')


# Cream si afisam graficul
plt.figure(figsize=(10, 6))
plt.scatter(df['horsepower'], df['mpg'])
plt.title('Relatia dintre CP (Cai Putere) si mpg (Mile pe Galon)')
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Mile pe Galon (mpg)')
plt.grid(True)
plt.show()
