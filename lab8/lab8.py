import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

# incarcam datele
data = pd.read_csv('Prices.csv')
data['log_HardDrive'] = np.log(data['HardDrive'])

# construim modelul de regresie liniara cu PyMC
with pm.Model() as model:
    # parametrii a priori
    alpha = pm.Normal('alpha', mu=0, sigma=100)
    beta1 = pm.Normal('beta1', mu=0, sigma=100)
    beta2 = pm.Normal('beta2', mu=0, sigma=100)
    sigma = pm.HalfNormal('sigma', sigma=100)

    # modelul liniar
    mu = alpha + beta1 * data['Speed'] + beta2 * data['log_HardDrive']

    # likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=data['Price'])

    # esantionul
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# testare
summary = pm.summary(trace)
print(summary)

# calculam intervalului HDI de 95% pentru beta1 si beta2
hdi_beta1 = az.hdi(trace, var_names=['beta1'], hdi_prob=0.95)
hdi_beta2 = az.hdi(trace, var_names=['beta2'], hdi_prob=0.95)

print("95% HDI pentru beta1:", hdi_beta1)
print("95% HDI pentru beta2:", hdi_beta2)


# definim specificatiile pentru pc-ul de interes
speed_interest = 33  # frecventa procesorului in MHz
harddrive_interest = 540  # dimensiunea hard diskului in MB
log_harddrive_interest = np.log(harddrive_interest)

with pm.Model() as model2:
    # cream o noua variabila mu pentru simularea datelor
    mu_simulated = alpha + beta1 * speed_interest + beta2 * log_harddrive_interest
    mu_pred = pm.sample_posterior_predictive(trace, vars=[mu_simulated], samples=5000)

    price_pred = pm.Normal('price_pred', mu=mu_simulated, sigma=sigma, observed=None)
    price_pred_samples = pm.sample_posterior_predictive(trace, vars=[price_pred], samples=5000)


# calculam intervalului HDI de 90% pentru pretul de vanzare asteptat
hdi_90 = az.hdi(mu_pred['mu_simulated'], hdi_prob=0.90)

print("Intervalul HDI de 90% pentru pretul de vanzare asteptat:", hdi_90)

# calculam intervalul de predictie de 90% HDI pentru pretul pc-ului
price_pred_hdi_90 = az.hdi(price_pred_samples['price_pred'], hdi_prob=0.90)

print("Intervalul de predictie HDI de 90% pentru pretul pc-ului:", price_pred_hdi_90)
