import pymc as pm
import pandas as pd
import arviz as az

# veriicare citire corecta din csv
df = pd.read_csv('trafic.csv')
df = df.sort_values(by='minut')
pd.set_option('display.max_rows', None)
#print(df)


with pm.Model() as traffic_model:
    # Parametrii lambda pentru fiecare interval de timp
    lambda_1 = pm.Exponential('lambda_1', lam=1)
    lambda_2 = pm.Exponential('lambda_2', lam=1)
    lambda_3 = pm.Exponential('lambda_3', lam=1)
    lambda_4 = pm.Exponential('lambda_4', lam=1)
    lambda_5 = pm.Exponential('lambda_5', lam=1)

    # Alocarea valorilor de trafic la fiecare interval de timp
    idx_interval_1 = (df['minut'] >= 4 * 60) & (df['minut'] < 7 * 60)
    idx_interval_2 = (df['minut'] >= 7 * 60) & (df['minut'] < 8 * 60)
    idx_interval_3 = (df['minut'] >= 8 * 60) & (df['minut'] < 16 * 60)
    idx_interval_4 = (df['minut'] >= 16 * 60) & (df['minut'] < 19 * 60)
    idx_interval_5 = (df['minut'] >= 19 * 60) & (df['minut'] <= 24 * 60)

    # Definirea observațiilor pentru fiecare interval de timp
    obs_1 = pm.Poisson('obs_1', mu=lambda_1, observed=df['nr. masini'][idx_interval_1])
    obs_2 = pm.Poisson('obs_2', mu=lambda_2, observed=df['nr. masini'][idx_interval_2])
    obs_3 = pm.Poisson('obs_3', mu=lambda_3, observed=df['nr. masini'][idx_interval_3])
    obs_4 = pm.Poisson('obs_4', mu=lambda_4, observed=df['nr. masini'][idx_interval_4])
    obs_5 = pm.Poisson('obs_5', mu=lambda_5, observed=df['nr. masini'][idx_interval_5])


with traffic_model:
    trace = pm.sample(2000)
    az.plot_posterior(trace, var_names=['lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_5'])

    lambda_1_est = trace['lambda_1'].mean()
    lambda_2_est = trace['lambda_2'].mean()
    lambda_3_est = trace['lambda_3'].mean()
    lambda_4_est = trace['lambda_4'].mean()
    lambda_5_est = trace['lambda_5'].mean()
