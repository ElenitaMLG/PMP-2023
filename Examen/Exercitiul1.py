import string
import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az


def read_data():  # a
    file_path = 'Titanic.csv'
    df = pd.read_csv(file_path)
    df = df[df['Age'] != '']
    df = df[df['Fare'] != '']
    df = df[df['Cabin'] != '']

    # atributele care au valori nule in csv
    Age = df['Age'].values.astype(int)
    Fare = df['Fare'].values.astype(float)
    Cabin = df['Fare'].values.astype(string)



def main():
    age, fare, cabin, pclass, survived = read_data()
    # citim datele de care avem nevoie si calculam survived in functe de age si pclass
    with pm.Model() as model_regression:  # b
        alfa = pm.Normal('alfa', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=1)
        eps = pm.HalfCauchy('eps', 5)
        niu = pm.Deterministic('niu', age * beta + pclass * beta + alfa)
        survived_pred = pm.Normal('survived_pred', mu=niu, sigma=eps, observed=survived)
        idata = pm.sample(2000, tune=2000, return_inferencedata=True)

    az.plot_trace(idata, var_names=['alfa', 'beta', 'eps'])
    plt.show()

    # variabila care a influentat cel mai mult rezultatul de supravietuire este Sex-ul

    # d
    posterior_data = idata['posterior']
    alpha_m = posterior_data['alfa'].mean().item()
    beta_m = posterior_data['beta'].mean().item()
    print(alpha_m, beta_m)

    plt.scatter(age=30, pclass=2, marker='o')
    plt.xlabel('age')
    plt.ylabel('pclass')
    plt.plot(age, alpha_m + beta_m * survived, c='k')
    az.plot_hdi(survived, posterior_data['niu'], hdi_prob=0.90, color='k')
    ppc = pm.sample_posterior_predictive(idata, model=model_regression)
    posterior_predictive = ppc['posterior_predictive']
    az.plot_hdi(survived, posterior_predictive['mpg_pred'], hdi_prob=0.90, color='gray', smooth=False)
    plt.show()


if __name__ == "__main__":
    np.random.seed(1)
    main()
