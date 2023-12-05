import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

def import_data(path):
    data = pd.read_csv(path)
    admission = data["Admission"].tolist()
    x = data[["GRE", "GPA"]].values
    return admission, x

def calculate_probabilities(idata, x):
    beta0 = idata.posterior['β0'][1]
    beta = idata.posterior['β'][1]
    mu = beta0 + np.dot(x, beta)
    probabilities = 1 / (1 + np.exp(-mu))
    hdi = pm.stats.hdi(probabilities, hdi_prob=0.9)
    hdi_probabilities = probabilities[(probabilities >= hdi[0]) & (probabilities <= hdi[1])]
    hdi_probabilities.sort()
    return hdi_probabilities

def plot_gre_gpa(idata, gre, gpa, x):
    probabilities = calculate_probabilities(idata, x)
    plt.figure()
    plt.plot(probabilities)
    plt.title(f'GRE = {gre} GPA = {gpa}')
    plt.show()

def plot_scatter_with_hdi(idata, x, admission):
    idx = np.argsort(x[:, 1])
    bd = idata.posterior['bd'].mean(("chain", "draw"))[idx]
    plt.scatter(x[:, 1], x[:, 0], c=[f'C{x}' for x in admission])
    plt.plot(x[:, 1][idx], bd, color='k')
    az.plot_hdi(x[:, 1], idata.posterior['bd'], color='k')
    plt.xlabel("GPA")
    plt.ylabel("GRE")
    plt.show()

def main():
    admission, x = import_data("Lab09/Admission.csv")
    rng = np.random.default_rng(100)

    with pm.Model() as model_1:
        beta0 = pm.Normal('β0', mu=0, sigma=10)
        beta = pm.Normal('β', mu=0, sigma=2, shape=2)
        mu = beta0 + pm.math.dot(x, beta)
        theta = pm.Deterministic('θ', 1 / (1 + pm.math.exp(-mu)))
        bd = pm.Deterministic('bd', -beta0 / beta[0] - beta[1] / beta[0] * x[:, 1])
        yl = pm.Bernoulli('yl', p=theta, observed=admission)
        idata = pm.sample(2000, random_seed=rng, return_inferencedata=True)

    plot_scatter_with_hdi(idata, x, admission)
    plot_gre_gpa(idata, 550, 3.5, x)
    plot_gre_gpa(idata, 500, 3.2, x)
    plot_gre_gpa(idata, 1000, 4.5, x)

if __name__ == '__main__':
    main()
