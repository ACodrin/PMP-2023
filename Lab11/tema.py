import numpy as np
import arviz as az
import pymc as pm

def main():
    #ex1
    clusters = 3
    n_cluster = [200, 110, 190]
    n_total = sum(n_cluster)
    means = [5, 0, -5]
    std_devs = [2, 2, 2]
    mix = np.random.normal(np.repeat(means, n_cluster),
    np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix))

    #ex2
    mix = np.array(mix)
    clusters = [2, 3, 4]
    models = []
    idatas = []
    for cluster in clusters:
        with pm.Model() as model:
            weights = pm.Dirichlet("weights", np.ones(cluster))
            means = pm.Normal("means", mu=np.linspace(mix.min(), mix.max(), cluster), sigma=10, shape=cluster)
            std_devs = pm.Uniform("std_devs", lower=0, upper=10, shape=cluster)
            data_likelihood = pm.NormalMixture("data_likelihood", w=weights, mu=means, sigma=std_devs, observed=mix)
            idata = pm.sample(2000, return_inferencedata=True)
            idatas.append(idata)
            models.append(model)

    #ex3
    az.plot_compare( az.compare({"models[0]": idatas[0], "models[1]": idatas[1], "models[2]": idatas[2]},method="BB-pseudo-BMA",ic="waic",scale="deviance",))
    az.plot_compare(az.compare({"models[0]": idatas[0], "models[1]":idatas[1], "model[2]": idatas[2]},method="BB-pseudo-BMA",ic="loo",scale="deviance",))

if __name__ == "__main__":
    main()
