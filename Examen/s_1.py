import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

def main():
    Titanic = pd.read_csv('PMP-2023/Examen/Titanic.csv')
    y = Titanic["Survived"]
    print("Date nebalansate: "+str(len(y[y==1]))+" "+str(len(y[y==0])))
    #Stergem indici la intamplare
    Index = np.random.choice(np.flatnonzero(y==0), size=len(y[y==0])-len(y[y==1]), replace=False)
    Titanic = Titanic.drop(labels=Index)

    #Definirea vlaorilor (pentru Survived, Pclass, Age)
    y = Titanic["Survived"]
    x_Pclass = Titanic["Pclass"].values
    x_Pclass_mean = x_Pclass.mean()
    x_Pclass_std = x_Pclass.std()
    x_Age = Titanic["Age"].values
    x_Age_mean = x_Age.mean()
    x_Age_std = x_Age.mean()

    #Se standardizeaza datele
    x_Pclass = (x_Pclass-x_Pclass_mean)/x_Pclass_std
    x_Age = (x_Age-x_Age_mean)/x_Age_std
    X = np.column_stack((x_Pclass, x_Age))

    with pm.Model() as surv_model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=1, shape = 2)
        X_shared = pm.MutableData('x_shared',X)
        mu = pm.Deterministic('μ',alpha + pm.math.dot(X_shared, beta))
        theta = pm.Deterministic("theta", pm.math.sigmoid(mu))
        bd = pm.Deterministic("bd", -alpha/beta[1] - beta[0]/beta[1] * x_Pclass)
        y_pred = pm.Bernoulli("y_pred", p=theta, observed=y)
        idata = pm.sample(2000, return_inferencedata = True)

    #Afisarea datelor
    idx = np.argsort(x_Pclass)
    bd = idata.posterior["bd"].mean(("chain", "draw"))[idx]
    plt.scatter(x_Pclass, x_Age, c=[f"C{x}" for x in y])
    plt.xlabel("Pclass")
    plt.ylabel("Age")
    plt.show()
    idx = np.argsort(x_Pclass)
    bd = idata.posterior["bd"].mean(("chain", "draw"))[idx]
    plt.scatter(x_Pclass, x_Age, c=[f"C{x}" for x in y])
    plt.plot(x_Pclass[idx], bd, color = 'k')
    az.plot_hdi(x_Pclass, idata.posterior["bd"], color ='k')
    plt.xlabel("Pclass")
    plt.ylabel("Age")
    plt.show()

    #Crearea intervalului 90% HDI pt un pasager de 30 ani la clasa II
    obs_std1 = [(2-x_Pclass_mean)/x_Pclass_std,(30-x_Age_mean)/x_Age_std]
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    posterior_g = idata.posterior.stack(samples={"chain", "draw"})
    mu = posterior_g['alpha'] + posterior_g['beta'][0]*obs_std1[0] + posterior_g['beta'][1]*obs_std1[1]
    theta = sigmoid(mu)
    az.plot_posterior(theta.values, hdi_prob=0.9)

if __name__ == '__main__':
    main()