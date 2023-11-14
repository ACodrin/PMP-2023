import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm

def load_data():
    data = pd.read_csv("auto-mpg.csv")
    return data.loc[:, "mpg"], data.loc[:, "horsepower"]

def plot_scatter(x, y):
    plt.scatter(x, y)
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.show()

def build_and_sample_model(hp, mpg):
    with pm.Model():
        alpha = pm.Normal('α', mu=mpg.mean(), sigma=1)
        beta = pm.Normal('β', mu=0, sigma=1)
        mu = alpha + beta * hp
        epsilon = pm.HalfCauchy('ε', 5)
        pm.Normal('mpg_predicted', mu=mu, sigma=epsilon, observed=mpg)
        idata = pm.sample(2000, return_inferencedata=True)
    return idata

def plot_posterior_predictions(hp, mpg, idata):
    plt.plot(hp, mpg, 'C0.')
    
    posterior_g = idata.posterior.stack(samples={"chain", "draw"})
    alpha_m = posterior_g['α'].mean().item()
    beta_m = posterior_g['β'].mean().item()
    
    range(0, posterior_g.samples.size, 1)
    plt.plot(hp, alpha_m + beta_m * hp[:, None], c='gray', alpha=0.5)
    plt.plot(hp, alpha_m + beta_m * hp, c='k', label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
    
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()
    plt.show() 

if __name__ == '__main__':
    mpg, horsepower = load_data()
    plot_scatter(horsepower, mpg)
    
    idata = build_and_sample_model(horsepower, mpg)
    
    plot_posterior_predictions(horsepower, mpg, idata)
