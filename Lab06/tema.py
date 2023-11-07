import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def bayesian_model(vals_Y, vlas_theta, sample_count=100):
    traces = {}
    
    for Y in vals_Y:
        for t in vlas_theta:
            with pm.Model():
                n = pm.Poisson(f'n_{Y}_{t}', mu=10)
                pm.Binomial(f'obs_{Y}_{t}', n=n, p=t, observed=Y)
                trace = pm.sample(sample_count)
                traces[(Y, t)] = trace

    return traces

def plot_custom_posteriors(traces, vals_Y, vals_theta):
    axes = plt.subplots(len(vals_Y), len(vals_theta), figsize=(12, 8))[1]
    
    for i, Y in enumerate(vals_Y):
        for j, t in enumerate(vals_theta):
            az.plot_posterior(traces[(Y, t)], var_names=[f'n_{Y}_{t}'], ax=axes[i, j])
            axes[i, j].set_title(f'Y = {Y}, t = {t}')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    vals_Y = [0, 5, 10]
    vals_theta = [0.2, 0.5]
    traces = bayesian_model(vals_Y, vals_theta)
    plot_custom_posteriors(traces, vals_Y, vals_theta)