import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

def main():
    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt('C:\\Users\\anech\\Desktop\\dummy.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]

    order = 5
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    with pm.Model() as model_linear:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=10)
        ϵ = pm.HalfNormal('ϵ', 5)
        µ = α + β * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=µ, sigma=ϵ, observed=y_1s)
        idata_linear = pm.sample(10,tune=10, return_inferencedata=True)

    with pm.Model() as model_sd_10:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=10, shape=order)
        ϵ = pm.HalfNormal('ϵ', 5)
        µ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=µ, sigma=ϵ, observed=y_1s)
        idata_sd_10 = pm.sample(10,tune=10, return_inferencedata=True)

    with pm.Model() as model_sd_100:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=100, shape=order)
        ϵ = pm.HalfNormal('ϵ', 5)
        µ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=µ, sigma=ϵ, observed=y_1s)
        idata_sd_100 = pm.sample(10, tune=10, return_inferencedata=True)

    sd = np.array([10, 0.1, 0.1, 0.1, 0.1])
    with pm.Model() as model_sd_np_arr:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=sd, shape=order)
        ϵ = pm.HalfNormal('ϵ', 5)
        µ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=µ, sigma=ϵ, observed=y_1s)
        idata_sd_np_arr = pm.sample(10, tune=10, return_inferencedata=True)

    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)

    α_l_post = idata_linear.posterior['α'].mean(("chain", "draw")).values
    β_l_post = idata_linear.posterior['β'].mean(("chain", "draw")).values
    y_l_post = α_l_post + β_l_post * x_new
    plt.plot(x_new, y_l_post, 'C1', label='linear model')

    α_p_post = idata_sd_10.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_sd_10.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'sd=10')

    α_p_post = idata_sd_100.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_sd_100.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C4', label=f'sd=100')

    α_p_post = idata_sd_np_arr.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_sd_np_arr.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C5', label=f'sd=[10, 0.1, 0.1, 0.1, 0.1]')

    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()