import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

def main():
    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt('C:\\Users\\anech\\Desktop\\dummy.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]

    order = 3
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

    with pm.Model() as model_quadratic:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=10, shape=2)
        ϵ = pm.HalfNormal('ϵ', 5)
        µ = α + pm.math.dot(β, x_1s[0:2])
        y_pred = pm.Normal('y_pred', mu=µ, sigma=ϵ, observed=y_1s)
        idata_quadratic = pm.sample(10,tune=10, return_inferencedata=True)

    with pm.Model() as model_order_3:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=10, shape=3)
        ϵ = pm.HalfNormal('ϵ', 5)
        µ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=µ, sigma=ϵ, observed=y_1s)
        idata_order_3 = pm.sample(10, tune=10, return_inferencedata=True)

    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)

    α_l_post = idata_linear.posterior['α'].mean(("chain", "draw")).values
    β_l_post = idata_linear.posterior['β'].mean(("chain", "draw")).values
    y_l_post = α_l_post + β_l_post * x_new
    plt.plot(x_new, y_l_post, 'C1', label='linear model')

    α_p_post = idata_quadratic.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_quadratic.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s[0:2])
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label='quadratic')

    α_p_post = idata_order_3.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_order_3.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C3', label='order 3')

    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()

    pm.compute_log_likelihood(idata_linear, model=model_linear)
    linear = az.waic(idata_linear, scale="deviance")

    pm.compute_log_likelihood(idata_quadratic, model=model_quadratic)
    quadratic = az.waic(idata_quadratic, scale="deviance")

    pm.compute_log_likelihood(idata_order_3, model=model_order_3)
    order_3 = az.waic(idata_order_3, scale="deviance")

    print(linear)
    print(quadratic)
    print(order_3)

    cmd_df = az.compare({"model_linear": idata_linear, "model_quadratic": idata_quadratic, "model_order_3": idata_order_3},
                        method='BB-pseudo-BMA', ic='waic', scale='deviance')
    print(cmd_df.to_string())

    linear_loo = az.loo(idata_linear, pointwise=True)
    order_2_loo = az.loo(idata_quadratic, pointwise=True)
    order_3_loo = az.loo(idata_order_3, pointwise=True)

    print(linear_loo)
    print(order_2_loo)
    print(order_3_loo)

    cmd_df_loo = az.compare({"model_linear": idata_linear, "model_quadratic": idata_quadratic, "model_order_3": idata_order_3},
                            method='BB-pseudo-BMA', ic='loo', scale='deviance')
    print(cmd_df_loo.to_string())

if __name__ == '__main__':
    main()