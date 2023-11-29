import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# Importarea datelor din fisier
def importData(path):
    data = pd.read_csv(path)
    return data

# Definirea si rularea modelului
def run_model(data):
    with pm.Model() as model:
        # Distributii a priori (distributii informative slabe)
        alpha = pm.Normal('alpha', mu=0, tau=1.0/1**2)
        beta1 = pm.Normal('beta1', mu=0, tau=1.0/1**2)
        beta2 = pm.Normal('beta2', mu=0, tau=1.0/1**2)
        sigma = pm.Uniform('sigma', lower=0, upper=10)

        mu = alpha + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive'])
        price = pm.Normal('price', mu=mu, tau=1.0/sigma**2, observed=data['Price'])

        trace = pm.sample(2000, tune=1000, progressbar=True)

    return trace

# Afisarea grafica a distributiei posterioare
def plot_trace(trace):
    az.plot_trace(trace, var_names=['beta1', 'beta2', 'sigma'])
    plt.subplots_adjust(hspace=0.5)
    plt.show()

# Calcularea intervalului de incredere de 95% pentru beta1 si beta2
def compute_HDI(trace):
    hdi_beta1 = np.percentile(trace.posterior['beta1'], [2.5, 97.5])
    hdi_beta2 = np.percentile(trace.posterior['beta2'], [2.5, 97.5])
    print(f'95% HDI pentru beta1: {hdi_beta1}')
    print(f'95% HDI pentru beta2: {hdi_beta2}')

# Evaluarea importantei predictorilor (frecventa procesorului si marimea hard diskului)
# Bazat pe intervalul de incredere, putem evalua semnificatia acestor predictor
def predict_price(trace, speed, hard_drive):
    alpha = np.mean(trace.posterior['alpha'])
    beta1 = np.mean(trace.posterior['beta1'])
    beta2 = np.mean(trace.posterior['beta2'])
    sigma = np.mean(trace.posterior['sigma'])
    
    # Simulare a 5000 de extrageri din distributia posterioara
    predicted_price = alpha + beta1 * speed + beta2 * np.log(hard_drive)
    return np.random.normal(predicted_price, sigma, size=5000)

# Calcularea intervalului de incredere de 90% pentru pretul asteptat
def compute_HDI_prediction(predicted_prices):
    hdi_prediction = az.hdi(predicted_prices, hdi_prob=0.1)
    print(f'90% HDI pentru pretul asteptat: {hdi_prediction}')

# Predictia pentru un computer cu 33 MHz si 540 MB hard disk bazat pe distributia predictiva posterioara
def predict_price_distribution(trace, speed, hard_drive):
    # Simulare a 5000 de extrageri din distributia predictiva posterioara
    predicted_prices = predict_price(trace, speed, hard_drive)

    # Calcularea intervalului de predictie de 90% HDI
    hdi_prediction = az.hdi(predicted_prices, hdi_prob=0.1)
    print(f'90% HDI pentru distributia pretului anticipat: {hdi_prediction}')

def main():
    # Importarea datelor
    data = importData("Lab08/Prices.csv")

    # Rularea modelului de regresie Bayesiană și obținerea urmei
    trace = run_model(data)

    # Afisarea grafica a distributiei posterioare
    plot_trace(trace)

    # Calcularea intervalului de incredere de 95% pentru beta1 si beta2
    compute_HDI(trace)

    # Predictia pentru un computer cu 33 MHz si 540 MB hard disk
    predicted_prices = predict_price(trace, 33, 540)
    compute_HDI_prediction(predicted_prices)

    # Predictia pentru un computer cu 33 MHz si 540 MB hard disk bazat pe distributia predictiva posterioara
    predict_price_distribution(trace, 33, 540)

if __name__ == '__main__':
    main()
