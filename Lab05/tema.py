import csv
import pymc as pm
import numpy as np

#Reads the csv file
file_reader = open("trafic.csv", "r")
trafic_data = [int(x[1]) for x in list(csv.reader(file_reader, delimiter=','))[1:]]
file_reader.close()

with pm.Model() as model:
    alpha = 1.0/np.mean(trafic_data)
    lambda1 = pm.Exponential("lambda1", alpha)
    lambda2 = pm.Exponential("lambda2", alpha)
    lambda3 = pm.Exponential("lambda3", alpha)
    tau = pm.DiscreteUniform("tau", lower=0, upper=len(trafic_data)-1)

with model:
    idx = np.arange(len(trafic_data))
    lambda_ = pm.math.switch(idx < tau, lambda2 + 10, lambda1)
    lambda_ = pm.math.switch(idx > tau, lambda3 - 10, lambda1)
    observ = pm.Poisson("obs", lambda_, observed=trafic_data)

with model:
    trace = pm.sample(10000, cores=1, step=pm.Metropolis(), return_inferencedata=False)
    print(trace['lambda1'])
    print(trace['lambda2'])
    print(trace['lambda3'])