import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
import arviz as az

time1 = stats.expon(scale=1/4)
time2 = stats.expon(scale=1/6)
samples = 10000

result = []

for i in range(samples):
    mecanic1 = np.random.rand() < 0.4

    if(mecanic1 == True):
        time = time1.rvs()
    elif(mecanic1 != True):
        time = time2.rvs()

    result.append(time)

az.plot_posterior({'x':result})
plt.show() 

print("Media X:", np.mean(result))
print("Dev. standard pt X:", np.std(result))