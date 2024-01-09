import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#ex1
def posterior_grid(grid_points=50, heads=6, tails=9, prior_type=0):
    grid = np.linspace(0, 1, grid_points)
    if prior_type == 0:
        prior = np.repeat(1/grid_points, grid_points)
    elif prior_type == 1:
        prior = (grid<= 0.5).astype(int)
    elif prior_type == 2:
        prior = abs(grid - 0.5)
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

data = np.repeat([0, 1], (100, 30))  
points = 50  
h = data.sum()
t = len(data) - h 
for prior_type in range(3):
    grid, posterior = posterior_grid(points, h, t, prior_type)
    plt.subplot(1, 3, prior_type + 1)
    plt.plot(grid, posterior, 'o-')
    plt.title(f'heads = {h}, tails = {t}')
    plt.yticks([])
    plt.xlabel('θ')

plt.tight_layout()
plt.show()

#ex2
def estimate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum() * 4 / N
    error = abs((pi - np.pi) / np.pi) * 100
    return error

N_vals = [100, 1000, 10000]
sim_count = 100  
errors = np.zeros((len(N_vals), sim_count))
for i in range(len(N_vals)):
    for j in range(sim_count):
        errors[i, j] = estimate_pi(N_vals[i])
mean_errors = np.mean(errors, axis=1)
std_errors = np.std(errors, axis=1)

plt.errorbar(N_vals, mean_errors, yerr=std_errors, fmt='o', capsize=5)
plt.xscale('log')
plt.xlabel('Number of Points (N)')
plt.ylabel('Error estimation for π (%)')
plt.title('Error estimation for π vs. Number of Points')
plt.tight_layout()
plt.show()
