import numpy as np
import matplotlib.pyplot as plt

# definirea gridului
grid_points = 1000
grid = np.linspace(0, 1, grid_points)

# definirea diferitelor priori de distributie
prior_uniform = np.ones(grid_points)  # distribtuei unifroma
prior_half = (grid <= 0.5).astype(int)  # proiri prima jumatate
prior_abs = abs(grid - 0.5)  # priori bazati pe distanda de la 0.5

# likelihood 
data = 6   # numar de succese
trials = 9 # mi,ar de incercari
likelihood = grid**data * (1 - grid)**(trials - data)

# posteriors
posterior_uniform = likelihood * prior_uniform
posterior_half = likelihood * prior_half
posterior_abs = likelihood * prior_abs

# normalizarea
posterior_uniform /= np.sum(posterior_uniform)
posterior_half /= np.sum(posterior_half)
posterior_abs /= np.sum(posterior_abs)

# rezultatul
plt.plot(grid, posterior_uniform, label='Uniform Prior')
plt.plot(grid, posterior_half, label='Half Prior')
plt.plot(grid, posterior_abs, label='Abs Prior')
plt.title('Posterior distributions with different priors')
plt.xlabel('Parameter value')
plt.ylabel('Probability')
plt.legend()
plt.show()
