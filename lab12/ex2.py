import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(N):
    points = np.random.rand(N, 2)
    inside_circle = np.sum(points**2, axis=1) < 1
    pi_estimate = 4 * np.sum(inside_circle) / N
    return pi_estimate

# definirea a diferitelor valori pentru N si numarul de iteratii
N_values = [100, 1000, 10000]
iterations = 100
errors = {N: [] for N in N_values}

# simulare
for N in N_values:
    for _ in range(iterations):
        pi_estimate = estimate_pi(N)
        error = abs(np.pi - pi_estimate)
        errors[N].append(error)

# calculare medie si deviatie standard a erorilor
mean_errors = [np.mean(errors[N]) for N in N_values]
std_errors = [np.std(errors[N]) for N in N_values]

# rezultat
plt.errorbar(N_values, mean_errors, yerr=std_errors, fmt='o')
plt.xlabel('Number of Points (N)')
plt.ylabel('Error in Estimation of π')
plt.title('Error in Estimation of π as a Function of N')
plt.xscale('log')
plt.show()
