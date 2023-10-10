import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

# parametrii
alpha_server1 = 4
lambda_server1 = 1/3

alpha_server2 = 4
lambda_server2 = 1/2

alpha_server3 = 5
lambda_server3 = 1/2

alpha_server4 = 5
lambda_server4 = 1/3

# probabilitate server
prob_server1 = 0.25
prob_server2 = 0.25
prob_server3 = 0.30
prob_server4 = 0.20

# intervalul de timp
x = np.linspace(0, 10, 1000)  # Alegem un interval suficient de mare

# calcul densitatea de probabilitate pentru fiecare server
pdf_server1 = stats.gamma.pdf(x, alpha_server1, scale=1/lambda_server1)
pdf_server2 = stats.gamma.pdf(x, alpha_server2, scale=1/lambda_server2)
pdf_server3 = stats.gamma.pdf(x, alpha_server3, scale=1/lambda_server3)
pdf_server4 = stats.gamma.pdf(x, alpha_server4, scale=1/lambda_server4)

# calcul densitatea de probabilitate totala ponderata
pdf_total = (
    prob_server1 * pdf_server1 +
    prob_server2 * pdf_server2 +
    prob_server3 * pdf_server3 +
    prob_server4 * pdf_server4
)

# grafic
plt.figure(figsize=(8, 6))
plt.plot(x, pdf_total, label='Densitatea de probabilitate a lui X')
plt.xlabel('Timp (milisecunde)')
plt.ylabel('Densitatea de probabilitate')
plt.title('Densitatea de probabilitate a timpului necesar pentru servirea unui client')
plt.legend()
plt.grid(True)
plt.show()
