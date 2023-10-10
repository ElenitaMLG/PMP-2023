import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# parametrii distributiei exponentiale
λ1 = 4  
λ2 = 6  

# probabilitate primul mecanic
probabilitate_primul_mecanic = 0.4

# nr de valori generate
numar_valori = 10000

# generarea timpilor de servire folosind distribuția exponentiala
timp_servire = np.random.choice([1 / λ1, 1 / λ2], size=numar_valori, p=[probabilitate_primul_mecanic, 1 - probabilitate_primul_mecanic])

# calcularea mediei și deviatie standard
media_x = np.mean(timp_servire)
deviatia_standard_x = np.std(timp_servire)

print("Media lui X:", media_x)
print("Deviatia standard a lui X:", deviatia_standard_x)

x = np.linspace(0, max(timp_servire), 1000)

pdf_primul_mecanic = stats.expon.pdf(x, scale=1 / λ1)
pdf_al_doilea_mecanic = stats.expon.pdf(x, scale=1 / λ2)
pdf_combinat = probabilitate_primul_mecanic * pdf_primul_mecanic + (1 - probabilitate_primul_mecanic) * pdf_al_doilea_mecanic

# afisare grafic
plt.figure(figsize=(10, 6))
plt.plot(x, pdf_primul_mecanic, label='Primul Mecanic')
plt.plot(x, pdf_al_doilea_mecanic, label='Al Doilea Mecanic')
plt.plot(x, pdf_combinat, label='Combinat (Probabilitate ponderată)')
plt.title('Densitatea distribuției lui X')
plt.xlabel('Timp de servire (ore)')
plt.ylabel('Densitate')
plt.legend()
plt.grid(True)
plt.show()
