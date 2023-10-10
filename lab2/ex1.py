import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parametrii distribuției exponențiale
λ1 = 4  # Rate pentru primul mecanic
λ2 = 6  # Rate pentru al doilea mecanic

# Probabilitatea de a fi servit de primul mecanic
probabilitate_primul_mecanic = 0.4

# Numărul de valori generate
numar_valori = 10000

# Generarea timpilor de servire folosind distribuția exponențială
timp_servire = np.random.choice([1 / λ1, 1 / λ2], size=numar_valori, p=[probabilitate_primul_mecanic, 1 - probabilitate_primul_mecanic])

# Calcularea mediei și deviației standard
media_x = np.mean(timp_servire)
deviatia_standard_x = np.std(timp_servire)

print("Media lui X:", media_x)
print("Deviatia standard a lui X:", deviatia_standard_x)

# Crearea un set de valori pentru x
x = np.linspace(0, max(timp_servire), 1000)

# Calcularea densității pentru fiecare valoare din x folosind distribuția exponențială corespunzătoare
pdf_primul_mecanic = stats.expon.pdf(x, scale=1 / λ1)
pdf_al_doilea_mecanic = stats.expon.pdf(x, scale=1 / λ2)

# Calcularea densității combinate bazate pe probabilitatea de a fi servit de primul mecanic
pdf_combinat = probabilitate_primul_mecanic * pdf_primul_mecanic + (1 - probabilitate_primul_mecanic) * pdf_al_doilea_mecanic

# Afisare grafic
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
