import numpy as np

# nr de iteratii pentru metoda Monte Carlo
num_iteratii = 10000

# nr de repetari pentru a calcula media si deviatia standard
num_repetari = 30

# parametrii pentru distributiile geometrice
param_X = 0.3
param_Y = 0.5

# lista pentru a stoca rezultatele fiecarei aproximari
aproximatii = []

for _ in range(num_repetari):
    # generam variabile aleatoare X si Y
    X = np.random.geometric(param_X, num_iteratii)
    Y = np.random.geometric(param_Y, num_iteratii)

    # calculam nr de cazuri în care X > Y^2
    numar_cazuri_favorabile = np.sum(X > Y**2)

    # calculam probabilitatea aproximativa pentru aceasta iteratie
    prob_aprox = numar_cazuri_favorabile / num_iteratii

    # adaugam rezultatul la lista de aproximatii
    aproximatii.append(prob_aprox)

# calculam media si deviatia standard a aproximatiilor
media_aproximatiilor = np.mean(aproximatii)
deviatia_standard = np.std(aproximatii)

print(media_aproximatiilor, deviatia_standard)
