import numpy as np
import matplotlib.pyplot as plt

num_experiments = 100
num_flips = 10
p_stema = 0.3 # probabilitatea de a obtine stema intr-o aruncare (pentru a doua moneda)

# listele pentru a stoca rezultatele
results_ss = []  # stema stema
results_sb = []  # stema ban
results_bs = []  # ban stema
results_bb = []  # ban ban

# experiment
for _ in range(num_experiments):
    experiment_result = []
    for _ in range(num_flips):
        # aruncare prima moneda
        coin1 = np.random.choice(['s', 'b'])

        # aruncare a doua moneda masluita
        coin2 = np.random.choice(['s', 'b'], p=[p_stema, 1 - p_stema])

        # adaugarea rezultatului aruncarii celor doua monede la experiment
        experiment_result.append(coin1 + coin2)

    # rezultatele posibile în experiment
    results_ss.append(experiment_result.count('ss'))
    results_sb.append(experiment_result.count('sb'))
    results_bs.append(experiment_result.count('bs'))
    results_bb.append(experiment_result.count('bb'))

# grafic
plt.figure(figsize=(12, 8))
plt.hist(results_ss, bins=np.arange(0, num_flips + 2) - 0.5, alpha=0.5, label='SS')
plt.hist(results_sb, bins=np.arange(0, num_flips + 2) - 0.5, alpha=0.5, label='SB')
plt.hist(results_bs, bins=np.arange(0, num_flips + 2) - 0.5, alpha=0.5, label='BS')
plt.hist(results_bb, bins=np.arange(0, num_flips + 2) - 0.5, alpha=0.5, label='BB')
plt.xlabel('Numarul de rezultate posibile')
plt.ylabel('Frecventa')
plt.title('Distributiile rezultatelor în 10 aruncari de două monezi')
plt.xticks(np.arange(0, num_flips + 1))
plt.legend()
plt.grid(True)
plt.show()
