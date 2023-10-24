import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson, norm

# ------------------------------------------- exercitiul 1----------------------------------------------------------------------

# definirea parametrilor
lambda_customer_arrival = 20
mean_order_time = 2
std_dev_order_time = 0.5

mean_cooking_time = 5  # alfa am ales sa fie 5
std_dev_cooking_time = 2  # la fel ca la cooking_time, valoare pe care am ales-o eu pentru calcule

# cream variabile aleatoare pentru sosirile clientilor, timpul de comanda si timpul de gatire
customer_arrivals = poisson(lambda_customer_arrival)  # variabila poisson aleatoare
order_time = norm(loc=mean_order_time, scale=std_dev_order_time)  # distributia normala

# definim distributia pentru timpul de gatit
cooking_time = norm(loc=mean_cooking_time, scale=std_dev_cooking_time)

# generam exemple aleatoare pentru distributii
num_customers = customer_arrivals.rvs(size=1)
order_processing_time = order_time.rvs(size=num_customers)
cooking_time_for_orders = cooking_time.rvs(size=num_customers)

# ------------------------------------------- exercitiul 2----------------------------------------------------------------------

total_time_constraint = 0.25  # 15 minute in ore


def calculate_probability(alpha):
    total_time = 0
    for _ in range(customer_arrivals.rvs()):
        total_time += order_time.rvs() + cooking_time.rvs()
        if total_time > total_time_constraint:
            return 0  # daca o comanda trece peste limita de timp, atunci probabilitatea este zero
    return 1


# gasirea lui alfa maxim
alpha = 0.1  # valoarea de start a lui alfa
step = 0.1  # marimea pasului lui alfa
max_alpha = alpha

while True:
    probability = calculate_probability(alpha)
    if probability < 0.95:
        break
    max_alpha = alpha
    alpha += step

print(f"Maximum α: {max_alpha:.2f}")


# ------------------------------------------- exercitiul 3----------------------------------------------------------------------


# calculam media timpului de asteptare pentru ca un client sa fie servit
def calculate_average_waiting_time():
    total_waiting_time = 0
    for i in range(len(order_processing_time)):
        total_waiting_time += order_processing_time[i] + cooking_time_for_orders[i]
    return total_waiting_time / len(order_processing_time)


average_waiting_time = calculate_average_waiting_time()
print(f"Average Waiting Time: {average_waiting_time:.2f} minutes")
