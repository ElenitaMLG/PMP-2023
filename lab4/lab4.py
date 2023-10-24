from scipy.stats import poisson, norm

# ------------------------------------------- exercitiul 1----------------------------------------------------------------------

# definirea parametrilor
lambda_customer_arrival = 20
mean_order_time = 2
std_dev_order_time = 0.5

mean_cooking_time = 5 # alfa am ales sa fie 5
std_dev_cooking_time = 2 # la fel ca la cooking_time, valoare pe care am ales-o eu pentru calcule

# cream variabile aleatoare pentru sosirile clientilor, timpul de comanda si timpul de gatire
customer_arrivals = poisson(lambda_customer_arrival)    # variabila poisson aleatoare
order_time = norm(loc=mean_order_time, scale=std_dev_order_time)    # distributia normala

# definim distributia pentru timpul de gatit
cooking_time = norm(loc=mean_cooking_time, scale=std_dev_cooking_time)

# generam exemple aleatoare pentru distributii
num_customers = customer_arrivals.rvs(size=1)
order_processing_time = order_time.rvs(size=num_customers)
cooking_time_for_orders = cooking_time.rvs(size=num_customers)
