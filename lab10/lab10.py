import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Functie pentru generarea de date polinomiale
def generate_data(order, n_samples=100, noise_std=0.1):
    np.random.seed(0)
    X = np.linspace(0, 1, n_samples)
    y = sum([np.random.randn() * (X ** i) for i in range(order + 1)]) + np.random.normal(0, noise_std, n_samples)
    return X, y

# Functie pentru a face fit and plot regresiei polinomiale
def fit_and_plot(X, y, order, title):
    X_poly = PolynomialFeatures(degree=order).fit_transform(X[:, np.newaxis])
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    plt.scatter(X, y, label='Data')
    plt.plot(X, y_pred, color='red', label=f'Order {order} fit')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    return model, mean_squared_error(y, y_pred)

# 1: Polinomiala de ordin 5
X, y = generate_data(order=5)
model_5, mse_5 = fit_and_plot(X, y, 5, 'Polynomial Regression (Order 5)')

# 1b: Repetam cu standardul devierii diferit pentru distributia beta

# Generam date cu sd=100
X_100, y_100 = generate_data(order=5, noise_std=100)
model_100, mse_100 = fit_and_plot(X_100, y_100, 5, 'Polynomial Regression (Order 5, sd=100)')

# Generam date cu sd=[10, 0.1, 0.1, 0.1, 0.1]
noises = [10, 0.1, 0.1, 0.1, 0.1]
y_custom_sd = sum([np.random.randn() * noises[i] * (X ** i) for i in range(5)]) + np.random.normal(0, 0.1, 100)
model_custom_sd, mse_custom_sd = fit_and_plot(X, y_custom_sd, 5, 'Polynomial Regression (Order 5, Custom sd)')

# 2: Marim numarul de data points la 500 si repetam
X_500, y_500 = generate_data(order=5, n_samples=500)
model_500, mse_500 = fit_and_plot(X_500, y_500, 5, 'Polynomial Regression (Order 5, 500 Data Points)')

# 3: Implementam un model cubic cu ordin 3
X_cubic, y_cubic = generate_data(order=3)
model_cubic, mse_cubic = fit_and_plot(X_cubic, y_cubic, 3, 'Cubic Regression (Order 3)')
