"""
LINEAR REGRESSION
=================
The simplest ML model — fit a straight line to data.

WHY IT MATTERS FOR ML:
- Foundation of ALL neural networks (a neuron IS a linear regression + activation)
- Introduces the core ML loop: predict → measure error → update weights
- Same math scales up to deep learning

WHAT IS LINEAR REGRESSION?
- Given data points (x, y), find the best line: y = mx + b
- "Best" = minimizes the total error between predictions and actual values
- m = slope (weight), b = intercept (bias)

    y
    |         *
    |       *    <- actual data points
    |     /
    |   / *      <- best fit line
    | / *
    |/*___________x

HOW IT FINDS THE BEST LINE:
- Error for one point:  error = predicted - actual
- Cost function (MSE):  MSE = (1/n) * sum(errors^2)
- Goal: find m and b that minimize MSE

    Cost
    |
    | \
    |  \
    |   \___/    <- we want to find this minimum
    |       \
    |________\__ m (slope)

TWO WAYS TO FIND THE BEST LINE:
1. Normal Equation — solve directly with math (what Scikit-learn uses)
2. Gradient Descent — step-by-step optimization (what neural networks use)

KEY FORMULAS:
- Prediction:    y_pred = m * x + b
- MSE:           (1/n) * sum((y_pred - y_actual)^2)
- R-squared:     1 - (sum of squared errors / sum of squared total)
                 R^2 = 1.0 means perfect fit, R^2 = 0.0 means no better than guessing the mean

SCIKIT-LEARN USAGE:
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)        # find best m and b
    predictions = model.predict(X_test) # use the line
    model.coef_                        # the slope(s) — weights
    model.intercept_                   # the intercept — bias
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# === EXAMPLE: Predict house price from square footage ===

# Generate fake data: price = 200 * sqft + 50000 + noise
np.random.seed(42)
sqft = np.random.randint(500, 3000, size=100).astype(float)
price = 200 * sqft + 50000 + np.random.randn(100) * 15000

# Reshape X to 2D (sklearn requires 2D input)
X = sqft.reshape(-1, 1)  # shape: (100, 1)
y = price

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Results
print("=== Linear Regression: House Prices ===")
print(f"Slope (weight):     {model.coef_[0]:.2f}  (actual: 200)")
print(f"Intercept (bias):   {model.intercept_:.2f}  (actual: 50000)")

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE:                {mse:.2f}")
print(f"R-squared:          {r2:.4f}")
print(f"\nSample predictions vs actual:")
for i in range(5):
    print(f"  {X_test[i][0]:.0f} sqft -> predicted ${y_pred[i]:,.0f} | actual ${y_test[i]:,.0f}")


# ============================================================
# PROBLEM 1: Fit a linear regression to study hours vs exam score
# ============================================================
# Given: hours studied and exam scores for 50 students
# Task:
#   1. Create a LinearRegression model
#   2. Fit it on X_train, y_train
#   3. Predict on X_test
#   4. Print the R-squared score
#
# Data is ready below — just build and evaluate the model

# np.random.seed(7)
# hours = np.random.uniform(1, 10, size=50)
# scores = 8 * hours + 20 + np.random.randn(50) * 5

# X = hours.reshape(-1, 1)
# y = scores

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Your code below:
# model = LinearRegression()
# model.fit(X_train, y_train)


# y_pred = model.predict(X_test)

# mse = mean_squared_error(y_test ,y_pred)
# r2 = r2_score(y_test ,y_pred)

# print(f"R2 {r2}")


# ============================================================
# PROBLEM 2: Predict temperature from altitude
# ============================================================
# In reality, temperature drops ~6.5°C per 1000m of altitude
# Task:
#   1. Fit a LinearRegression on the training data
#   2. Print the slope (model.coef_[0]) — what does it tell you?
#   3. Print the R-squared score

np.random.seed(15)
altitude = np.random.uniform(0, 4000, size=80)
temperature = 25 - 0.0065 * altitude + np.random.randn(80) * 2

X = altitude.reshape(-1, 1)
y = temperature

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Your code below:


model = LinearRegression()

model.fit(X_train, y_train)

print(model.coef_[0])
y_pred = model.predict(X_test)   # add this line before mse/r2


mse = mean_squared_error(y_test ,y_pred)
r2 = r2_score(y_test ,y_pred)

print(f"R2 {r2}")


# ============================================================
# PROBLEM 3: Multiple Linear Regression (2 features)
# ============================================================
# Now X has TWO columns: sqft and bedrooms
# The model learns: price = w1*sqft + w2*bedrooms + bias
#
# Task:
#   1. Fit a LinearRegression on X_train, y_train
#   2. Print model.coef_ — you'll see TWO weights (one per feature)
#   3. Print the R-squared score
#
# Hint: X is already 2D so no need to reshape

np.random.seed(99)
sqft = np.random.randint(600, 3500, size=100).astype(float)
bedrooms = np.random.randint(1, 6, size=100).astype(float)
price = 150 * sqft + 20000 * bedrooms + 30000 + np.random.randn(100) * 20000

X = np.column_stack([sqft, bedrooms])  # shape: (100, 2)
y = price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Your code below:
model = LinearRegression()

model.fit(X_train ,y_train)

print(model.coef_)

y_pred = model.predict(X_test)   # add this line before mse/r2


mse = mean_squared_error(y_test ,y_pred)
r2 = r2_score(y_test ,y_pred)

print(f"R2 {r2}")