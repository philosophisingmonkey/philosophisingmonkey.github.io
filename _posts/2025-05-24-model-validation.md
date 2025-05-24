---
layout: post
title:  "Introduction to Model validation"
date:   2025-05-24 10:00:00
---

# Getting the Data (regression tasks)
We can create synthetic data depending on tasks: for regression(make_regression) and for classification (make_classification).
Regression data refers to the dataset used to train and evaluate regression models, which are designed to predict continuous numerical outcomes based on input features. In regression tasks, the target variable (dependent variable) is continuous rather than categorical.
Input Features (Independent Variables): These can be continuous or categorical variables that serve as predictors.
Target Variable (Dependent Variable): A continuous numeric value that the model aims to predict.


```python
from sklearn.datasets import make_regression
import pandas as pd

X,y = make_regression(n_samples =10, n_features =1, noise =10, random_state = 42)
df = pd.DataFrame(X, columns =['Features'])
df['Target'] = y
print(df)
```

       Features     Target
    0 -0.138264 -11.754819
    1 -0.469474 -15.569654
    2  0.767435   6.529811
    3 -0.234137 -19.495954
    4 -0.234153   3.906958
    5  0.542560 -11.531110
    6  0.496714   3.317702
    7  1.523030  37.196182
    8  1.579213  24.433571
    9  0.647689 -14.348895


## Split the data into train and test data


```python
from sklearn.model_selection import train_test_split

X = df.drop('Target', axis = 1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


```

# Train the model


```python
from sklearn.linear_model import LinearRegression

# Linear Regression: Models linear relationships between inputs and continuous output.
model = LinearRegression()
model.fit(X_train, y_train)
print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)
```

    Model coefficients: [21.58579783]
    Model intercept: -9.867887773598884


# Validate model


```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate regression performance

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

```

    Mean Squared Error (MSE): 9.84
    Root Mean Squared Error (RMSE): 3.14
    Mean Absolute Error (MAE): 2.32
    R² Score: 0.98


# Analysis of scores

- Mean square error - Average of the squared differences between predicted and actual values. Lower MSE is better — indicates predictions are closer to actual values.
- RMSE - Represents the standard deviation of the residuals (prediction errors). Easier to interpret than MSE because it’s in the same units as the target.Lower RMSE is better — smaller average prediction error.
- Mean Absolute Error (MAE) - Average of the absolute differences between predicted and actual values. If RMSE is much larger than MAE, it suggests the presence of outliers or large errors.
- R² Score  - Proportion of variance in the dependent variable explained by the model. 1.0 = perfect prediction. 0.0 = model performs as well as predicting the mean. Negative values = model is worse than predicting the mean Higher R² (closer to 1) is better — model explains more variance. Values above 0.7 are generally considered good, but this depends on the problem domain. Negative values indicate poor model fit.


