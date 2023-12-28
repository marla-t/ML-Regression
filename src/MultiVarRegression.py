import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset
dataset = pd.read_csv("housing.csv")

# Display the first few rows of the dataset
print(dataset.head())

# Select features and target variable
x = dataset[['TV', 'Radio', 'Newspaper']]
y = dataset['Sales']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# Initialize and train the Linear Regression model
mlr = LinearRegression()
mlr.fit(x_train, y_train)

# Print the intercept and coefficients
print("Intercept: ", mlr.intercept_)
print("Coefficients:", list(zip(x.columns, mlr.coef_)))

# Predictions on the test set
y_pred_mlr = mlr.predict(x_test)

# Display actual vs predicted values
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
print(mlr_diff.head())

# Evaluate the model
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
r_squared = mlr.score(x_test, y_test)  # R squared for the test set

print('R squared: {:.2f}'.format(r_squared * 100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)



plt.scatter(y_test, y_pred_mlr, label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Line of Best Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values with Line of Best Fit')
plt.legend()
plt.show()

plt.scatter(x_test['TV'], y_test, label='Actual')
plt.scatter(x_test['TV'], y_pred_mlr, label='Predicted')
plt.plot(x_test['TV'], mlr.predict(x_test), color='red', label='Line of Best Fit')
plt.xlabel('TV Ad Spending')
plt.ylabel('Sales')
plt.title('TV Ad Spending vs Sales with Line of Best Fit')
plt.legend()
plt.show()