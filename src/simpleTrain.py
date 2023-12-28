import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples (houses)
num_samples = 1000

# Features
area = np.random.normal(1500, 300, num_samples)  # House area in square feet
bedrooms = np.random.randint(2, 6, num_samples)  # Number of bedrooms
bathrooms = np.random.uniform(1, 4, num_samples)  # Number of bathrooms
garage_spaces = np.random.randint(0, 3, num_samples)  # Number of garage spaces

# Generating target variable (price)
price = 50000 + 200 * area + 10000 * bedrooms + 15000 * bathrooms + 5000 * garage_spaces + np.random.normal(0, 50000, num_samples)

# Create a DataFrame
housing_data = pd.DataFrame({
    'Area': area,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'GarageSpaces': garage_spaces,
    'Price': price
})


# Display the first few rows of the dataset
print(housing_data.head())

# Save the dataset to a CSV file
housing_data.to_csv("housing_price_dataset.csv", index=False)

