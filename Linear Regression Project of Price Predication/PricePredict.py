import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("prediction.csv")

# Plot the data
plt.xlabel("Area")
plt.ylabel("Prices")
plt.scatter(df.area, df.prices, color="r")

# Display the scatter plot
plt.show()

# Prepare data for model training
new_df = df.drop('prices', axis='columns')
Y = df.prices

# Create a linear regression model and fit it to the data
model = LinearRegression()
model.fit(new_df, Y)

# Take user input for the area
area = float(input("Input the area for prediction: "))

# Reshape the input for prediction
predicted_price = model.predict([[area]])

# Print the predicted price
print("Predicted Price is:", predicted_price[0])
