import pandas as pd
import numpy as np
import math
from sklearn import linear_model

df = pd.read_csv('homeprices.csv')
print(df);

df.bedrooms =  df.bedrooms.fillna(df.bedrooms.median())
print(df);

model = linear_model.LinearRegression()
model.fit(df.drop('price',axis='columns'),df.price)

price = model.predict([[3000,3,40]])

print(price,model.coef_,model.intercept_)
 