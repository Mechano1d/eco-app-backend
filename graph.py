import pandas
import matplotlib.pyplot as plt
import pandas as pd
import numpy
import xarray
from pandas import array

df = pd.read_csv("energy-consumption-by-source-and-country.csv")
print(df)
x = df['Year']
df = df.drop(columns=['Entity', 'Code', 'Year'])
print(df.columns)
fig, ax = plt.subplots(1, 1)
fff = df.to_numpy()
print(x.to_numpy())
print(df.to_numpy())
ax.stackplot(x=x.to_numpy(), y=df.to_numpy())
plt.show()
