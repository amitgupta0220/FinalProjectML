import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# read the dataset
df = pd.read_csv('Seed_Data.csv')
# df.columns = ['A', 'P', 'C', 'LK', 'WK', 'A_Coef', 'LKG', 'target']

# print the first few rows of the dataset
print(df.head())

# summarize the dataset
print(df.describe())


sns.pairplot(df)
plt.show()
