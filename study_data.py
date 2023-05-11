import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read the dataset
df = pd.read_csv('Seed_Data.csv')

# print the first few rows of the dataset
print(df.head())

# summarize the dataset
print(df.describe())


sns.pairplot(df)
plt.show()
