import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# read the dataset
df = pd.read_csv('Seed_Data.csv', header=None)
df.columns = ['A', 'P', 'C', 'LK', 'WK', 'A_Coef', 'LKG', 'target']

# print the first few rows of the dataset
print(df.head())

# summarize the dataset
print(df.describe())


# plot a histogram for each column in the dataframe
for col in df.columns:
    sns.histplot(df[col], kde=False)
    plt.title(col)
    plt.show()
