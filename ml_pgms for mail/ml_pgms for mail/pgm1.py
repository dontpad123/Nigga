import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(r'housing.csv')


df.head()


df.shape


df.info()


df.nunique()

df.isnull().sum()


df.duplicated().sum()


df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)


df.describe().T


Numerical = df.select_dtypes(include=[np.number]).columns
print(Numerical)


for col in Numerical:
    plt.figure(figsize=(5, 3))
    df[col].plot(kind='hist', bins=60, edgecolor='black')
    plt.title(col)
    plt.ylabel('Frequency')
    plt.show()


for col in Numerical:
    plt.figure(figsize=(5, 3))
    sns.boxplot(df[col], color="black")
    plt.ylabel(col)
    plt.title(col)
    plt.show()
