import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


import warnings
warnings.filterwarnings('ignore')

 data = pd.read_csv(r"Boston housing dataset.csv")

data.head()

data.shape

data.info()

data.nunique()

data.MEDV.unique()

data.isnull().sum()

data.duplicated().sum()

df=data.copy()

df['CRIM'].fillna(df['CRIM'].mean(), inplace=True)
df['ZN'].fillna(df['ZN'].mean(), inplace=True)
df['CHAS'].fillna(df['CHAS'].mode()[0], inplace=True)
df['INDUS'].fillna(df['INDUS'].mean(), inplace=True)
df['AGE'].fillna(df['AGE'].median(), inplace=True)  # Median is often preferred for
df['LSTAT'].fillna(df['LSTAT'].median(), inplace=True)

df.isnull().sum()

df.head()

df.describe().T

for i in df.columns:
    plt.figure(figsize=(6,3))
    
    plt.subplot(1, 2, 1)
    df[i].hist(bins=20, color='b',edgecolor='black')
    plt.title(f'Histogram of {i}')
    plt.xlabel(i)
    plt.ylabel('Frequency')


    
    plt.subplot(1, 2, 2)
    plt.boxplot(df[i], vert=False)
    plt.title(f'Boxplot of {i}')
    
    plt.show()

corr = df.corr(method='pearson')
plt.figure(figsize=(10, 8))

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.xticks(rotation=90, ha='right')
plt.yticks(rotation=0)
plt.title("Correlation Matrix Heatmap")
plt.show()

X = df.drop('MEDV', axis=1)  # All columns except 'MEDV'
y = df['MEDV']  # Target variable

 # Scale the features
scale = StandardScaler()
X_scaled  = scale.fit_transform(X)

 # Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled , y, test_size=0.2,random_state=45)

# Initialize the linear regression model
model = LinearRegression()

 # Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
 # Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
 # Calculate R-squared value
r2 = r2_score(y_test, y_pred)



print("Mean Squared Error:",mse)
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2:.2f}')

