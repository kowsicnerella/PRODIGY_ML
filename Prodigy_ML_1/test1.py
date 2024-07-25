import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
train_data = pd.read_csv('train.csv')

# Preprocess the data
numeric_columns = train_data.select_dtypes(include=['number']).columns
train_data[numeric_columns] = train_data[numeric_columns].fillna(train_data[numeric_columns].median())

# Selecting relevant features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
X = train_data[features]
y = train_data['SalePrice']

# Visualize the distribution of Sale Prices
plt.figure(figsize=(10, 6))
sns.histplot(train_data['SalePrice'], kde=True, bins=30)
plt.title('Distribution of Sale Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()

# Visualize the relationship between GrLivArea and Sale Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x=train_data['GrLivArea'], y=train_data['SalePrice'])
plt.title('Sale Price vs. Above Ground Living Area (GrLivArea)')
plt.xlabel('Above Ground Living Area (GrLivArea)')
plt.ylabel('Sale Price')
plt.show()

# Visualize the relationship between BedroomAbvGr and Sale Price
plt.figure(figsize=(10, 6))
sns.boxplot(x=train_data['BedroomAbvGr'], y=train_data['SalePrice'])
plt.title('Sale Price vs. Number of Bedrooms Above Ground (BedroomAbvGr)')
plt.xlabel('Number of Bedrooms Above Ground (BedroomAbvGr)')
plt.ylabel('Sale Price')
plt.show()

# Visualize the relationship between FullBath and Sale Price
plt.figure(figsize=(10, 6))
sns.boxplot(x=train_data['FullBath'], y=train_data['SalePrice'])
plt.title('Sale Price vs. Number of Full Bathrooms (FullBath)')
plt.xlabel('Number of Full Bathrooms (FullBath)')
plt.ylabel('Sale Price')
plt.show()

# Visualize the correlation matrix for numeric columns only
plt.figure(figsize=(12, 10))
numeric_columns = train_data.select_dtypes(include=['number']).columns
correlation_matrix = train_data[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
