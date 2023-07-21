# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:02:39 2023

@author: Yigitalp
"""
# Import the libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Import the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Dataset information
train_df.info()
test_df.info()
train_df_desc = train_df.describe()
test_df_desc = test_df.describe()
train_df_head = train_df.head()
test_df_head = test_df.head()

# Drop unnecessary columns (Id and variables having 1K+ missing values)
sum_na_train = train_df.isna().sum()
train_df = train_df.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis = 1)
sum_na_test = test_df.isna().sum()
test_df = test_df.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis = 1)

# Fill missing values - numerical
# for train set
train_df['LotFrontage'] = train_df['LotFrontage'].fillna(round(train_df['LotFrontage'].mean()))
train_df['GarageYrBlt'] = train_df['GarageYrBlt'].fillna(round(train_df['GarageYrBlt'].mean()))
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(round(train_df['MasVnrArea'].mean()))
#for test set
test_df['LotFrontage'] = test_df['LotFrontage'].fillna(round(test_df['LotFrontage'].mean()))
test_df['GarageYrBlt'] = test_df['GarageYrBlt'].fillna(round(test_df['GarageYrBlt'].mean()))
test_df['MasVnrArea'] = test_df['MasVnrArea'].fillna(round(test_df['MasVnrArea'].mean()))
test_df['BsmtFullBath'] = test_df['BsmtFullBath'].fillna(round(test_df['BsmtFullBath'].mean()))
test_df['BsmtHalfBath'] = test_df['BsmtHalfBath'].fillna(round(test_df['BsmtHalfBath'].mean()))
test_df['BsmtFinSF1'] = test_df['BsmtFinSF1'].fillna(round(test_df['BsmtFinSF1'].mean()))
test_df['BsmtFinSF2'] = test_df['BsmtFinSF2'].fillna(round(test_df['BsmtFinSF2'].mean()))
test_df['BsmtUnfSF'] = test_df['BsmtUnfSF'].fillna(round(test_df['BsmtUnfSF'].mean()))
test_df['TotalBsmtSF'] = test_df['TotalBsmtSF'].fillna(round(test_df['TotalBsmtSF'].mean()))
test_df['GarageCars'] = test_df['GarageCars'].fillna(round(test_df['GarageCars'].mean()))
test_df['GarageArea'] = test_df['GarageArea'].fillna(round(test_df['GarageArea'].mean()))

# Fill missing values - categorical
# for train set
sum_na_train = train_df.isna().sum()
headers_train = sum_na_train[sum_na_train != 0].index.tolist()
for col in headers_train:
    train_df[col] = train_df[col].fillna('None')
train_df.info()
# for test set
sum_na_test = test_df.isna().sum()
headers_test = sum_na_test[sum_na_test != 0].index.tolist()
for col in headers_test:
    test_df[col] = test_df[col].fillna('None')
test_df.info()

# Encode categorical variables
# for train set
category_train = []
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        category_train.append(col)

for col in category_train:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].values)    
# for test set
category_test = []
for col in test_df.columns:
    if test_df[col].dtype == 'object':
        category_test.append(col)

for col in category_test:
    le = LabelEncoder()
    test_df[col] = le.fit_transform(test_df[col].values)
    
# Split train and test datasets
correlation = train_df.corr()
correlation = correlation['SalePrice']
correlation = correlation.reset_index()
correlation = correlation.rename(columns={'index': 'Variables'})
correlation = correlation.drop(index=len(correlation)-1)
correlation['SalePrice'] = correlation['SalePrice'].abs()
features = correlation[correlation['SalePrice'] > 0.047]
features = features['Variables'].tolist()

X = train_df[features]
y = train_df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Scale feature
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Reduce dimensionality
#pca = PCA(n_components = 10)
#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)

# Train the regression model on train set
rf = RandomForestRegressor(random_state = 1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Calculate accuracy score
rf_test_mae = mean_absolute_error(y_pred, y_test)

# Predict full data
rf_full_data = RandomForestRegressor(random_state = 1)
rf_full_data.fit(X, y)
test_X = test_df[features]
test_preds = rf_full_data.predict(test_X)

# Create submission csv
id_column = pd.read_csv('test.csv', usecols = ['Id']).values
id_column = pd.DataFrame(id_column, columns = ['Id'])
                       
submission = pd.DataFrame({
        'Id': id_column['Id'],
        'SalePrice': test_preds
    })

submission.to_csv('submission.csv', index=False)




    
    
        

    