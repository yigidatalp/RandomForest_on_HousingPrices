# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:02:39 2023

@author: Yigitalp
"""
# Import helpful libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
pd.set_option('display.max_rows', 80)

# Load the data, and separate the target
home_data = pd.read_csv('train.csv')
y = home_data['SalePrice']

# Check the number of missing values and eliminate columns having more than 1K missing values
sum_of_nan = home_data.isna().sum()
home_data = home_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)

#%%

# Convert FireplaceQu
home_data['FireplaceQu'] = home_data['FireplaceQu'].fillna('FireplaceQuNA')
# home_data_pivot = home_data.groupby(by = 'FireplaceQu').mean()
# home_data_pivot = home_data_pivot['SalePrice']

home_data['FireplaceQu'] = home_data['FireplaceQu'].replace(
    ['Po', 'FireplaceQuNA', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5])

#%%

# Fill missing values with average for LotFrontage
home_data['LotFrontage'] = home_data['LotFrontage'].fillna(
    round(home_data['LotFrontage'].mean()))

#%%

# Fill missing values with NA for GarageType and convert it to numerical columns
home_data['GarageType'] = home_data['GarageType'].fillna('GarageTypeNA')
# home_data_pivot = home_data.groupby(by = 'GarageType').mean()
# home_data_pivot = home_data_pivot['SalePrice']

home_data['GarageType'] = home_data['GarageType'].replace(
    ['GarageTypeNA', 'CarPort', 'Detchd', '2Types', 'Basment', 'Attchd', 'BuiltIn'], [0, 1, 2, 3, 4, 5, 6])

#%%

# Fill missing values with average for GarageYrBlt
home_data['GarageYrBlt'] = home_data['GarageYrBlt'].fillna(
    round(home_data['GarageYrBlt'].mean()))

#%%

# Convert GarageFinish
home_data['GarageFinish'] = home_data['GarageFinish'].fillna('GarageFinishNA')
# home_data_pivot = home_data.groupby(by = 'GarageFinish').mean()
# home_data_pivot = home_data_pivot['SalePrice']

home_data['GarageFinish'] = home_data['GarageFinish'].replace(
    ['GarageFinishNA', 'Unf', 'RFn', 'Fin'], [0, 1, 2, 3])

#%%

# Convert GarageQual
home_data['GarageQual'] = home_data['GarageQual'].fillna('GarageQualNA')
# home_data_pivot = home_data.groupby(by = 'GarageQual').mean()
# home_data_pivot = home_data_pivot['SalePrice']

home_data['GarageQual'] = home_data['GarageQual'].replace(
    ['Po', 'GarageQualNA', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5])

#%%

# Convert GarageCond
home_data['GarageCond'] = home_data['GarageCond'].fillna('GarageCondNA')
# home_data_pivot = home_data.groupby(by = 'GarageCond').mean()
# home_data_pivot = home_data_pivot['SalePrice']

home_data['GarageCond'] = home_data['GarageCond'].replace(
    ['GarageCondNA', 'Po', 'Fa', 'Ex', 'Gd', 'TA'], [0, 1, 2, 3, 4, 5])

#%%

# Convert BsmtExposure
home_data['BsmtExposure'] = home_data['BsmtExposure'].fillna('BsmtExposureNA')
# home_data_pivot = home_data.groupby(by = 'BsmtExposure').mean()
# home_data_pivot = home_data_pivot['SalePrice']

home_data['BsmtExposure'] = home_data['BsmtExposure'].replace(
    ['BsmtExposureNA', 'No', 'Mn', 'Av', 'Gd'], [0, 1, 2, 3, 4])

#%%

# Convert BsmtFinType2
home_data['BsmtFinType2'] = home_data['BsmtFinType2'].fillna('BsmtFinType2NA')
# home_data_pivot = home_data.groupby(by = 'BsmtFinType2').mean()
# home_data_pivot = home_data_pivot['SalePrice']

home_data['BsmtFinType2'] = home_data['BsmtFinType2'].replace(
    ['BsmtFinType2NA', 'BLQ', 'LwQ', 'Rec', 'GLQ', 'Unf', 'ALQ'], [0, 1, 2, 3, 4, 5, 6])

#%%

# Convert BsmtQual
home_data['BsmtQual'] = home_data['BsmtQual'].fillna('BsmtQualNA')
# home_data_pivot = home_data.groupby(by = 'BsmtQual').mean()
# home_data_pivot = home_data_pivot['SalePrice']

home_data['BsmtQual'] = home_data['BsmtQual'].replace(
    ['BsmtQualNA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5])

#%%

# Convert BsmtCond
home_data['BsmtCond'] = home_data['BsmtCond'].fillna('BsmtCondNA')
# home_data_pivot = home_data.groupby(by = 'BsmtCond').mean()
# home_data_pivot = home_data_pivot['SalePrice']

home_data['BsmtCond'] = home_data['BsmtCond'].replace(
    ['Po', 'BsmtCondNA', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5])

#%%

# Convert BsmtFinType1
home_data['BsmtFinType1'] = home_data['BsmtFinType1'].fillna('BsmtFinType1NA')
# home_data_pivot = home_data.groupby(by = 'BsmtFinType1').mean()
# home_data_pivot = home_data_pivot['SalePrice']

home_data['BsmtFinType1'] = home_data['BsmtFinType1'].replace(
    ['BsmtFinType1NA', 'Rec', 'BLQ', 'LwQ', 'ALQ', 'Unf', 'GLQ'], [0, 1, 2, 3, 4, 5, 6])

#%%

# Fill missing values with NA for MasVnrType  and convert it to numerical columns
home_data['MasVnrType'] = home_data['MasVnrType'].fillna('MasVnrTypeNA')
# home_data_pivot = home_data.groupby(by = 'MasVnrType').mean()
# home_data_pivot = home_data_pivot['SalePrice']

home_data['MasVnrType'] = home_data['MasVnrType'].replace(
    ['BrkCmn', 'None', 'BrkFace', 'MasVnrTypeNA', 'Stone'], [0, 1, 2, 3, 4])

#%%

# Fill missing values with average for MasVnrArea
home_data['MasVnrArea'] = home_data['MasVnrArea'].fillna(
    round(home_data['MasVnrArea'].mean()))

#%%

# Convert Electrical
home_data['Electrical'] = home_data['Electrical'].fillna('ElectricalNA')
# home_data_pivot = home_data.groupby(by = 'Electrical').mean()
# home_data_pivot = home_data_pivot['SalePrice']

home_data['Electrical'] = home_data['Electrical'].replace(
    ['Mix', 'FuseP', 'FuseF', 'FuseA', 'ElectricalNA', 'SBrkr'], [0, 1, 2, 3, 4, 5])

#%%

# Convert objects to numeric
for col in home_data.columns:
    if (home_data[col].dtype == 'object'):
        home_data[col] = home_data[col].astype('category')
        home_data[col] = home_data[col].cat.codes


correlation = home_data.corr()
correlation = correlation['SalePrice']
correlation = correlation.reset_index()
correlation = correlation.rename(columns={'index': 'Variables'})
correlation = correlation.drop(index=len(correlation)-1)

pos_features = correlation[correlation['SalePrice'] > 0.05]
neg_features = correlation[correlation['SalePrice'] < -0.05]
features = pd.concat([pos_features['Variables'], neg_features['Variables']])
features = features.tolist()

# Select columns corresponding to features, and preview the data
X = home_data[features]
for col in X.columns:
    X[col] = (X[col] - X[col].min()) / (X[col].max() - X[col].min())

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=1)

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X, y)
#%%

test_data = pd.read_csv('test.csv')
sum_of_nan_test = test_data.isna().sum()

# Check the number of missing values and eliminate columns having more than 1K missing values
test_data = test_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)

#%%

# Convert FireplaceQu
test_data['FireplaceQu'] = test_data['FireplaceQu'].fillna('FireplaceQuNA')
# home_data_pivot = home_data.groupby(by = 'FireplaceQu').mean()
# home_data_pivot = home_data_pivot['SalePrice']

test_data['FireplaceQu'] = test_data['FireplaceQu'].replace(
    ['Po', 'FireplaceQuNA', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5])

#%%

# Fill missing values with average for LotFrontage
test_data['LotFrontage'] = test_data['LotFrontage'].fillna(
    round(test_data['LotFrontage'].mean()))

#%%

# Fill missing values with average for GarageYrBlt
test_data['GarageYrBlt'] = test_data['GarageYrBlt'].fillna(
    round(test_data['GarageYrBlt'].mean()))

#%%

# Convert GarageFinish
test_data['GarageFinish'] = test_data['GarageFinish'].fillna('GarageFinishNA')
# home_data_pivot = home_data.groupby(by = 'GarageFinish').mean()
# home_data_pivot = home_data_pivot['SalePrice']

test_data['GarageFinish'] = test_data['GarageFinish'].replace(
    ['GarageFinishNA', 'Unf', 'RFn', 'Fin'], [0, 1, 2, 3])

#%%

# Convert GarageQual
test_data['GarageQual'] = test_data['GarageQual'].fillna('GarageQualNA')
# home_data_pivot = home_data.groupby(by = 'GarageQual').mean()
# home_data_pivot = home_data_pivot['SalePrice']

test_data['GarageQual'] = test_data['GarageQual'].replace(
    ['Po', 'GarageQualNA', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5])

#%%

# Convert GarageCond
test_data['GarageCond'] = test_data['GarageCond'].fillna('GarageCondNA')
# home_data_pivot = home_data.groupby(by = 'GarageCond').mean()
# home_data_pivot = home_data_pivot['SalePrice']

test_data['GarageCond'] = test_data['GarageCond'].replace(
    ['GarageCondNA', 'Po', 'Fa', 'Ex', 'Gd', 'TA'], [0, 1, 2, 3, 4, 5])

#%%

# Fill missing values with NA for GarageType and convert it to numerical columns
test_data['GarageType'] = test_data['GarageType'].fillna('GarageTypeNA')
# home_data_pivot = home_data.groupby(by = 'GarageType').mean()
# home_data_pivot = home_data_pivot['SalePrice']

test_data['GarageType'] = test_data['GarageType'].replace(
    ['GarageTypeNA', 'CarPort', 'Detchd', '2Types', 'Basment', 'Attchd', 'BuiltIn'], [0, 1, 2, 3, 4, 5, 6])

#%%

# Convert BsmtCond
test_data['BsmtCond'] = test_data['BsmtCond'].fillna('BsmtCondNA')
# home_data_pivot = home_data.groupby(by = 'BsmtCond').mean()
# home_data_pivot = home_data_pivot['SalePrice']

test_data['BsmtCond'] = test_data['BsmtCond'].replace(
    ['Po', 'BsmtCondNA', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5])

#%%

# Convert BsmtQual
test_data['BsmtQual'] = test_data['BsmtQual'].fillna('BsmtQualNA')
# home_data_pivot = home_data.groupby(by = 'BsmtQual').mean()
# home_data_pivot = home_data_pivot['SalePrice']

test_data['BsmtQual'] = test_data['BsmtQual'].replace(
    ['BsmtQualNA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5])

#%%

# Convert BsmtExposure
test_data['BsmtExposure'] = test_data['BsmtExposure'].fillna('BsmtExposureNA')
# home_data_pivot = home_data.groupby(by = 'BsmtExposure').mean()
# home_data_pivot = home_data_pivot['SalePrice']

test_data['BsmtExposure'] = test_data['BsmtExposure'].replace(
    ['BsmtExposureNA', 'No', 'Mn', 'Av', 'Gd'], [0, 1, 2, 3, 4])

#%%

# Convert BsmtFinType1
test_data['BsmtFinType1'] = test_data['BsmtFinType1'].fillna('BsmtFinType1NA')
# home_data_pivot = home_data.groupby(by = 'BsmtFinType1').mean()
# home_data_pivot = home_data_pivot['SalePrice']

test_data['BsmtFinType1'] = test_data['BsmtFinType1'].replace(
    ['BsmtFinType1NA', 'Rec', 'BLQ', 'LwQ', 'ALQ', 'Unf', 'GLQ'], [0, 1, 2, 3, 4, 5, 6])

#%%

# Convert BsmtFinType2
test_data['BsmtFinType2'] = test_data['BsmtFinType2'].fillna('BsmtFinType2NA')
# home_data_pivot = home_data.groupby(by = 'BsmtFinType2').mean()
# home_data_pivot = home_data_pivot['SalePrice']

test_data['BsmtFinType2'] = test_data['BsmtFinType2'].replace(
    ['BsmtFinType2NA', 'BLQ', 'LwQ', 'Rec', 'GLQ', 'Unf', 'ALQ'], [0, 1, 2, 3, 4, 5, 6])

#%%

# Fill missing values with NA for MasVnrType and convert it to numerical columns
test_data['MasVnrType'] = test_data['MasVnrType'].fillna('MasVnrTypeNA')
# home_data_pivot = home_data.groupby(by = 'MasVnrType').mean()
# home_data_pivot = home_data_pivot['SalePrice']

test_data['MasVnrType'] = test_data['MasVnrType'].replace(
    ['BrkCmn', 'None', 'BrkFace', 'MasVnrTypeNA', 'Stone'], [0, 1, 2, 3, 4])

#%%

# Fill missing values with average for MasVnrArea
test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(
    round(test_data['MasVnrArea'].mean()))

#%%

# Fill missing values with NA for MSZoning and convert it to numerical columns
test_data['MSZoning'] = test_data['MSZoning'].fillna('MSZoningNA')
test_data['MSZoning'] = test_data['MSZoning'].replace(
    ['MSZoningNA', 'C (all)', 'RM', 'RH', 'RL', 'FV'], [0, 1, 2, 3, 4, 5])

#%%

# Fill missing values with NA for Utilities and convert it to numerical columns
test_data['Utilities'] = test_data['Utilities'].fillna('UtilitiesNA')
test_data['Utilities'] = test_data['Utilities'].replace(
    ['UtilitiesNA', 'ELO', 'NoSeWa', 'NoSewr', 'AllPub'], [0, 1, 2, 3, 4])

#%%

# Fill missing values with average for BsmtFullBath
test_data['BsmtFullBath'] = test_data['BsmtFullBath'].fillna(
    round(test_data['MasVnrArea'].mean()))

#%%

# Fill missing values with average for BsmtHalfBath
test_data['BsmtHalfBath'] = test_data['BsmtHalfBath'].fillna(
    round(test_data['BsmtHalfBath'].mean()))

#%%

# Fill missing values with NA for Functional and convert it to numerical columns
test_data['Functional'] = test_data['Functional'].fillna('FunctionalNA')
test_data['Functional'] = test_data['Functional'].replace(
    ['FunctionalNA', 'Maj2', 'Sev', 'Min2', 'Min1', 'Maj1', 'Mod', 'Typ'], [0, 1, 2, 3, 4, 5, 6, 7])

#%%

# Fill missing values with NA for Exterior1st and convert it to numerical columns
test_data['Exterior1st'] = test_data['Exterior1st'].fillna('Exterior1stNA')
test_data['Exterior1st'] = test_data['Exterior1st'].replace(
    ['Exterior1stNA', 'BrkComm', 'AsphShn', 'CBlock', 'AsbShng', 'MetalSd', 'Wd Sdng', 'WdShing', 'Stucco', 'HdBoard', 'Plywood', 'BrkFace', 'VinylSd', 'CemntBd', 'Stone', 'ImStucc'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

#%%

# Fill missing values with NA for Exterior2nd and convert it to numerical columns
test_data['Exterior2nd'] = test_data['Exterior2nd'].fillna('Exterior2ndNA')
test_data['Exterior2nd'] = test_data['Exterior2nd'].replace(
    ['Exterior2ndNA', 'CBlock', 'AsbShng', 'Brk Cmn', 'AsphShn', 'Wd Sdng', 'MetalSd', 'Stucco', 'Stone', 'Wd Shng', 'HdBoard', 'Plywood', 'BrkFace', 'VinylSd', 'CmentBd', 'ImStucc', 'Other'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

#%%

# Fill missing values with average for BsmtFinSF1
test_data['BsmtFinSF1'] = test_data['BsmtFinSF1'].fillna(
    round(test_data['BsmtFinSF1'].mean()))

#%%

# Fill missing values with average for BsmtFinSF2
test_data['BsmtFinSF2'] = test_data['BsmtFinSF2'].fillna(
    round(test_data['BsmtFinSF2'].mean()))

#%%

# Fill missing values with average for BsmtUnfSF
test_data['BsmtUnfSF'] = test_data['BsmtUnfSF'].fillna(
    round(test_data['BsmtUnfSF'].mean()))

#%%

# Fill missing values with average for BsmtUnfSF
test_data['TotalBsmtSF'] = test_data['TotalBsmtSF'].fillna(
    round(test_data['TotalBsmtSF'].mean()))

#%%

# Fill missing values with NA for KitchenQual and convert it to numerical columns
test_data['KitchenQual'] = test_data['KitchenQual'].fillna('KitchenQualNA')
test_data['KitchenQual'] = test_data['KitchenQual'].replace(
    ['KitchenQualNA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5])

#%%

# Fill missing values with average for GarageCars
test_data['GarageCars'] = test_data['GarageCars'].fillna(
    round(test_data['GarageCars'].mean()))

#%%

# Fill missing values with average for GarageArea
test_data['GarageArea'] = test_data['GarageArea'].fillna(
    round(test_data['GarageArea'].mean()))

#%%

# Fill missing values with NA for SaleType and convert it to numerical columns
test_data['SaleType'] = test_data['SaleType'].fillna('SaleTypeNA')
test_data['SaleType'] = test_data['SaleType'].replace(
    ['SaleTypeNA', 'Oth', 'ConLD', 'ConLw', 'COD', 'WD', 'ConLI', 'CWD', 'Con', 'New'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

#%%

# Convert objects to numeric
for col in test_data.columns:
    if (test_data[col].dtype == 'object'):
        test_data[col] = test_data[col].astype('category')
        test_data[col] = test_data[col].cat.codes

for col in test_data.columns:
    if col != 'Id':
        test_data[col] = (test_data[col] - test_data[col].min()) / \
            (test_data[col].max() - test_data[col].min())

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

# make predictions which we will submit.
test_preds = rf_model_on_full_data.predict(test_X)

# Run the code to save predictions in the format used for competition scoring
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
