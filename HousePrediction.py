# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:47:47 2023

@author: mooha
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import GenericUnivariateSelect, f_regression, SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from xgboost import XGBRegressor



# read files
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# make a copy from datas
train_prep = train_data.copy()
test_prep = test_data.copy()


# checking for Nan values
train_prep.info()
test_prep.info()


# describe the table
train_prep.describe()
test_prep.describe()



# dealing with Nan datas
list_none=['Alley','BsmtQual', 'BsmtCond', 'BsmtExposure' ,'BsmtFinType1', 'BsmtFinType2' ,'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual' ,'GarageCond', 'PoolQC' ,'Fence','MiscFeature']
train_prep[list_none] = train_prep[list_none].fillna('None')
train_prep['LotFrontage'] = train_prep['LotFrontage'].fillna(np.round_(np.mean(train_prep['LotFrontage'])))
train_prep['GarageYrBlt'] = train_prep['GarageYrBlt'].fillna(-1)
list_Mas = ['MasVnrType','MasVnrArea','Electrical']
train_prep[list_Mas] = train_prep[list_Mas].fillna(train_prep.mode().iloc[0])


# dealing with Nan datas
list_none_test=['Alley','BsmtQual', 'BsmtCond','MSZoning' ,'BsmtExposure' ,'BsmtFinType1', 'BsmtFinType2' ,'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual' ,'GarageCond', 'PoolQC' ,'Fence','MiscFeature']
test_prep[list_none_test] = test_prep[list_none_test].fillna('None')
test_prep['LotFrontage'] = test_prep['LotFrontage'].fillna(np.round_(np.mean(test_prep['LotFrontage'])))
test_prep['GarageYrBlt'] = test_prep['GarageYrBlt'].fillna(-1)
list_Mas = ['MasVnrType','MasVnrArea','Electrical','Utilities','Exterior1st','Exterior2nd','BsmtHalfBath','BsmtFullBath','KitchenQual','Functional','GarageCars','GarageArea','SaleType']
test_prep[list_Mas] = test_prep[list_Mas].fillna(test_prep.mode().iloc[0])
list_0 = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']
test_prep[list_0] = test_prep[list_0].fillna(0)


# remove Id feature
train_prep.drop(['Id'], axis=1, inplace=True)
test_prep.drop(['Id'], axis=1, inplace=True)


# dealing with outliers
Q1= train_prep['SalePrice'].quantile(0.99)
Q3= train_prep['SalePrice'].quantile(0.05)
train_prep = train_prep[(train_prep['SalePrice'] <= Q1) & (train_prep['SalePrice'] >= Q3)]


# Heatmap corr numerical features
corr = train_prep.select_dtypes('number').corr()
plt.figure(figsize=(24, 18))
sns.heatmap(corr , fmt = '0.2f', cmap = 'YlGnBu', annot=True)
plt.tight_layout()
plt.show()


# Remove low correlated numerical features
def remove_low_features(df, target, threshold):
    corr = df.select_dtypes('number').corr()[target]
    low_corr_features = corr[corr < threshold].index.tolist()
    df = df.drop(low_corr_features, axis=1)
    return df

train_prep = remove_low_features(train_prep, 'SalePrice', 0.2)

for feature in test_prep.keys():
    if feature not in train_prep.keys():
        test_prep = test_prep.drop(feature, axis=1)
        

        





train_cat = train_prep.select_dtypes('object')

def prepare_inputs(X):
    oe = OrdinalEncoder()
    oe.fit(X)
    X_cat = oe.transform(X)
    return X_cat

train_cat_transform = prepare_inputs(train_cat)

selector = SelectKBest(score_func=chi2, k=30)
selector.fit(train_cat_transform, train_prep['SalePrice'])
X_selected_categorical = selector.transform(train_cat_transform)

col = selector.get_support(indices=True)
selected_cols = train_cat.columns[col].tolist()


for feature in train_prep.select_dtypes('object').keys():
    if feature not in selected_cols:
        train_prep = train_prep.drop(feature, axis=1)

for feature in test_prep.select_dtypes('object').keys():
    if feature not in selected_cols:
        test_prep = test_prep.drop(feature, axis=1)












X = train_prep.drop(['SalePrice'], axis=1)
Y = train_prep['SalePrice']

SS = StandardScaler()
cols = X.select_dtypes('number').keys()
X[cols] =  SS.fit_transform(X.select_dtypes('number'))
test_prep[cols] = SS.fit_transform(test_prep.select_dtypes('number'))

X_final = pd.get_dummies(X,dtype=float, drop_first=True)
test_final = pd.get_dummies(test_prep, dtype=float, drop_first=True)

for feature in X_final:
    if feature not in test_final.keys():
        X_final = X_final.drop(feature, axis=1)
        
        
for feature in test_final:
    if feature not in X_final.keys():
        test_final = test_final.drop(feature, axis=1)

test_final_unselect = test_final.copy()








selector = GenericUnivariateSelect(score_func=f_regression, mode='percentile',param=70)

X_feature = selector.fit_transform(X_final, Y)
cols = selector.get_support(indices=True)
selected_cols = X_final.columns[cols].tolist()


X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X_feature , Y, test_size=0.2, random_state=1234)


for i in test_final:
    if i not in selected_cols:
        test_final.drop(i, axis=1, inplace=True)














X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y, test_size=0.2, random_state=1234)

# MLP model
ML = MLPRegressor(random_state=1234, max_iter=500, solver='lbfgs', alpha=0.01, hidden_layer_sizes=(5,80))

ML.fit(X_feature, Y)
Y_pred_ML = ML.predict(X_test1)
Y_pred_test_ML = ML.predict(test_final)

r2_ML = r2_score(Y_test1, Y_pred_ML)




# Ridge
ridge = Ridge(alpha=1)

ridge.fit(X_train1, Y_train1)
Y_pred_ridge = ridge.predict(X_test1)
Y_pred_test_ridge = ridge.predict(test_final)

r2_ridge = r2_score(Y_test1, Y_pred_ridge)




# catboost
cat = CatBoostRegressor(learning_rate=0.3, max_depth=5 , iterations=1000, random_state=1234)

cat.fit(X_train, Y_train)
Y_pred_cat = cat.predict(X_test)
Y_pred_test_cat = cat.predict(test_final_unselect)

r2_cat = r2_score(Y_test, Y_pred_cat)


# Xgb
xgb = XGBRegressor(learning_rate=0.2, n_estimators=40, max_depth=5)

xgb.fit(X_train, Y_train)
Y_pred_xgb = xgb.predict(X_test)
Y_pred_test_xgb = xgb.predict(test_final_unselect)

r2_xgb = r2_score(Y_test, Y_pred_xgb)





# output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': Y_pred_test_cat})
# output.to_csv('submission22.csv', index=False)
# print("Your submission was successfully saved!")



