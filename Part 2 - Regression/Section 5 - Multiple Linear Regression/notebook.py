# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:37:46 2021
@author: zhuchentong
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### 导入数据
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import OneHotEncoder;
from sklearn.compose import ColumnTransformer 
# 对地址列实现虚拟编码
onehotEncoder = OneHotEncoder() 
ct = ColumnTransformer([("OneHot", OneHotEncoder(),[3])], remainder="passthrough")
X = ct.fit_transform(X)

# 避免虚拟变量陷阱
X = X[:,1:]

### 生成训练集与测试集
from sklearn.model_selection import train_test_split
# 使用固定值生成训练集/测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# 多元线性回归模型
from sklearn.linear_model import LinearRegression
# 创建回归器
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# 回归器预测
y_pred = regressor.predict(X_test)

# 反向淘汰
import statsmodels.regression.linear_model as sm
X_train = np.append(arr=np.ones((40,1),dtype=float),values=X_train,axis=1)

# P_Value Top 0.05
# step 1 - all in
# X_opt = np.array(object=X_train[:,[0,1,2,3,4,5]],dtype=float)

# step 2 - remove x2
# X_opt = np.array(object=X_train[:,[0,1,3,4,5]],dtype=float)

# step 3 - remove x1
# X_opt = np.array(object=X_train[:,[0,3,4,5]],dtype=float)

# step 4 - remove x4
# X_opt = np.array(object=X_train[:,[0,3,5]],dtype=float)

# step 5 - remove x2
X_opt = np.array(object=X_train[:,[0,3]],dtype=float)

regressor_OLS = sm.OLS(endog=y_train,exog=X_opt).fit()
summary = regressor_OLS.summary()