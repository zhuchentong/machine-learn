# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:23:14 2021

@author: zhuchentong
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### 导入数据
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# 创建线性回归模型
from sklearn.linear_model import LinearRegression

line_regression = LinearRegression()
line_regression.fit(X,y)

# 创建多项式回归模型
from sklearn.preprocessing import PolynomialFeatures

ploy_regression = PolynomialFeatures(degree = 4)
X_ploy = ploy_regression.fit_transform(X)

# 使用多项式矩阵集合线性回归模型
line_regression_2 = LinearRegression()
line_regression_2.fit(X_ploy,y)

# 图像绘制
# 线性回归模型图像
plt.scatter(X,y,color='red')
plt.plot(X,line_regression.predict(X),color='blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# 多项式回归模型图像
plt.scatter(X,y,color='red')
plt.plot(X,line_regression_2.predict(X_ploy),color='green')
plt.title('Polynomal Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# 让多项式回归模型图像变得平滑
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,line_regression_2.predict(ploy_regression.fit_transform(X_grid)),color='black')
plt.title('Polynomal Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# 预测线性回归模型结果
line_pred = line_regression.predict(np.array(6.5).reshape(1,-1))
ploy_pred = line_regression_2.predict(ploy_regression.fit_transform(np.array(6.5).reshape(1,-1)))