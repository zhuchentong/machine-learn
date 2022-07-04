# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:09:41 2021

@author: zhuchentong
"""

### SVM 支持向量机
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### 导入数据
dataset =pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

### -------------------------------

### 生成训练集与测试集
from sklearn.model_selection import train_test_split
# 使用固定值生成训练集/测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

### -------------------------------

# 特征缩放
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

### -------------------------------

### 创建SVM分类器
from sklearn.svm import SVC
# 使用linear作为kernel
classifier = SVC(kernel='linear',random_state=0)

### 拟合分类器
classifier.fit(X_train,y_train)

### 预测测试集
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
# 混淆矩阵评估模型
# 评估分类器
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
###  65   3
###  8    24

# 图像显示分类结果
# 预测边界
# 训练集
from matplotlib.colors import ListedColormap

X_set,y_set = X_train,y_train

X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01),np.arange(start = X_set[:,1].min() - 1, stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.75,cmap=ListedColormap(('red','green')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X1.min(),X2.max())

for i,j in  enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1],c=ListedColormap(('orange','blue'))(i),label=j)
    
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

###  测试集
from matplotlib.colors import ListedColormap

X_set,y_set = X_test,y_test

X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01),np.arange(start = X_set[:,1].min() - 1, stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.75,cmap=ListedColormap(('red','green')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X1.min(),X2.max())

for i,j in  enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1],c=ListedColormap(('orange','blue'))(i),label=j)
    
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()