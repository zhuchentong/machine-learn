# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==================
# STEP 1: 数据准备
# ==================

# Data Preprocessing
# 加载数据 并设置为无标题k
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header = None)

# 创建交易数组
transactions = []

# 将数据复制到交易数组
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
# ====================
# STEP 2: 生成关联规律
# ====================

from apyori import apriori
 
#  关联规律
rules = apriori(
    transactions, 
    min_support=0.003,      # 每天最少交易3次
    min_confidence=0.2,     # 买了A大于等于20%购买B 
    min_lift = 3,           # 每个规律中最低有两件商品有关联
    min_length = 2)

# ====================
# STEP 3: 打印关联规律
# ====================

results = list(rules)
myResults = [list(x) for x in results]