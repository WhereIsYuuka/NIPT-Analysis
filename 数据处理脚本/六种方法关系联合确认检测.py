# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr, kendalltau
import pingouin as pg   # 用于距离相关

file = r"C:\Users\admin\Desktop\国赛论文\修改后附件男.xlsx"
df = pd.read_excel(file, engine="openpyxl")
num_df = df.select_dtypes(include=["number"])
y = num_df["Y染色体浓度"]
X = num_df.drop(columns=["Y染色体浓度"])

methods = {}

methods["Spearman"] = pd.Series(
    {col: spearmanr(X[col], y)[0] for col in X.columns}
).abs().sort_values(ascending=False)

methods["Kendall"] = pd.Series(
    {col: kendalltau(X[col], y)[0] for col in X.columns}
).abs().sort_values(ascending=False)

methods["MutualInfo"] = pd.Series(
    dict(zip(X.columns, mutual_info_regression(X, y, random_state=42)))
).abs().sort_values(ascending=False)

# 随机森林特征重要度
rf = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1)
rf.fit(X, y)
methods["RF_importance"] = pd.Series(
    dict(zip(X.columns, rf.feature_importances_))
).abs().sort_values(ascending=False)

# 捕捉任意非线性关系，距离相关
dcor_series = {}
for col in X.columns:
    dcor_series[col] = pg.distance_corr(X[col], y)
methods["DistanceCorr"] = pd.Series(dcor_series).abs().sort_values(ascending=False)

# 线性回归标准化系数
import statsmodels.api as sm
X_const = sm.add_constant(X)
ols = sm.OLS(y, X_const).fit()
methods["OLS_beta"] = (ols.params.drop("const")
                       .abs()
                       .sort_values(ascending=False))

# 打印
for name, ser in methods.items():
    print(f"\n{name} TOP10")
    print(ser.head(10))