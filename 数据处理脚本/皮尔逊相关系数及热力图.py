# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file = r"C:\Users\admin\Desktop\国赛论文\修改后附件男.xlsx"
df = pd.read_excel(r"C:\Users\admin\Desktop\国赛论文\修改后附件男.xlsx", engine="openpyxl")

# 只保留数值型列
numeric_df = df.select_dtypes(include=["number"])   # 含 int、float、bool、0/1

# 确保目标列在里面
if "Y染色体浓度" not in numeric_df.columns:
    raise KeyError("请检查列名，找不到 'Y染色体浓度'")

# 计算Pearson相关矩阵
corr = numeric_df.corr(method="pearson")

# 输出与Y染色体浓度最相关的前N个变量
topN = (corr["Y染色体浓度"]
        .drop("Y染色体浓度")      # 去掉自己和自己的 1.0
        .abs()
        .sort_values(ascending=False))
print("Pearson |r| 与 Y染色体浓度最相关 Top10")
print(topN)

# 画热力图
plt.figure(figsize=(14, 12))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=.5)
plt.title("Pearson 相关系数矩阵（未预处理）")
plt.tight_layout()

# 保存到本地
save_path = r"C:\Users\admin\Desktop\国赛论文\pearson_heatmap.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"热力图已保存至：{save_path}")
plt.close()