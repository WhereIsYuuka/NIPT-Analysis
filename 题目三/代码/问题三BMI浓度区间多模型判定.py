import matplotlib
matplotlib.use('Agg')
import os, math, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeRegressor
from lifelines.statistics import multivariate_logrank_test
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns

PATIENT_TSTAR = r"C:\Users\admin\Desktop\国赛论文\问题三_full_results_fixed\per_patient_tstar.csv"
SURV_FILE = r"C:\Users\admin\Desktop\国赛论文\问题三_full_results_fixed\per_patient_survival.csv"
OUTDIR = r"C:\Users\admin\Desktop\国赛论文\grouping_comparison"
MIN_GROUP_SIZE = 25
N_BOOT = 200
RANDOM_STATE = 42
os.makedirs(OUTDIR, exist_ok=True)


p = pd.read_csv(PATIENT_TSTAR)
surv = pd.read_csv(SURV_FILE)
# 保证有 patient, bmi_mean, t_star_clean, time_to_event, event
if 'bmi_mean' not in p.columns and 'bmi_for_group' in p.columns:
    p['bmi_mean'] = p['bmi_for_group']
# 如有需要合并生存信息
p = p.merge(surv[['patient','time_to_event','event']], on='patient', how='left')

# 去除无 bmi 或 t_star 信息的患者
p0 = p[['patient','bmi_mean','t_star_clean','time_to_event','event']].copy()
p0['t_star_clean'] = pd.to_numeric(p0['t_star_clean'], errors='coerce')
p0['bmi_mean'] = pd.to_numeric(p0['bmi_mean'], errors='coerce')
p0 = p0.dropna(subset=['bmi_mean']).reset_index(drop=True)

# 分组统计辅助函数
def group_summary(df, group_col='bmi_group'):
    rows=[]
    for g, grp in df.groupby(group_col):
        n = grp['patient'].nunique()
        med = float(grp['t_star_clean'].dropna().median()) if grp['t_star_clean'].dropna().size>0 else np.nan
        p90 = float(np.nanpercentile(grp['t_star_clean'].dropna(),90)) if grp['t_star_clean'].dropna().size>0 else np.nan
        rows.append({'method':group_col,'group':g,'n':n,'median_tstar':med,'p90_tstar':p90})
    return pd.DataFrame(rows)

# 方法结果收集器
all_results = []

# 临床分组
def clinical_bin(b):
    if b < 18.5: return '<18.5'
    if b < 24: return '18.5-24'
    if b < 28: return '24-28'
    if b < 32: return '28-32'
    return '>=32'
p0['bmi_clin'] = p0['bmi_mean'].apply(clinical_bin)
all_results.append(('clinical', group_summary(p0, 'bmi_clin')))

# 自定义密集分组
def custom_bin(b):
    if b < 28: return '<28'
    if b < 30: return '28-30'
    if b < 32: return '30-32'
    if b < 35: return '32-35'
    return '>=35'
p0['bmi_custom'] = p0['bmi_mean'].apply(custom_bin)
all_results.append(('custom', group_summary(p0, 'bmi_custom')))

# 分位数分组
p0['bmi_q4'] = pd.qcut(p0['bmi_mean'], 4, labels=['Q1','Q2','Q3','Q4'])
all_results.append(('quantile4', group_summary(p0, 'bmi_q4')))

# KMeans 聚类分组
def kmeans_bins(df, k=4):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE).fit(df[['bmi_mean']])
    labels = km.labels_
    # 将聚类中心按中心值排序
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)
    mapping = {old: f'KM_{i+1}' for i,old in enumerate(order)}
    mapped = [mapping[x] for x in labels]
    return mapped, centers
p0['bmi_km_labels'], centers = kmeans_bins(p0, k=4)
all_results.append(('kmeans4', group_summary(p0, 'bmi_km_labels')))

# GMM
gmm = GaussianMixture(n_components=4, random_state=RANDOM_STATE).fit(p0[['bmi_mean']])
p0['bmi_gmm'] = gmm.predict(p0[['bmi_mean']])
# 按均值排序映射标签
comp_means = gmm.means_.flatten()
order = np.argsort(comp_means)
mapg = {old: f'GMM_{i+1}' for i,old in enumerate(order)}
p0['bmi_gmm_label'] = p0['bmi_gmm'].map(mapg)
all_results.append(('gmm4', group_summary(p0, 'bmi_gmm_label')))

# 决策树回归分组
dt_df = p0.dropna(subset=['t_star_clean'])
if dt_df.shape[0] > 0:
    # 保证每叶最小样本数 >= MIN_GROUP_SIZE
    min_leaf = max(1, MIN_GROUP_SIZE)
    dt = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=min_leaf, random_state=RANDOM_STATE)
    dt.fit(dt_df[['bmi_mean']], dt_df['t_star_clean'])
    # 提取分割阈值
    tree = dt.tree_
    thresholds = sorted([tree.threshold[i] for i in range(tree.node_count) if tree.children_left[i] != -1])
    # 按阈值分箱
    bins = [-1e9] + thresholds + [1e9]
    labels = [f"DT_{i+1}" for i in range(len(bins)-1)]
    p0['bmi_dt'] = pd.cut(p0['bmi_mean'], bins=bins, labels=labels)
    all_results.append(('decision_tree', group_summary(p0, 'bmi_dt')))
else:
    print("无 t_star_clean 可用于决策树监督分箱")


res_df = pd.concat([r for (name,r) in all_results], keys=[name for (name,_) in all_results], names=['method','row'])
res_df.to_csv(os.path.join(OUTDIR,"grouping_summaries.csv"))
print("已保存分组统计到", os.path.join(OUTDIR,"grouping_summaries.csv"))

# 对每种方法，若所有组均 >= MIN_GROUP_SIZE，则计算 log-rank 检验
logrank_rows=[]
for name, df_summary in zip([t[0] for t in all_results],[t[1] for t in all_results]):
    # 找到 p0 中的分组列名
    col = None
    if name == 'clinical': col = 'bmi_clin'
    elif name == 'custom': col = 'bmi_custom'
    elif name == 'quantile4': col = 'bmi_q4'
    elif name == 'kmeans4': col = 'bmi_km_labels'
    elif name == 'gmm4': col = 'bmi_gmm_label'
    elif name == 'decision_tree': col = 'bmi_dt'
    else: continue
    counts = p0.groupby(col)['patient'].nunique()
    enough = (counts >= MIN_GROUP_SIZE).all()
    if not enough:
        logrank_rows.append({'method':name,'logrank_p': None, 'note':'部分分组样本数小于 MIN_GROUP_SIZE', 'group_counts':counts.to_dict()})
        continue
    # 构建生存数据
    sub = p0[[col,'time_to_event','event']].dropna(subset=['time_to_event']).copy()
    try:
        lr = multivariate_logrank_test(sub['time_to_event'], sub[col], sub['event'])
        logrank_rows.append({'method':name,'logrank_p': lr.p_value, 'stat': lr.test_statistic, 'group_counts':counts.to_dict()})
    except Exception as e:
        logrank_rows.append({'method':name,'logrank_p': None, 'note':str(e), 'group_counts':counts.to_dict()})
pd.DataFrame(logrank_rows).to_csv(os.path.join(OUTDIR,"grouping_logrank.csv"))
print("已保存 logrank 结果到", os.path.join(OUTDIR,"grouping_logrank.csv"))

# 重采样患者并重复决策树分割
import random
boot_thresholds=[]
for b in range(N_BOOT):
    sample = p0.sample(frac=1.0, replace=True, random_state=RANDOM_STATE+b)
    sample_dt = sample.dropna(subset=['t_star_clean'])
    if sample_dt.shape[0] < MIN_GROUP_SIZE*2:
        boot_thresholds.append(None)
        continue
    dt = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=max(1,MIN_GROUP_SIZE), random_state=RANDOM_STATE+b)
    try:
        dt.fit(sample_dt[['bmi_mean']], sample_dt['t_star_clean'])
        tree = dt.tree_
        thresholds = sorted([tree.threshold[i] for i in range(tree.node_count) if tree.children_left[i] != -1])
        boot_thresholds.append(thresholds)
    except:
        boot_thresholds.append(None)

pd.DataFrame({'thresholds':boot_thresholds}).to_csv(os.path.join(OUTDIR,"bootstrap_thresholds_decision_tree.csv"))
print("已保存 bootstrap 阈值.")

# 比较各方法的 p90 t*
plot_rows = []
for name, df in all_results:
    for _, row in df.iterrows():
        plot_rows.append({'method':name,'group':row['group'],'p90':row['p90_tstar'],'n':row['n']})
plot_df = pd.DataFrame(plot_rows)
plt.figure(figsize=(10,6))
sns.barplot(data=plot_df, x='method', y='p90', hue='group')
plt.xticks(rotation=45)
plt.ylabel('p90 t* (天)')
plt.title('不同分组方法下各组 p90 t*')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR,"p90_by_method.png"))
plt.close()
print("已保存 p90_by_method.png")
