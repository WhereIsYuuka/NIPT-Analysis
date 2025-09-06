# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lifelines.statistics import multivariate_logrank_test
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

# 配置
DATA_PATH = r"C:\Users\admin\Desktop\国赛论文\修改后附件男.xlsx"   # 数据文件路径
OUTDIR = r"C:\Users\admin\Desktop\国赛论文\问题三_KMeans_groups"  # 输出目录
THRESH = 0.04
K_MIN, K_MAX = 4, 8         # KMeans聚类K的范围
MIN_GROUP_SIZE = 10         # 某些K组较小时可设小阈值（可改回25）
RECOMMEND_P = 0.90
os.makedirs(OUTDIR, exist_ok=True)
RANDOM_STATE = 42

print("Reading data:", DATA_PATH)
df = pd.read_excel(DATA_PATH, engine='openpyxl')
# 自动检测列名
cols = list(df.columns)
def find_col(*cands):
    for c in cands:
        for col in cols:
            if c.lower() in str(col).lower():
                return col
    return None

mapping = {
    'patient': find_col('孕妇代码','patient','preg','id'),
    'gest': find_col('检测孕周','孕周','gest'),
    'bmi': find_col('孕妇BMI','bmi'),
    'yconc': find_col('Y染色体浓度','yconc','y_conc','y染'),
    'weight': find_col('体重','weight'),
    'age': find_col('年龄','age'),
    'parity': find_col('生产次数','怀孕次数','parity'),
}
print("Auto mapping:", mapping)

# 检查核心列
for k in ['patient','gest','bmi','yconc']:
    if mapping.get(k) is None:
        raise RuntimeError(f"Column for {k} not found.")

data = df[[mapping['patient'], mapping['gest'], mapping['bmi'], mapping['yconc']]].copy()
data.columns = ['patient','gest_day','bmi','yconc']
# add optional
data['weight'] = pd.to_numeric(df[mapping['weight']], errors='coerce') if mapping.get('weight') else np.nan
data['age'] = pd.to_numeric(df[mapping['age']], errors='coerce') if mapping.get('age') else np.nan
data['parity'] = pd.to_numeric(df[mapping['parity']], errors='coerce') if mapping.get('parity') else np.nan

data = data.dropna(subset=['patient','gest_day','bmi','yconc']).reset_index(drop=True)
print("Observations:", data.shape)


# 计算每位孕妇的BMI均值
patient_bmi = data.groupby('patient', as_index=False).agg(bmi_mean=('bmi','mean'))
patient_bmi.to_csv(os.path.join(OUTDIR,"patient_bmi_mean.csv"), index=False)


# 全局中心化
mean_gest = data['gest_day'].mean()
mean_bmi = data['bmi'].mean()
mean_weight = data['weight'].mean() if data['weight'].notna().sum()>0 else np.nan
mean_age = data['age'].mean() if data['age'].notna().sum()>0 else np.nan

data['gest_day_c'] = data['gest_day'] - mean_gest
data['bmi_c'] = data['bmi'] - mean_bmi
data['weight_c'] = data['weight'] - mean_weight if not np.isnan(mean_weight) else np.nan
data['age_c'] = data['age'] - mean_age if not np.isnan(mean_age) else np.nan
data['reach'] = (data['yconc'] >= THRESH).astype(int)

# 合并孕妇BMI均值回原表
data = data.merge(patient_bmi, on='patient', how='left')


# LMM拟合
# 构建模型公式
model_vars = ['gest_day_c','bmi_c']
if data['weight'].notna().sum()>0:
    model_vars.append('weight_c')
if data['parity'].notna().sum()>0:
    model_vars.append('parity')
if data['age'].notna().sum()>0:
    model_vars.append('age_c')

formula = "yconc ~ " + " + ".join(model_vars)
print("LMM formula:", formula)

md = smf.mixedlm(formula, data, groups=data['patient'], re_formula="~gest_day_c")
mdf = md.fit(reml=False, method='lbfgs', maxiter=2000)
print("LMM done. converged:", getattr(mdf,'converged',None))
with open(os.path.join(OUTDIR,"LMM_summary.txt"),"w",encoding="utf-8") as f:
    f.write(mdf.summary().as_text())
print("Saved LMM_summary.txt")


# 提取LMM结果
re_dict = mdf.random_effects
beta0 = float(mdf.fe_params['Intercept'])
beta_t = float(mdf.fe_params.get('gest_day_c',0.0))
beta_bmi = float(mdf.fe_params.get('bmi_c',0.0))
beta_weight = float(mdf.fe_params.get('weight_c',0.0)) if 'weight_c' in model_vars else 0.0
beta_parity = float(mdf.fe_params.get('parity',0.0)) if 'parity' in model_vars else 0.0
beta_age = float(mdf.fe_params.get('age_c',0.0)) if 'age_c' in model_vars else 0.0
scale = float(getattr(mdf,'scale', np.nan))

# 构建包含BLUP的孕妇表
re_df = pd.DataFrame.from_dict(re_dict, orient='index').reset_index().rename(columns={'index':'patient'})
if 'Intercept' not in re_df.columns: re_df['Intercept'] = 0.0
if 'gest_day_c' not in re_df.columns: re_df['gest_day_c'] = 0.0
patients = patient_bmi.merge(re_df, on='patient', how='left')
patients['bmi_c_patient'] = patients['bmi_mean'] - mean_bmi
patients['Intercept'] = patients['Intercept'].fillna(0.0)
patients['gest_day_c'] = patients['gest_day_c'].fillna(0.0)

# 计算每位孕妇的t*
def compute_tstar_row(r):
    fixed_intercept = beta0 + beta_bmi * (r['bmi_mean'] - mean_bmi)
    fixed_slope = beta_t
    b0j = r['Intercept']; b1j = r['gest_day_c']
    denom = fixed_slope + b1j
    numer = THRESH - (fixed_intercept + b0j)
    if pd.isna(denom) or denom <= 0:
        return np.nan
    tstar = mean_gest + numer / denom
    return float(tstar)

patients['t_star'] = patients.apply(compute_tstar_row, axis=1)
min_g = data['gest_day'].min()
patients['t_star_clean'] = patients['t_star'].where((patients['t_star']>=min_g) & (patients['t_star']<=300), np.nan)
patients.to_csv(os.path.join(OUTDIR,"per_patient_tstar_base.csv"), index=False)
print("Saved per_patient_tstar_base.csv")


# KMeans聚类与分组汇总
k_results = []      # 存储每个K的分组汇总行用于绘图
k_details = {}      # 存储每个K的完整分组表
sil_scores = []     # 轮廓系数
inertias = []       # inertia
for K in range(K_MIN, K_MAX+1):
    X = patients[['bmi_mean']].values
    kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.flatten()
    # 将标签按中心升序映射为C1..CK
    order = np.argsort(centers)
    label_map = {old: f"C{(np.where(order==i)[0][0]+1)}" for i,old in enumerate(order)}
    # 其余映射与排序逻辑保持
    centers_series = pd.Series(centers, index=np.arange(len(centers)))
    centers_sorted = centers_series.sort_values()
    new_name_map = {old: f"C{int(np.where(centers_sorted.index==old)[0][0]+1)}" for old in centers_series.index}
    new_labels = [new_name_map[l] for l in labels]
    patients[f'k{K}_cluster_raw'] = labels
    patients[f'k{K}_cluster'] = new_labels
    # 轮廓系数和inertia
    try:
        sil = silhouette_score(X, labels) if K>1 else np.nan
    except Exception:
        sil = np.nan
    sil_scores.append({'K':K,'silhouette':sil})
    inertias.append({'K':K,'inertia':kmeans.inertia_})
    # 分组汇总
    rows=[]
    for grp_name, grp in patients.groupby(f'k{K}_cluster'):
        n = int(grp.shape[0])
        med = float(grp['t_star_clean'].dropna().median()) if grp['t_star_clean'].dropna().size>0 else np.nan
        p90 = float(np.nanpercentile(grp['t_star_clean'].dropna(),90)) if grp['t_star_clean'].dropna().size>0 else np.nan
        bmi_min = float(grp['bmi_mean'].min()); bmi_max = float(grp['bmi_mean'].max())
        rows.append({'K':K,'cluster':grp_name,'n':n,'bmi_min':bmi_min,'bmi_max':bmi_max,'median_tstar':med,'p90_tstar':p90})
        k_results.append({'K':K,'cluster':grp_name,'n':n,'median_tstar':med,'p90_tstar':p90})
    kdf = pd.DataFrame(rows).sort_values('cluster')
    kdf.to_csv(os.path.join(OUTDIR,f"kmeans_K{K}_group_summary.csv"), index=False)
    k_details[K] = kdf
    print(f"K={K}: saved kmeans_K{K}_group_summary.csv (clusters: {kdf.shape[0]}) Sil={sil:.4f} Inertia={kmeans.inertia_}")


# 汇总并保存轮廓系数/inertia
pd.DataFrame(sil_scores).to_csv(os.path.join(OUTDIR,"kmeans_silhouette.csv"), index=False)
pd.DataFrame(inertias).to_csv(os.path.join(OUTDIR,"kmeans_inertia.csv"), index=False)

# 保存所有K的分组结果
pd.DataFrame(k_results).to_csv(os.path.join(OUTDIR,"kmeans_allK_group_results.csv"), index=False)
print("Saved kmeans_allK_group_results.csv and silhouette/inertia metrics.")


sns.set(style="whitegrid")
Ks = list(range(K_MIN, K_MAX+1))
n_plots = len(Ks)
fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 5), sharey=True)
for ax, K in zip(axes, Ks):
    kdf = k_details[K].copy()
    # 按bmi_min升序排列分组
    kdf = kdf.sort_values('bmi_min')
    sns.barplot(data=kdf, x='cluster', y='p90_tstar', ax=ax, palette='tab10')
    ax.set_title(f"K={K}")
    ax.set_xlabel("")
    if ax==axes[0]:
        ax.set_ylabel(f"p90 t* (days) (threshold={THRESH})")
    else:
        ax.set_ylabel("")
    for i,row in kdf.reset_index().iterrows():
        val = row['p90_tstar']
        if not np.isnan(val):
            ax.text(i, val + 0.5, f"{val:.1f}", ha='center', va='bottom', fontsize=9)
plt.suptitle("KMeans BMI grouping: p90 t* by cluster, K=4..8")
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(os.path.join(OUTDIR,"kmeans_p90_comparison_K4to8.png"), dpi=200)
plt.close()
print("Saved kmeans_p90_comparison_K4to8.png")


# 绘制肘部法则和轮廓系数图
sil_df = pd.read_csv(os.path.join(OUTDIR,"kmeans_silhouette.csv"))
inertia_df = pd.read_csv(os.path.join(OUTDIR,"kmeans_inertia.csv"))
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(inertia_df['K'], inertia_df['inertia'], marker='o'); plt.title('KMeans inertia (elbow)'); plt.xlabel('K'); plt.ylabel('inertia')
plt.subplot(1,2,2)
plt.plot(sil_df['K'], sil_df['silhouette'], marker='o'); plt.title('Silhouette score'); plt.xlabel('K'); plt.ylabel('silhouette')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR,"kmeans_elbow_silhouette.png"), dpi=200)
plt.close()
print("Saved kmeans_elbow_silhouette.png")


# 选择最佳K
bestK = K_MIN
patients.to_csv(os.path.join(OUTDIR,"patients_with_k_clusters.csv"), index=False)
print("Saved patients_with_k_clusters.csv (contains per-K cluster assignments)")


# 保存运行总结
summary = {
    'n_obs': int(len(data)),
    'n_patients': int(patients.shape[0]),
    'K_range': f"{K_MIN}-{K_MAX}",
    'LMM_formula': formula,
    'LMM_converged': bool(getattr(mdf,'converged',False))
}
pd.Series(summary).to_csv(os.path.join(OUTDIR,"run_summary_KMeans.csv"))
print("Saved run_summary_KMeans.csv. Finished. Check:", OUTDIR)
