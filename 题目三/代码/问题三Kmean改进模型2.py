# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import os, sys, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


PER_PATIENT_CSV = r"C:\Users\admin\Desktop\国赛论文\问题三_KMeans_groups\per_patient_tstar_base.csv"
OUTDIR = r"C:\Users\admin\Desktop\国赛论文\kmeans_grouping_inspect"
K_MIN, K_MAX = 4, 8
RANDOM_STATE = 42
MAX_RETRIES = 10  
MIN_SAMPLE_FOR_CLUSTER = 1  
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(PER_PATIENT_CSV)
required = ['patient','bmi_mean','t_star_clean']
for c in required:
    if c not in df.columns:
        raise RuntimeError(f"输入文件必须包含列: {required}. 当前文件列: {list(df.columns)}")
patients = df[['patient','bmi_mean','t_star_clean']].copy()
# 保证为数值型
patients['bmi_mean'] = pd.to_numeric(patients['bmi_mean'], errors='coerce')
patients['t_star_clean'] = pd.to_numeric(patients['t_star_clean'], errors='coerce')

# 简要检查
print("已加载患者数:", patients.shape[0], "BMI 范围:", patients['bmi_mean'].min(), patients['bmi_mean'].max())

all_k_summaries = []
k_details = {}
sil_list = []
inertia_list = []

for K in range(K_MIN, K_MAX+1):
    print("处理 K =", K)
    X = patients[['bmi_mean']].values
    # 若唯一标签数 < K，多次尝试
    success = False
    attempt = 0
    while attempt < MAX_RETRIES and not success:
        seed = RANDOM_STATE + attempt
        kmeans = KMeans(n_clusters=K, random_state=seed, n_init=20).fit(X)
        labels = kmeans.labels_
        unique_labels = np.unique(labels)
        if len(unique_labels) == K:
            success = True
        else:
            attempt += 1
    if not success:
        print(f"警告: 尝试 {MAX_RETRIES} 次后 KMeans K={K} 仅得到 {len(unique_labels)} 个标签。继续。")
    # 构建稳定映射
    centers = kmeans.cluster_centers_.flatten()
    old_labels = np.arange(len(centers))
    center_map = {int(l): float(centers[int(l)]) for l in old_labels}
    # 按中心值升序排序簇
    sorted_clusters = sorted(center_map.items(), key=lambda x: x[1])  # (old_label, center)
    # 按 BMI 升序
    new_name_map = {old: f"C{idx+1}" for idx, (old,_) in enumerate(sorted_clusters)}
    # 新标签数组
    new_labels = [ new_name_map[int(l)] if int(l) in new_name_map else f"C_unk{int(l)}" for l in labels ]
    patients[f'k{K}_lab_old'] = labels
    patients[f'k{K}_lab'] = new_labels

    # 计算 silhouette/inertia
    try:
        sil = silhouette_score(X, labels) if len(unique_labels) > 1 else np.nan
    except Exception:
        sil = np.nan
    sil_list.append({'K':K,'silhouette':sil})
    inertia_list.append({'K':K,'inertia': float(kmeans.inertia_) if hasattr(kmeans,'inertia_') else np.nan})

    # 计算每簇摘要及 BMI 范围
    rows=[]
    # 保证顺序为 C1..CK
    expected_clusters = [f"C{i+1}" for i in range(K)]
    for cname in expected_clusters:
        grp = patients[patients[f'k{K}_lab']==cname]
        n = int(grp.shape[0])
        bmi_min = float(grp['bmi_mean'].min()) if n>0 else np.nan
        bmi_max = float(grp['bmi_mean'].max()) if n>0 else np.nan
        center_val = np.nan
        # 找到 cname 的原标签
        old_for_new = None
        for old, new in new_name_map.items():
            if new == cname:
                old_for_new = old; center_val = center_map.get(old, np.nan); break
        med = float(grp['t_star_clean'].median()) if (n>0 and grp['t_star_clean'].dropna().size>0) else np.nan
        p90 = float(np.nanpercentile(grp['t_star_clean'].dropna(), 90)) if (n>0 and grp['t_star_clean'].dropna().size>0) else np.nan
        rows.append({'K':K, 'cluster':cname, 'n':n, 'center_bmi':center_val, 'bmi_min':bmi_min, 'bmi_max':bmi_max, 'median_tstar':med, 'p90_tstar':p90})
        all_k_summaries.append({'K':K, 'cluster':cname, 'n':n, 'center_bmi':center_val, 'bmi_min':bmi_min, 'bmi_max':bmi_max, 'median_tstar':med, 'p90_tstar':p90})
    kdf = pd.DataFrame(rows)
    kdf.to_csv(os.path.join(OUTDIR, f"kmeans_K{K}_group_summary_fixed.csv"), index=False)
    k_details[K] = kdf
    print(f"已保存 kmeans_K{K}_group_summary_fixed.csv (clusters={len(kdf)})")

# 保存合并结果
pd.DataFrame(all_k_summaries).to_csv(os.path.join(OUTDIR,"kmeans_allK_group_summary_fixed.csv"), index=False)
pd.DataFrame(sil_list).to_csv(os.path.join(OUTDIR,"kmeans_silhouette_fixed.csv"), index=False)
pd.DataFrame(inertia_list).to_csv(os.path.join(OUTDIR,"kmeans_inertia_fixed.csv"), index=False)
print("已保存合并摘要与指标。")

# 绘图
sns.set(style="whitegrid")
Ks = list(range(K_MIN, K_MAX+1))
n_plots = len(Ks)
fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 5), sharey=True)
for ax, K in zip(axes, Ks):
    kdf = k_details[K].copy()
    # 按中心 BMI 升序排序
    kdf = kdf.sort_values('center_bmi', na_position='last').reset_index(drop=True)

    kdf['p90_plot'] = kdf['p90_tstar'].fillna(0.0)
    palette = sns.color_palette("tab10", n_colors=max(3, kdf.shape[0]))
    sns.barplot(data=kdf, x='cluster', y='p90_plot', ax=ax, palette=palette)
    ax.set_title(f"K={K}")
    ax.set_xlabel("")
    if ax==axes[0]:
        ax.set_ylabel(f"p90 t* (天)")
    else:
        ax.set_ylabel("")
    for i,row in kdf.iterrows():
        val = row['p90_tstar']
        n = int(row['n'])
        if not np.isnan(val):
            ax.text(i, row['p90_plot'] + 0.6, f"{val:.1f}\n(n={n})", ha='center', va='bottom', fontsize=9)
        else:
            ax.text(i, 0.2, f"n={n}", ha='center', va='bottom', fontsize=9, color='gray')
plt.suptitle("KMeans BMI 分组: 各簇 p90 t*，K=4..8（固定映射）")
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(os.path.join(OUTDIR, "kmeans_p90_comparison_K4to8_fixed.png"), dpi=200)
plt.close()
print("已保存 kmeans_p90_comparison_K4to8_fixed.png")

# 保存带 k 标签的患者文件
patients.to_csv(os.path.join(OUTDIR,"patients_with_k_assignments_fixed.csv"), index=False)
print("已保存 patients_with_k_assignments_fixed.csv")

print("完成。请检查输出目录:", OUTDIR)
