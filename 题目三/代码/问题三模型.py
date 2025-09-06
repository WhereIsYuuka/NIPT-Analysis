# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import os, sys, warnings, traceback
from datetime import datetime
warnings.filterwarnings("ignore")

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable
    from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import multivariate_logrank_test
    from lifelines.utils import concordance_index
    from scipy.stats import pearsonr
except Exception as e:
    print("缺少依赖或导入错误:", e)
    raise

# 配置
DATA_PATH = r"C:\Users\admin\Desktop\国赛论文\修改后附件男.xlsx"
OUTDIR = r"C:\Users\admin\Desktop\国赛论文\问题三_full_results_fixed"
THRESH = 0.04
GROUP_QUANTILE = 4
RECOMMEND_P = 0.90
MIN_GROUP_SIZE = 25
BMI_PATIENT_METHOD = 'mean'
SIM_TAUS = [0.001, 0.002, 0.005]
N_MC = 500
MC_SEED = 123456
os.makedirs(OUTDIR, exist_ok=True)


LOGPATH = os.path.join(OUTDIR, "run_log.txt")
logf = open(LOGPATH, "a", encoding="utf-8")
def log(msg):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    logf.write(line + "\n")
log("开始运行. DATA_PATH=%s OUTDIR=%s" % (DATA_PATH, OUTDIR))


log("读取数据...")
df = pd.read_excel(DATA_PATH, engine='openpyxl')
log(f"原始形状: {df.shape}")

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
    'healthy': find_col('胎儿是否健康','healthy')
}
log("自动检测到的映射: " + str(mapping))

# 检查核心列
for k in ['patient','gest','bmi','yconc']:
    if mapping.get(k) is None:
        raise RuntimeError(f"未找到 '{k}' 所需列。所有列: {cols}")

# 子集并重命名
data = df[[mapping['patient'], mapping['gest'], mapping['bmi'], mapping['yconc']]].copy()
data.columns = ['patient','gest_day','bmi','yconc']
# 可选列
if mapping.get('weight') is not None:
    data['weight'] = pd.to_numeric(df[mapping['weight']], errors='coerce')
else:
    data['weight'] = np.nan
if mapping.get('age') is not None:
    data['age'] = pd.to_numeric(df[mapping['age']], errors='coerce')
else:
    data['age'] = np.nan
if mapping.get('parity') is not None:
    data['parity'] = pd.to_numeric(df[mapping['parity']], errors='coerce')
else:
    data['parity'] = np.nan
if mapping.get('healthy') is not None:
    data['healthy'] = df[mapping['healthy']]
else:
    data['healthy'] = np.nan

# 去除缺失核心列
before = len(data)
data = data.dropna(subset=['patient','gest_day','bmi','yconc']).reset_index(drop=True)
log(f"去除缺失核心值的行 {before - len(data)} 条. 剩余: {len(data)}")

# 确保数值型
log(f"按方法 '{BMI_PATIENT_METHOD}' 计算 patient-level BMI")
if BMI_PATIENT_METHOD == 'first':
    pfirst = data.sort_values(['patient','gest_day']).groupby('patient', as_index=False).first()
    patient_bmi = pfirst[['patient','bmi']].rename(columns={'bmi':'bmi_mean'})
else:
    patient_bmi = data.groupby('patient', as_index=False).agg(bmi_mean=('bmi','mean'), bmi_first=('bmi','first'), n_obs=('bmi','size'))

patient_bmi['bmi_for_group'] = patient_bmi['bmi_mean']

# 分组：临床→自定义→分位数（保证最小组大小）
def clinical_group(b):
    if b < 18.5: return '<18.5'
    if b < 24: return '18.5-24'
    if b < 28: return '24-28'
    if b < 32: return '28-32'
    return '>=32'
patient_bmi['group_clin'] = patient_bmi['bmi_for_group'].apply(clinical_group)
clin_counts = patient_bmi['group_clin'].value_counts()
if (clin_counts < MIN_GROUP_SIZE).any():
    # 针对 28-35 密集区间自定义
    def custom_group(b):
        if b < 28: return '<28'
        if b < 30: return '28-30'
        if b < 32: return '30-32'
        if b < 35: return '32-35'
        return '>=35'
    patient_bmi['group_custom'] = patient_bmi['bmi_for_group'].apply(custom_group)
    custom_counts = patient_bmi['group_custom'].value_counts()
    if (custom_counts >= MIN_GROUP_SIZE).all():
        patient_bmi['bmi_group'] = patient_bmi['group_custom']
        group_method = 'custom_28_35'
    else:
        patient_bmi['bmi_group'] = pd.qcut(patient_bmi['bmi_for_group'], GROUP_QUANTILE, labels=[f"Q{i+1}" for i in range(GROUP_QUANTILE)])
        group_method = 'quantile'
else:
    patient_bmi['bmi_group'] = patient_bmi['group_clin']
    group_method = 'clinical'

log(f"分组方法: {group_method}; 各组计数: {patient_bmi['bmi_group'].value_counts().to_dict()}")
patient_bmi.to_csv(os.path.join(OUTDIR,"patient_bmi_groups.csv"), index=False)
print("666666666666666666666666666666\n")


import math

# 计算分位切点
quantiles = patient_bmi['bmi_for_group'].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).round(3)
q0, q25, q50, q75, q100 = quantiles.values
log(f"patient-level BMI 分位切点 (0%,25%,50%,75%,100%): {q0}, {q25}, {q50}, {q75}, {q100}")


intervals = {
    'Q1': (q0, q25),
    'Q2': (q25, q50),
    'Q3': (q50, q75),
    'Q4': (q75, q100)
}


rows = []
for k, (lo, hi) in intervals.items():
    # 将区间写成字符串
    s = f"({lo}, {hi}]"
    # 统计该区间内实际患者 min/max/count（防止 qcut 分配边界差异）
    grp = patient_bmi[(patient_bmi['bmi_for_group'] > lo) & (patient_bmi['bmi_for_group'] <= hi)]
    cnt = int(grp.shape[0])
    if cnt>0:
        real_min = float(grp['bmi_for_group'].min())
        real_max = float(grp['bmi_for_group'].max())
    else:
        real_min = float('nan'); real_max = float('nan')
    log(f"{k}: 名义区间 {s} ; 实际患者数 = {cnt} ; 实际范围 = ({real_min}, {real_max})")
    rows.append({'group':k, 'nominal_interval':s, 'n_patients':cnt, 'observed_min':real_min, 'observed_max':real_max})



print("66666666666666666666666666666666666666666")
# 保存区间信息
mean_gest = data['gest_day'].mean()
mean_bmi = data['bmi'].mean()
mean_weight = data['weight'].mean() if data['weight'].notna().sum()>0 else np.nan
mean_age = data['age'].mean() if data['age'].notna().sum()>0 else np.nan
log(f"中心化均值: mean_gest={mean_gest:.6f}, mean_bmi={mean_bmi:.6f}, mean_weight={mean_weight}, mean_age={mean_age}")

data['gest_day_c'] = data['gest_day'] - mean_gest
data['bmi_c'] = data['bmi'] - mean_bmi
data['weight_c'] = data['weight'] - mean_weight if not np.isnan(mean_weight) else np.nan
data['age_c'] = data['age'] - mean_age if not np.isnan(mean_age) else np.nan

# 二元达标指示
data['reach'] = (data['yconc'] >= THRESH).astype(int)

# 合并分组信息到观测
data = data.merge(patient_bmi[['patient','bmi_for_group','bmi_group','bmi_mean']], on='patient', how='left')

# 检查每个孕妇的观测次数
counts = data['patient'].value_counts()
pct_ge3 = (counts>=3).mean()
pct_ge5 = (counts>=5).mean()
log(f"孕妇数: {counts.size}, >=3次比例={pct_ge3:.3f}, >=5次比例={pct_ge5:.3f}")
use_random_slope = True if (pct_ge5 >= 0.6 or pct_ge3 >= 0.8) else False
re_formula = "~gest_day_c" if use_random_slope else None
log(f"use_random_slope={use_random_slope}, re_formula={re_formula}")

# LMM 拟合
model_vars = ['gest_day_c','bmi_c']
if data['weight'].notna().sum() > 0:
    model_vars.append('weight_c')
if data['parity'].notna().sum() > 0:
    model_vars.append('parity')
if data['age'].notna().sum() > 0:
    model_vars.append('age_c')

formula = "yconc ~ " + " + ".join(model_vars)
log(f"拟合 LMM: {formula}")
md = smf.mixedlm(formula, data, groups=data['patient'], re_formula=re_formula)
mdf = None
try:
    mdf = md.fit(reml=False, method='lbfgs', maxiter=2000)
    log("ML 拟合完成: converged=%s" % getattr(mdf,'converged',None))
except Exception as e:
    log("ML 拟合失败（lbfgs）: " + str(e))
    try:
        mdf = md.fit(reml=False, method='powell', maxiter=2000)
        log("ML（powell）完成: converged=%s" % getattr(mdf,'converged',None))
    except Exception as e2:
        log("ML 两种方法均失败: " + str(e2))
        raise

# 若 ML 收敛则尝试 REML
if getattr(mdf,'converged',False):
    try:
        tmp = md.fit(reml=True, method='lbfgs', maxiter=2000)
        if getattr(tmp,'converged',False):
            mdf = tmp
            log("REML 收敛，采用 REML。")
        else:
            log("REML 未收敛，保留 ML。")
    except Exception as e:
        log("REML 尝试失败: " + str(e))

# 保存 LMM 摘要
with open(os.path.join(OUTDIR,"LMM_summary.txt"),"w",encoding="utf-8") as f:
    f.write(mdf.summary().as_text())
log("已保存 LMM_summary.txt")
log("固定效应:\n%s" % str(mdf.fe_params))
cov_re = getattr(mdf,'cov_re',None)
log("随机效应协方差:\n%s" % str(cov_re))
resid_var = float(getattr(mdf,'scale',np.nan))
log(f"残差方差 (scale) = {resid_var:.9f}")

# 计算 LMM 预测及指标
data['pred_fixed'] = mdf.fe_params['Intercept'] + sum(mdf.fe_params.get(v,0.0)*data[v] for v in model_vars)
re_dict = mdf.random_effects
def blup_pred(row):
    pid = row['patient']
    bre = re_dict.get(pid, {})
    b0 = bre.get('Intercept',0.0) if isinstance(bre,(dict,pd.Series)) else 0.0
    b1 = bre.get('gest_day_c',0.0) if isinstance(bre,(dict,pd.Series)) else 0.0
    return b0 + b1 * row['gest_day_c']
data['pred_random'] = data.apply(blup_pred, axis=1)
data['pred_total'] = data['pred_fixed'] + data['pred_random']

rmse = mean_squared_error(data['yconc'], data['pred_total'], squared=False)
mae = mean_absolute_error(data['yconc'], data['pred_total'])
pearson_r = pearsonr(data['yconc'], data['pred_total'])[0] if len(data)>1 else np.nan

var_fixed = np.nanvar(data['pred_fixed'])
var_random = np.nanvar(data['pred_random'])
denom = var_fixed + var_random + resid_var
marginal_r2 = var_fixed/denom if denom>0 else np.nan
conditional_r2 = (var_fixed+var_random)/denom if denom>0 else np.nan
try:
    intercept_var = float(cov_re.iloc[0,0]) if hasattr(cov_re,'iloc') else np.nan
except Exception:
    intercept_var = np.nan
icc = intercept_var/(intercept_var+resid_var) if (not np.isnan(intercept_var) and (intercept_var+resid_var)>0) else np.nan

metrics = dict(rmse=rmse, mae=mae, pearson_r=pearson_r,
               var_fixed=var_fixed, var_random=var_random, resid_var=resid_var,
               marginal_r2=marginal_r2, conditional_r2=conditional_r2, icc=icc,
               n_obs=len(data), n_patients=data['patient'].nunique())
with open(os.path.join(OUTDIR,"lmm_key_metrics.txt"),"w",encoding="utf-8") as f:
    f.write(str(metrics))
log("已保存 lmm_key_metrics.txt")
log("LMM 指标: %s" % str(metrics))

# 计算每孕妇 t*
log("计算每孕妇 BLUP 和 t* ...")
re_df = pd.DataFrame.from_dict(re_dict, orient='index').reset_index().rename(columns={'index':'patient'})
if 'Intercept' not in re_df.columns: re_df['Intercept'] = 0.0
if 'gest_day_c' not in re_df.columns: re_df['gest_day_c'] = 0.0

# 合并孕妇级协变量到patient表
patient_covs = data.groupby('patient', as_index=False).agg(
    parity_patient = ('parity','first'),
    age_patient = ('age','first')
)
patients = patient_bmi.merge(re_df, on='patient', how='left').merge(patient_covs, on='patient', how='left')
patients['Intercept'] = patients['Intercept'].fillna(0.0)
patients['gest_day_c'] = patients['gest_day_c'].fillna(0.0)
patients['bmi_c_patient'] = patients['bmi_mean'] - mean_bmi
# 固定系数
beta0 = float(mdf.fe_params['Intercept'])
beta_t = float(mdf.fe_params.get('gest_day_c',0.0))
beta_bmi = float(mdf.fe_params.get('bmi_c',0.0))
beta_weight = float(mdf.fe_params.get('weight_c',0.0)) if 'weight_c' in model_vars else 0.0
beta_parity = float(mdf.fe_params.get('parity',0.0)) if 'parity' in model_vars else 0.0
beta_age = float(mdf.fe_params.get('age_c',0.0)) if 'age_c' in model_vars else 0.0

patients['fixed_intercept'] = beta0 + beta_bmi * patients['bmi_c_patient']
patients['fixed_slope'] = beta_t
patients['b0j'] = patients['Intercept']
patients['b1j'] = patients['gest_day_c']

def compute_tstar_row(r):
    denom = r['fixed_slope'] + r['b1j']
    numer = THRESH - (r['fixed_intercept'] + r['b0j'])
    if pd.isna(denom) or denom <= 0:
        return np.nan
    tstar = mean_gest + numer / denom
    return float(tstar)

patients['t_star'] = patients.apply(compute_tstar_row, axis=1)
min_gest = data['gest_day'].min()
patients['t_star_clean'] = patients['t_star'].where((patients['t_star']>=min_gest) & (patients['t_star']<=300), np.nan)

patients.to_csv(os.path.join(OUTDIR,"per_patient_tstar.csv"), index=False)
log("已保存 per_patient_tstar.csv")


log("计算分组 p50/p90 推荐 ...")
group_rows = []
for g,grp in patients.groupby('bmi_group'):
    vals = grp['t_star_clean'].dropna().values
    n = int(grp.shape[0])
    if vals.size==0:
        median_t = np.nan; p90_t = np.nan
    else:
        median_t = float(np.nanmedian(vals))
        p90_t = float(np.nanpercentile(vals, 100*RECOMMEND_P))
    group_rows.append({'bmi_group':g, 'n':n, 'median_tstar':median_t, 'p90_tstar':p90_t})
gdf = pd.DataFrame(group_rows).sort_values('bmi_group')
gdf.to_csv(os.path.join(OUTDIR,"group_recommendations.csv"), index=False)
log("已保存 group_recommendations.csv")
log("分组推荐:\n%s" % str(gdf))


log("准备生存数据集（首次达标/末次观测）...")
first_hit = data[data['yconc']>=THRESH].groupby('patient')['gest_day'].min().reset_index().rename(columns={'gest_day':'first_hit_day'})
last_obs = data.groupby('patient')['gest_day'].max().reset_index().rename(columns={'gest_day':'last_day'})

surv = patients.merge(first_hit, on='patient', how='left').merge(last_obs, on='patient', how='left')
surv['event'] = (~surv['first_hit_day'].isna()).astype(int)
surv['time_to_event'] = surv.apply(lambda r: r['first_hit_day'] if r['event']==1 else r['last_day'], axis=1)
surv.to_csv(os.path.join(OUTDIR,"per_patient_survival.csv"), index=False)
log("已保存 per_patient_survival.csv")

# KM 曲线
plt.figure(figsize=(8,6))
kmf = KaplanMeierFitter()
for (g,grp) in surv.groupby('bmi_group'):
    if len(grp) < 5:
        continue
    kmf.fit(grp['time_to_event'], event_observed=grp['event'], label=f"{g} (n={len(grp)})")
    kmf.plot_survival_function(ci_show=True)
plt.title(f"KM: 尚未达到 Yconc >= {THRESH} 按 BMI 分组")
plt.xlabel("孕天数")
plt.ylabel("生存概率（未达标）")
plt.grid(True)
plt.savefig(os.path.join(OUTDIR,"km_by_bmi_group_patientlevel.png"), dpi=150)
plt.close()
log("已保存 km_by_bmi_group_patientlevel.png")


try:
    lr = multivariate_logrank_test(surv['time_to_event'], surv['bmi_group'], surv['event'])
    lr_text = f"multivariate_logrank_test: stat={lr.test_statistic:.4f}, p={lr.p_value:.6f}, df={lr.degrees_of_freedom}"
    with open(os.path.join(OUTDIR,"logrank_summary.txt"),"w",encoding="utf-8") as f: f.write(lr_text)
    log("Log-rank: " + lr_text)
except Exception as e:
    log("Log-rank 失败: " + str(e))

# Cox 模型
cox_cols = ['time_to_event','event','bmi_mean']
if 'parity_patient' in surv.columns:
    cox_cols.append('parity_patient')
if 'age_patient' in surv.columns:
    # 若有 mean_age 则中心化
    surv['age_c'] = surv['age_patient'] - mean_age if not np.isnan(mean_age) else surv['age_patient']
    cox_cols.append('age_c')

cox_df = surv[cox_cols].copy().rename(columns={'bmi_mean':'bmi', 'parity_patient':'parity'})
cox_df = cox_df.dropna(subset=['time_to_event'])
log("Cox 用协变量: %s" % str([c for c in cox_df.columns if c not in ['time_to_event','event']]))

cph = CoxPHFitter()
cox_success = False
try:
    cph.fit(cox_df, duration_col='time_to_event', event_col='event')
    cph.summary.to_csv(os.path.join(OUTDIR,"cox_summary.csv"))
    c_index = concordance_index(cox_df['time_to_event'], -cph.predict_partial_hazard(cox_df), cox_df['event'])
    log("Cox 拟合成功. concordance_index=%.4f" % c_index)
    cox_success = True
except Exception as e:
    log("Cox 初始拟合失败: " + str(e))
    for pen in [0.1,0.01,0.001]:
        try:
            cph = CoxPHFitter(penalizer=pen)
            cph.fit(cox_df, duration_col='time_to_event', event_col='event')
            cph.summary.to_csv(os.path.join(OUTDIR,f"cox_summary_penalized_{pen}.csv"))
            log("Cox 带惩罚项拟合成功 (pen=%s)." % str(pen))
            cox_success = True
            break
        except Exception as e2:
            log("Cox 带惩罚项拟合 pen=%s 失败: %s" % (str(pen), str(e2)))
if not cox_success:
    log("Cox 拟合未成功.")

# GEE 模型
log("尝试 GEE（二项分布，交换型）: reach ~ gest_day + bmi + ...")
gee_vars = ['gest_day_c','bmi_c']
if 'weight_c' in data.columns and data['weight_c'].notna().sum()>0:
    gee_vars.append('weight_c')
if 'parity' in data.columns and data['parity'].notna().sum()>0:
    gee_vars.append('parity')
if 'age_c' in data.columns and data['age_c'].notna().sum()>0:
    gee_vars.append('age_c')

gee_formula = "reach ~ " + " + ".join(gee_vars)
try:
    gee_model = GEE.from_formula(gee_formula, groups="patient", data=data, family=Binomial(), cov_struct=Exchangeable())
    gee_res = gee_model.fit()
    with open(os.path.join(OUTDIR,"GEE_summary.txt"),"w",encoding="utf-8") as f: f.write(gee_res.summary().as_text())
    data['pred_reach_prob'] = gee_res.predict(data)
    try:
        auc = roc_auc_score(data['reach'], data['pred_reach_prob'])
    except Exception:
        auc = np.nan
    log(f"GEE 拟合完成. AUC (观测级别) = {auc:.4f}")
except Exception as e:
    log("GEE 拟合失败: " + str(e))
    gee_res = None

# BLUP 截距分布直方图
try:
    reints = [v.get('Intercept',0.0) if isinstance(v,(dict,pd.Series)) else 0.0 for v in re_dict.values()]
    plt.figure(figsize=(6,4)); sns.histplot(reints, kde=True); plt.title("BLUP 截距分布"); plt.savefig(os.path.join(OUTDIR,"blup_intercept_hist.png"),dpi=150); plt.close()
except Exception as e:
    log("BLUP 直方图失败: " + str(e))

# 样本轨迹
try:
    days = np.linspace(data['gest_day'].min(), data['gest_day'].max(), 200)
    plt.figure(figsize=(8,6))
    fixed_mean_curve = beta0 + beta_t*(days-mean_gest)
    plt.plot(days, fixed_mean_curve, color='black', label='总体均值')
    sample_patients = list(data['patient'].drop_duplicates()[:12])
    for pid in sample_patients:
        bre = re_dict.get(pid,{})
        b0 = bre.get('Intercept',0.0) if isinstance(bre,(dict,pd.Series)) else 0.0
        b1 = bre.get('gest_day_c',0.0) if isinstance(bre,(dict,pd.Series)) else 0.0
        plt.plot(days, (beta0 + b0) + (beta_t + b1)*(days-mean_gest), alpha=0.7)
    plt.axhline(THRESH, color='red', linestyle='--'); plt.title("样本预测轨迹"); plt.xlabel("孕天数"); plt.ylabel("Yconc"); plt.savefig(os.path.join(OUTDIR,"predicted_trajectories.png"), dpi=150); plt.close()
except Exception as e:
    log("轨迹图失败: " + str(e))

# t* 分组小提琴图
try:
    plt.figure(figsize=(8,6))
    sns.violinplot(x='bmi_group', y='t_star_clean', data=patients, inner=None)
    sns.stripplot(x='bmi_group', y='t_star_clean', data=patients, color='k', size=3, jitter=True)
    plt.title("t* 按 BMI 分组"); plt.savefig(os.path.join(OUTDIR,"tstar_by_group_violin.png"), dpi=150); plt.close()
except Exception as e:
    log("t* violin 绘图失败: " + str(e))

# 误差模拟
log("开始测量误差蒙特卡洛模拟 ...")
rng = np.random.default_rng(MC_SEED)
sim_records = []
for tau in SIM_TAUS:
    log(f"MC 模拟 tau={tau}, N_MC={N_MC}")
    per_patient_med = []
    for idx, r in patients.iterrows():
        t0 = r['t_star']
        slope = beta_t + r['gest_day_c']
        if pd.isna(t0) or slope <= 1e-8:
            per_patient_med.append(np.nan); continue
        eps = rng.normal(0, tau, size=N_MC)
        t_sim = t0 - eps / slope
        t_sim = t_sim[(t_sim>=min_gest) & (t_sim<=300)]
        if len(t_sim)==0:
            per_patient_med.append(np.nan)
        else:
            per_patient_med.append(float(np.nanmedian(t_sim)))
    patients[f'tstar_tau_{tau}'] = per_patient_med
    grp_p90 = patients.groupby('bmi_group')[f'tstar_tau_{tau}'].apply(lambda x: float(np.nanpercentile(x.dropna(), 100*RECOMMEND_P)) if x.dropna().size>0 else np.nan)
    for g,v in grp_p90.items():
        sim_records.append({'tau':tau, 'bmi_group':g, 'p90_tstar':v})
pd.DataFrame(sim_records).to_csv(os.path.join(OUTDIR,"measurement_error_sim.csv"), index=False)
log("已保存 measurement_error_sim.csv")

# t* vs tau 图
try:
    sim_df = pd.DataFrame(sim_records)
    plt.figure(figsize=(8,6))
    for g, grp in sim_df.groupby('bmi_group'):
        plt.plot(grp['tau'], grp['p90_tstar'], marker='o', label=g)
    plt.xlabel("tau"); plt.ylabel(f"P{int(100*RECOMMEND_P)} t*"); plt.legend(); plt.grid(True); plt.savefig(os.path.join(OUTDIR,"tstar_vs_tau.png"), dpi=150); plt.close()
except Exception as e:
    log("tstar_vs_tau 绘图失败: " + str(e))


summary = {
    'n_obs': int(len(data)),
    'n_patients': int(data['patient'].nunique()),
    'group_method': group_method,
    'group_counts': patient_bmi['bmi_group'].value_counts().to_dict(),
    'LMM_metrics': metrics,
    'cox_success': bool(cox_success),
    'GEE_available': bool(gee_res is not None)
}
pd.Series(summary).to_csv(os.path.join(OUTDIR,"run_summary.csv"))
with open(os.path.join(OUTDIR,"run_summary.txt"), "w", encoding="utf-8") as f:
    f.write(str(summary))
log("已保存 run_summary")


log("运行结束。请检查输出目录: %s" % OUTDIR)
logf.close()
