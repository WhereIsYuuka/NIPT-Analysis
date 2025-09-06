import matplotlib
matplotlib.use('Agg')
import os, sys, warnings, traceback
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

#  读配置
DATA_PATH = r"C:\Users\admin\Desktop\国赛论文\修改后附件男.xlsx"
OUTDIR = r"C:\Users\admin\Desktop\国赛论文\问题二_patientlevelBMI_results"
THRESH = 0.04             # Y浓度阈值
PCT_FOR_RECOMMEND = 0.90  # 推荐采用的百分位
MIN_GROUP_SIZE = 30       # 最小样本量阈值


os.makedirs(OUTDIR, exist_ok=True)
logf = open(os.path.join(OUTDIR, "run_log.txt"), "a", encoding="utf-8")
def log(msg):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    logf.write(line + "\n")
log(f"Start run. DATA_PATH={DATA_PATH} OUTDIR={OUTDIR}")

def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# 读取并清洗数据
log("Reading data...")
df = pd.read_excel(DATA_PATH, engine='openpyxl')
log(f"Raw data shape: {df.shape}")
log("Columns: " + ", ".join(list(df.columns)[:40]))

# 自动识别关键列
cols = list(df.columns)
def find_col(cands):
    for c in cands:
        for col in cols:
            if c.strip().lower() in str(col).strip().lower():
                return col
    return None

found = {
    'patient': find_col(['孕妇代码','patient','preg_id','孕妇ID','孕妇编号']),
    'gest': find_col(['检测孕周','孕周','gest']),
    'bmi': find_col(['孕妇BMI','BMI','bmi']),
    'yconc': find_col(['Y染色体浓度','Y_conc','yconc','Y浓度']),
    'healthy': find_col(['胎儿是否健康','healthy'])
}
log("Auto-detected columns mapping: " + str(found))
if any(v is None for v in found.values()):
    err = "Column autodetect failed — please manually set 'found' mapping. Columns:\n" + "\n".join(cols)
    log(err); save_text(os.path.join(OUTDIR,"error_columns.txt"), err); logf.close(); raise RuntimeError(err)

data = df[[found['patient'], found['gest'], found['bmi'], found['yconc'], found['healthy']]].copy()
data.columns = ['patient','gest_day','bmi','yconc','healthy']
# 单位从周转换为天
# data['gest_day'] = data['gest_day'] * 7

# 强制数值并去掉缺失
for c in ['gest_day','bmi','yconc']:
    data[c] = pd.to_numeric(data[c], errors='coerce')
before = len(data)
data = data.dropna(subset=['patient','gest_day','bmi','yconc']).reset_index(drop=True)
log(f"Dropped {before - len(data)} rows with missing values; remaining {len(data)} rows.")

log("Basic stats:")
log(" BMI: " + str(data['bmi'].describe().to_dict()))
log(" gest_day: " + str(data['gest_day'].describe().to_dict()))
log(" yconc: " + str(data['yconc'].describe().to_dict()))

# 确定单一BMI
log("Computing patient-level BMI (mean per patient)...")
patient_bmi = data.groupby('patient', as_index=False).agg(
    bmi_mean=('bmi','mean'),
    bmi_first=('bmi','first'),
    yconc_mean=('yconc','mean'),
    n_obs=('bmi','size')
)
log(f"Number of unique patients: {patient_bmi.shape[0]}")

# 分组依据
patient_bmi['bmi_for_group'] = patient_bmi['bmi_mean']

# 临床分组
def clinical_bins(b):
    if b < 18.5: return 'Under18.5'
    if b < 24: return '18.5-24'
    if b < 28: return '24-28'
    if b < 32: return '28-32'
    return '>=32'
patient_bmi['group_clin'] = patient_bmi['bmi_for_group'].apply(clinical_bins)
clin_counts = patient_bmi['group_clin'].value_counts()
log("patient-level clinical group counts: " + str(clin_counts.to_dict()))

use_custom = False
if (clin_counts < MIN_GROUP_SIZE).any():
    def custom_bins(b):
        if b < 28: return '<28'
        if b < 30: return '28-30'
        if b < 32: return '30-32'
        if b < 35: return '32-35'
        return '>=35'
    patient_bmi['group_custom'] = patient_bmi['bmi_for_group'].apply(custom_bins)
    custom_counts = patient_bmi['group_custom'].value_counts()
    log("patient-level custom group counts: " + str(custom_counts.to_dict()))
    if (custom_counts >= MIN_GROUP_SIZE).all():
        patient_bmi['bmi_group'] = patient_bmi['group_custom']
        group_method = 'custom_28_35'
        use_custom = True
    else:
        patient_bmi['bmi_group'] = pd.qcut(patient_bmi['bmi_for_group'], 4, labels=['Q1','Q2','Q3','Q4'])
        group_method = 'quantile'
else:
    patient_bmi['bmi_group'] = patient_bmi['group_clin']
    group_method = 'clinical'

log(f"Chosen patient-level group method: {group_method}")
log("patient-level bmi_group counts: " + str(patient_bmi['bmi_group'].value_counts().to_dict()))

if group_method == 'quantile':
    bins = patient_bmi['bmi_for_group'].quantile([0.0,0.25,0.5,0.75,1.0]).values
    log("Patient-level BMI quantile cut points (0%,25%,50%,75%,100%): " + ", ".join([f"{b:.3f}" for b in bins]))
    # 构造区间打印
    intervals = []
    for i in range(len(bins)-1):
        intervals.append({'group':f"Q{i+1}", 'interval':f"({bins[i]:.3f}, {bins[i+1]:.3f}]", 'n': (patient_bmi['bmi_for_group']>bins[i]) & (patient_bmi['bmi_for_group']<=bins[i+1]).sum()})
    log("Patient-level quantile intervals:")
    for r in intervals:
        log(f" {r['group']}: BMI {r['interval']}, patients = {r['n']}")
    pd.DataFrame(intervals).to_csv(os.path.join(OUTDIR,"bmi_group_intervals_patientlevel.csv"), index=False)
else:
    grp_counts = patient_bmi['bmi_group'].value_counts().sort_index()
    intervals = []
    for g,n in grp_counts.items():
        intervals.append({'group':g, 'n':int(n)})
    log("Patient-level group labels and counts saved.")
    pd.DataFrame(intervals).to_csv(os.path.join(OUTDIR,"bmi_group_intervals_patientlevel.csv"), index=False)

# 合并分组信息
log("Merging patient-level bmi_group back to observation-level data...")
data = data.merge(patient_bmi[['patient','bmi_for_group','bmi_group']], on='patient', how='left')

log("Merged. Example rows:")
log(data[['patient','gest_day','bmi','bmi_for_group','bmi_group']].head(5).to_string(index=False))

# 中心化变量
data['gest_day_c'] = data['gest_day'] - data['gest_day'].mean()
data['bmi_c'] = data['bmi'] - data['bmi'].mean()
mean_gest = data['gest_day'].mean()
mean_bmi = data['bmi'].mean()
log(f"Centering: mean_gest={mean_gest:.6f}, mean_bmi={mean_bmi:.6f}")

# 随机斜率决策
counts = data['patient'].value_counts()
pct_ge3 = (counts>=3).mean()
pct_ge5 = (counts>=5).mean()
n_groups = counts.size
log(f"Patient counts summary: n_groups={n_groups}, pct_ge3={pct_ge3:.3f}, pct_ge5={pct_ge5:.3f}")

if pct_ge5 >= 0.6 or pct_ge3 >= 0.8:
    use_random_slope = True
    re_formula = "~gest_day_c"
else:
    use_random_slope = False
    re_formula = None
log(f"use_random_slope = {use_random_slope}, re_formula = {re_formula}")

# 自动设置Monte Carlo参数
total_obs = len(data)
if total_obs <= 500:
    N_MC = 300
    SIM_TAUS = [0.002, 0.005]
elif total_obs <= 1500:
    N_MC = 500
    SIM_TAUS = [0.001, 0.002, 0.005]
else:
    N_MC = 1000
    SIM_TAUS = [0.001, 0.002, 0.005]
log(f"N_MC={N_MC}, SIM_TAUS={SIM_TAUS}")

# 拟合LMM
formula = "yconc ~ gest_day_c + bmi_c"
log(f"Fitting LMM: formula={formula}, groups=patient, re_formula={re_formula}")
md = smf.mixedlm(formula, data, groups=data['patient'], re_formula=re_formula)

mdf_ml = None
try:
    mdf_ml = md.fit(reml=False, method='lbfgs', maxiter=2000)
    log(f"ML fit converged: {getattr(mdf_ml,'converged',None)}")
except Exception as e:
    log("ML fit (lbfgs) failed: " + str(e))
    try:
        mdf_ml = md.fit(reml=False, method='powell', maxiter=2000)
        log(f"ML fit (powell) converged: {getattr(mdf_ml,'converged',None)}")
    except Exception as e2:
        log("ML fit (powell) also failed: " + str(e2))

if mdf_ml is None:
    err = "ERROR: ML fit failed; aborting."
    log(err); save_text(os.path.join(OUTDIR,"LMM_error.txt"), err); logf.close(); raise RuntimeError(err)

mdf = mdf_ml
if getattr(mdf_ml, 'converged', False):
    try:
        tmp = md.fit(reml=True, method='lbfgs', maxiter=2000)
        if getattr(tmp, 'converged', False):
            mdf = tmp
            log("REML fit converged and used.")
        else:
            log("REML did not converge; using ML result as final.")
    except Exception as e:
        log("REML attempt failed; using ML. Err: " + str(e))
else:
    log("ML did not converge; using ML result.")

# 保存摘要
lmm_summary = mdf.summary().as_text()
save_text(os.path.join(OUTDIR,"LMM_summary.txt"), lmm_summary)
log("Saved LMM_summary.txt")

# 提取关键参数
fe = mdf.fe_params.to_dict()
cov_re = getattr(mdf, 'cov_re', None)
resid_var = float(getattr(mdf, 'scale', np.nan))
log("Fixed effects: " + str(fe))
log("Random effects covariance (cov_re):\n" + str(cov_re))
log(f"Residual variance (scale) = {resid_var:.9f}")

# 预测与评估
re_dict = mdf.random_effects
preds = []
fixed_preds = []
for idx, row in data.iterrows():
    pid = row['patient']
    gest_c = row['gest_day_c']
    bmi_c = row['bmi_c']
    fixed = fe['Intercept'] + fe['gest_day_c'] * gest_c + fe['bmi_c'] * bmi_c
    bre = re_dict.get(pid, {})
    b0 = bre.get('Intercept', 0.0) if isinstance(bre, dict) or isinstance(bre, pd.Series) else 0.0
    b1 = bre.get('gest_day_c', 0.0) if isinstance(bre, dict) or isinstance(bre, pd.Series) else 0.0
    pred = fixed + b0 + b1 * gest_c
    preds.append(pred)
    fixed_preds.append(fixed)

data['pred_y'] = preds
data['pred_y_fixed'] = fixed_preds

rmse = mean_squared_error(data['yconc'], data['pred_y'], squared=False)
mae = mean_absolute_error(data['yconc'], data['pred_y'])
try:
    pearson_r = pearsonr(data['pred_y'], data['yconc'])[0]
except Exception:
    pearson_r = np.nan
log(f"Prediction metrics (obs vs pred): RMSE={rmse:.6f}, MAE={mae:.6f}, Pearson r={pearson_r:.6f}")

# 方差分解与R2
var_fixed = np.nanvar(data['pred_y_fixed'])

random_effects_vals = []
for idx, row in data.iterrows():
    pid = row['patient']
    bre = re_dict.get(pid, {})
    b0 = bre.get('Intercept', 0.0) if isinstance(bre, dict) or isinstance(bre, pd.Series) else 0.0
    b1 = bre.get('gest_day_c', 0.0) if isinstance(bre, dict) or isinstance(bre, pd.Series) else 0.0
    random_effects_vals.append(b0 + b1 * row['gest_day_c'])
var_random = np.nanvar(np.array(random_effects_vals))

denom = var_fixed + var_random + resid_var
marginal_r2 = var_fixed / denom if denom>0 else np.nan
conditional_r2 = (var_fixed + var_random) / denom if denom>0 else np.nan

try:
    intercept_var = float(cov_re.iloc[0,0]) if hasattr(cov_re, 'iloc') else np.nan
except Exception:
    intercept_var = np.nan
icc = intercept_var / (intercept_var + resid_var) if (not np.isnan(intercept_var) and (intercept_var+resid_var)>0) else np.nan

log(f"Variance decomposition: var_fixed={var_fixed:.9f}, var_random={var_random:.9f}, resid_var={resid_var:.9f}")
log(f"marginal_R2={marginal_r2:.6f}, conditional_R2={conditional_r2:.6f}, ICC~{icc:.6f}")

# 计算每个患者的t_star
log("Computing per-patient t_star using BLUPs...")

re_df = pd.DataFrame.from_dict(re_dict, orient='index').reset_index().rename(columns={'index':'patient'})
if 'Intercept' not in re_df.columns:
    re_df['Intercept'] = 0.0
if 'gest_day_c' not in re_df.columns:
    re_df['gest_day_c'] = 0.0

sub = patient_bmi.merge(re_df, on='patient', how='left')

sub['Intercept'] = sub['Intercept'].fillna(0.0)
sub['gest_day_c'] = sub['gest_day_c'].fillna(0.0)

sub['bmi_c_patient'] = sub['bmi_mean'] - mean_bmi 

beta0 = fe['Intercept']; beta1 = fe['gest_day_c']; beta2 = fe['bmi_c']
sub['fixed_intercept'] = beta0 + beta2 * sub['bmi_c_patient']
sub['fixed_slope'] = beta1
sub['b0j'] = sub['Intercept']
sub['b1j'] = sub['gest_day_c']

def compute_tstar_row(r):
    denom = r['fixed_slope'] + r['b1j']
    numer = THRESH - (r['fixed_intercept'] + r['b0j'])
    if pd.isna(denom) or denom <= 0:
        return np.nan
    tstar = mean_gest + numer / denom
    return tstar

sub['t_star'] = sub.apply(compute_tstar_row, axis=1)
# 清理到合理范围
min_g = data['gest_day'].min()
sub['t_star_clean'] = sub['t_star'].where((sub['t_star']>=min_g) & (sub['t_star']<=300), np.nan)
sub[['patient','bmi_mean','bmi_for_group','bmi_group','t_star','t_star_clean']].to_csv(os.path.join(OUTDIR,"per_patient_tstar_patientlevel.csv"), index=False)
log("Saved per_patient_tstar_patientlevel.csv")

# 计算组推荐
log("Computing group-level recommendations (patient-level groups)...")
group_rows = []
for g, grp in sub.groupby('bmi_group'):
    vals = grp['t_star_clean'].dropna().values
    n = grp.shape[0]
    median_t = float(np.nanmedian(vals)) if vals.size>0 else np.nan
    p90_t = float(np.nanpercentile(vals, 90)) if vals.size>0 else np.nan
    group_rows.append({'bmi_group':g, 'n':int(n), 'median_tstar':median_t, 'p90_tstar':p90_t})
pd.DataFrame(group_rows).sort_values('bmi_group').to_csv(os.path.join(OUTDIR,"group_recommendations_patientlevel.csv"), index=False)
log("Saved group_recommendations_patientlevel.csv")
log("Group recommendations (patient-level):")
log(str(pd.DataFrame(group_rows).sort_values('bmi_group')))

# 生存分析
log("Preparing survival data (time to first observed yconc >= THRESH)...")
first_hit = data[data['yconc']>=THRESH].groupby('patient')['gest_day'].min().reset_index().rename(columns={'gest_day':'first_hit_day'})
last_obs = data.groupby('patient')['gest_day'].max().reset_index().rename(columns={'gest_day':'last_day'})
sub = sub.merge(first_hit, on='patient', how='left').merge(last_obs, on='patient', how='left')
sub['event'] = (~sub['first_hit_day'].isna()).astype(int)
sub['time_to_event'] = sub.apply(lambda r: r['first_hit_day'] if r['event']==1 else r['last_day'], axis=1)
sub[['patient','bmi_mean','bmi_group','t_star_clean','time_to_event','event']].to_csv(os.path.join(OUTDIR,"per_patient_survival_patientlevel.csv"), index=False)
log("Saved per_patient_survival_patientlevel.csv")


try:
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8,6))
    for name, grp in sub.groupby('bmi_group'):
        if len(grp) < 5:
            log(f"Skipping KM for group {name} (n={len(grp)}) <5")
            continue
        kmf.fit(grp['time_to_event'], event_observed=grp['event'], label=f"{name} (n={len(grp)})")
        kmf.plot_survival_function(ci_show=True)
    plt.title(f"KM: Not yet reached Yconc >= {THRESH} (patient-level BMI groups)")
    plt.xlabel("Gestational day")
    plt.ylabel("Survival (not reached threshold)")
    plt.grid(True)
    plt.savefig(os.path.join(OUTDIR,"km_by_bmi_group_patientlevel.png"), dpi=150)
    plt.close()
    log("Saved km_by_bmi_group_patientlevel.png")
except Exception as e:
    log("KM plotting failed: " + str(e))
    traceback.print_exc(file=logf)

try:
    res_lr = multivariate_logrank_test(sub['time_to_event'], sub['bmi_group'], sub['event'])
    lr_text = f"multivariate_logrank_test: test_statistic={res_lr.test_statistic}, p_value={res_lr.p_value}, df={res_lr.degrees_of_freedom}"
    save_text(os.path.join(OUTDIR,"logrank_summary_patientlevel.txt"), lr_text)
    log("Log-rank result: " + lr_text)
except Exception as e:
    log("Log-rank failed: " + str(e))
    traceback.print_exc(file=logf)

# Cox回归
cox_df = sub[['time_to_event','event','bmi_mean']].copy().rename(columns={'bmi_mean':'bmi'})
n_events = cox_df['event'].sum()
log(f"Cox diagnostics: n_events={n_events}, bmi variance >0? {cox_df['bmi'].var() > 0}")
cox_success = False
if n_events >= 10 and cox_df['bmi'].var() > 0:
    try:
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col='time_to_event', event_col='event')
        cph.summary.to_csv(os.path.join(OUTDIR,"cox_summary_patientlevel.csv"))
        log("Cox fit successful (no penalizer). Saved cox_summary_patientlevel.csv")
        cox_success = True
    except Exception as e:
        log("Cox initial fit failed: " + str(e))
        traceback.print_exc(file=logf)
        for pen in [0.1, 0.01, 0.001]:
            try:
                cph = CoxPHFitter(penalizer=pen)
                cph.fit(cox_df, duration_col='time_to_event', event_col='event')
                cph.summary.to_csv(os.path.join(OUTDIR,f"cox_summary_patientlevel_penalized_{pen}.csv"))
                log(f"Cox fit succeeded with penalizer={pen}. Saved.")
                cox_success = True
                break
            except Exception as e2:
                log(f"Cox penalized fit (pen={pen}) failed: {e2}")
                traceback.print_exc(file=logf)
else:
    log("Cox not attempted due to insufficient events or zero variance.")

if not cox_success:
    log("Cox regression not successful or not attempted. See run_log for details.")

# 测量误差敏感性分析
log(f"Measurement-error sensitivity: N_MC={N_MC}, SIM_TAUS={SIM_TAUS}")
rng = np.random.default_rng(123456)
sim_records = []
for tau in SIM_TAUS:
    log(f"Simulating for tau={tau} ...")
    per_sub_med = []
    for idx, r in sub.iterrows():
        t_true = r['t_star']
        slope = r['fixed_slope'] if 'fixed_slope' in r else beta1
        slope = r['fixed_slope'] + r['b1j'] if ('b1j' in r and not pd.isna(r['b1j'])) else (beta1)
        if pd.isna(t_true) or slope <= 0:
            per_sub_med.append(np.nan)
            continue
        eps = rng.normal(0, tau, size=N_MC)
        t_sim = t_true - eps / slope
        t_sim = t_sim[(t_sim>=min_g) & (t_sim<=300)]
        if t_sim.size == 0:
            per_sub_med.append(np.nan)
        else:
            per_sub_med.append(np.nanmedian(t_sim))
    sub[f'tstar_tau_{tau}'] = per_sub_med
    grp_p90 = sub.groupby('bmi_group')[f'tstar_tau_{tau}'].apply(lambda x: np.nanpercentile(x.dropna(), 100*PCT_FOR_RECOMMEND) if x.dropna().size>0 else np.nan)
    for g,v in grp_p90.items():
        sim_records.append({'tau':tau, 'bmi_group':g, 'p90_tstar': float(v) if not pd.isna(v) else np.nan})
    log(f"Finished tau={tau}")

pd.DataFrame(sim_records).to_csv(os.path.join(OUTDIR,"measurement_error_sim_patientlevel.csv"), index=False)
sub.to_csv(os.path.join(OUTDIR,"per_patient_results_patientlevel_full.csv"), index=False)
log("Saved measurement_error_sim_patientlevel.csv and per_patient_results_patientlevel_full.csv")

# 绘图
try:
    sim_df = pd.DataFrame(sim_records)
    plt.figure(figsize=(8,6))
    for g, grp in sim_df.groupby('bmi_group'):
        plt.plot(grp['tau'], grp['p90_tstar'], marker='o', label=g)
    plt.xlabel("Measurement noise sigma (tau)")
    plt.ylabel(f"P{int(100*PCT_FOR_RECOMMEND)} t* (days)")
    plt.title("Sensitivity of recommended t* to measurement noise (patient-level groups)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTDIR,"tstar_vs_tau_patientlevel.png"), dpi=150)
    plt.close()
    log("Saved tstar_vs_tau_patientlevel.png")
except Exception as e:
    log("Plot tstar_vs_tau failed: " + str(e))
    traceback.print_exc(file=logf)

# 保存运行摘要
summary = {
    'total_obs': int(len(data)),
    'n_patients': int(patient_bmi.shape[0]),
    'n_groups': int(n_groups),
    'pct_ge3': float(pct_ge3),
    'pct_ge5': float(pct_ge5),
    'use_random_slope': bool(use_random_slope),
    'group_method': group_method,
    'N_MC': int(N_MC),
    'SIM_TAUS': SIM_TAUS,
    'n_events_first_hit': int(sub['event'].sum())
}
pd.Series(summary).to_csv(os.path.join(OUTDIR,"run_summary_patientlevel.csv"))
log("Saved run_summary_patientlevel.csv")

# 保存LMM关键数值
lmm_key = {
    'fe': fe,
    'cov_re': cov_re.to_dict() if cov_re is not None else None,
    'resid_var': resid_var,
    'var_fixed': var_fixed,
    'var_random': var_random,
    'marginal_r2': marginal_r2,
    'conditional_r2': conditional_r2,
    'icc': icc,
    'rmse': rmse,
    'mae': mae,
    'pearson_r': pearson_r
}
save_text(os.path.join(OUTDIR,"lmm_key_results_patientlevel.txt"), str(lmm_key))
log("Saved lmm_key_results_patientlevel.txt with key metrics.")

log("Run complete. Outputs saved to: " + OUTDIR)
logf.close()
