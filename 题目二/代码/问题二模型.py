# nipt_bmi_tstar_final.py
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

DATA_PATH = r"C:\Users\admin\Desktop\国赛论文\修改后附件男.xlsx"
OUTDIR = r"C:\Users\admin\Desktop\国赛论文\问题二结果_final"
THRESH = 0.04
PCT_FOR_RECOMMEND = 0.90
MIN_GROUP_SIZE = 30

os.makedirs(OUTDIR, exist_ok=True)
logf = open(os.path.join(OUTDIR, "run_log.txt"), "a", encoding="utf-8")
def log(s):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {s}"
    print(line)
    logf.write(line + "\n")
log("开始运行。输出目录 -> " + OUTDIR)

def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# 读数据
log("读取数据自: " + DATA_PATH)
df = pd.read_excel(DATA_PATH, engine='openpyxl')
log(f"原始数据形状: {df.shape}")
log("前40列: " + ", ".join(list(df.columns)[:40]))

# 自动检测列名
cols = list(df.columns)
def find_col(cands):
    for c in cands:
        for col in cols:
            if c.strip().lower() in str(col).strip().lower():
                return col
    return None

found = {
    'patient': find_col(['孕妇代码','patient','preg_id','孕妇ID','孕妇编号']),
    'gest': find_col(['检测孕周','孕周','gest_week','gestational']),
    'bmi': find_col(['孕妇BMI','BMI','bmi']),
    'yconc': find_col(['Y染色体浓度','Y_conc','yconc','Y浓度']),
    'healthy': find_col(['胎儿是否健康','healthy','胎儿是否健康(1/0)'])
}
log("自动检测到的列映射: " + str(found))
if any(v is None for v in found.values()):
    err = "列自动检测失败 — 请在脚本中手动设置 'found' 映射。可用列:\n" + "\n".join(cols)
    log(err)
    save_text(os.path.join(OUTDIR,"error_columns.txt"), err)
    logf.close()
    raise RuntimeError(err)

data = df[[found['patient'], found['gest'], found['bmi'], found['yconc'], found['healthy']]].copy()
data.columns = ['patient','gest_day','bmi','yconc','healthy']
# 如果孕周为周且需转为天，取消下行注释
# data['gest_day'] = data['gest_day'] * 7

# 转为数值并去除缺失
for c in ['gest_day','bmi','yconc']:
    data[c] = pd.to_numeric(data[c], errors='coerce')
before_n = len(data)
data = data.dropna(subset=['patient','gest_day','bmi','yconc']).reset_index(drop=True)
log(f"去除缺失 patient/gest_day/bmi/yconc 的行 {before_n - len(data)} 条; 剩余 {len(data)} 行。")

# 简要统计
log("BMI 描述: " + str(data['bmi'].describe().to_dict()))
log("孕周描述: " + str(data['gest_day'].describe().to_dict()))
log("yconc 描述: " + str(data['yconc'].describe().to_dict()))

# 分组观测检查与自动参数
counts = data['patient'].value_counts()
n_groups = counts.size
pct_ge3 = (counts>=3).mean()
pct_ge5 = (counts>=5).mean()
log(f"孕妇数={n_groups}, >=3次观测比例={pct_ge3:.3f}, >=5次比例={pct_ge5:.3f}")

use_random_slope = True if (pct_ge5 >= 0.6 or pct_ge3 >= 0.8) else False
log("是否使用随机斜率 = " + str(use_random_slope))

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
log(f"设置 N_MC={N_MC}, SIM_TAUS={SIM_TAUS}")

# 分组观测检查与自动参数
def clinical_bins(b):
    if b < 18.5: return 'Under18.5'
    if b < 24: return '18.5-24'
    if b < 28: return '24-28'
    if b < 32: return '28-32'
    return '>=32'

data['bmi_clin'] = data['bmi'].apply(clinical_bins)
group_counts = data.groupby('bmi_clin')['patient'].nunique()
log("临床分组唯一孕妇数: " + str(group_counts.to_dict()))

# 若临床分组过小则尝试自定义分组
use_custom = False
if (group_counts < MIN_GROUP_SIZE).any():
    def custom_bins(b):
        if b < 28: return '<28'
        if b < 30: return '28-30'
        if b < 32: return '30-32'
        if b < 35: return '32-35'
        return '>=35'
    data['bmi_custom'] = data['bmi'].apply(custom_bins)
    custom_counts = data.groupby('bmi_custom')['patient'].nunique()
    log("自定义分组唯一孕妇数: " + str(custom_counts.to_dict()))
    print("custom bins unique patient counts (观测级别统计，患者可能被计入多组):")
    print(custom_counts.to_dict(), " sum=", sum(custom_counts))

    if (custom_counts >= MIN_GROUP_SIZE).all():
        data['bmi_group'] = data['bmi_custom']
        group_method = 'custom_28_35'
        use_custom = True
    else:
        data['bmi_group'] = pd.qcut(data['bmi'], 4, labels=['Q1','Q2','Q3','Q4'])
        group_method = 'quantile'
else:
    data['bmi_group'] = data['bmi_clin']
    group_method = 'clinical'

log(f"最终分组方法: {group_method}")
log("最终 bmi_group 唯一孕妇数:")
try:
    gc = data.groupby('bmi_group')['patient'].nunique().to_dict()
    log(str(gc))
except Exception as e:
    log("获取分组计数出错: " + str(e))

# 拟合 LMM
data['gest_day_c'] = data['gest_day'] - data['gest_day'].mean()
data['bmi_c'] = data['bmi'] - data['bmi'].mean()
mean_gest = data['gest_day'].mean()
mean_bmi = data['bmi'].mean()

formula = "yconc ~ gest_day_c + bmi_c"
re_formula = "~gest_day_c" if use_random_slope else None
log(f"拟合 LMM: formula={formula}, re_formula={re_formula}")

md = smf.mixedlm(formula, data, groups=data['patient'], re_formula=re_formula)
# ML拟合
mdf_ml = None
try:
    mdf_ml = md.fit(reml=False, method='lbfgs', maxiter=2000)
    log(f"ML 拟合收敛: {getattr(mdf_ml,'converged',None)}")
except Exception as e:
    log("ML 拟合（lbfgs）失败: " + str(e))
    try:
        mdf_ml = md.fit(reml=False, method='powell', maxiter=2000)
        log(f"ML 拟合（powell）收敛: {getattr(mdf_ml,'converged',None)}")
    except Exception as e2:
        log("ML 拟合（powell）也失败: " + str(e2))

if mdf_ml is None:
    log("错误: 无法拟合 ML LMM。终止 LMM 步骤。")
    save_text = "LMM ML 拟合失败; 终止。详见 run_log。"
    save_text(os.path.join(OUTDIR,"LMM_error.txt"), save_text)
    logf.close()
    raise RuntimeError("LMM ML 拟合失败; 终止流程。")

# 若收敛则尝试REML
mdf = mdf_ml
if getattr(mdf_ml, 'converged', False):
    try:
        tmp = md.fit(reml=True, method='lbfgs', maxiter=2000)
        if getattr(tmp, 'converged', False):
            mdf = tmp
            log("REML 拟合收敛，采用 REML 结果。")
        else:
            log("REML 未收敛，保留 ML 结果。")
    except Exception as e:
        log("REML 尝试失败，保留 ML。Err: " + str(e))
else:
    log("ML 未收敛，仍使用 ML 结果。")

# 保存摘要
lmm_summary_text = mdf.summary().as_text()
save_text(os.path.join(OUTDIR,"LMM_summary.txt"), lmm_summary_text)
log("已保存 LMM_summary.txt")
log("LMM 固定效应:\n" + str(mdf.fe_params.to_dict()))
log("随机效应协方差 (cov_re):\n" + str(mdf.cov_re))
log("残差方差 (scale): " + str(mdf.scale))

# 计算每人 t*
re_dict = mdf.random_effects
re_df = pd.DataFrame.from_dict(re_dict, orient='index').reset_index().rename(columns={'index':'patient'})
# 保证列存在
if 'Intercept' not in re_df.columns:
    re_df['Intercept'] = 0.0
if 'gest_day_c' not in re_df.columns:
    re_df['gest_day_c'] = 0.0

sub_info = data.groupby('patient').agg({'bmi':'mean','bmi_c':'mean','gest_day':'max'}).reset_index()
sub = sub_info.merge(re_df, on='patient', how='left')

sub = sub.merge(data[['patient','bmi_group']].drop_duplicates(), on='patient', how='left')

sub['Intercept'] = sub['Intercept'].fillna(0.0)
sub['gest_day_c'] = sub['gest_day_c'].fillna(0.0)

beta0 = mdf.fe_params['Intercept']
beta1 = mdf.fe_params['gest_day_c']
beta2 = mdf.fe_params['bmi_c']

sub['fixed_intercept'] = beta0 + beta2 * sub['bmi_c']
sub['fixed_slope'] = beta1
sub['b0j'] = sub['Intercept']
sub['b1j'] = sub['gest_day_c']

def compute_tstar(row):
    denom = row['fixed_slope'] + row['b1j']
    numer = THRESH - (row['fixed_intercept'] + row['b0j'])
    if pd.isna(denom) or denom <= 0:
        return np.nan
    return mean_gest + numer/denom

sub['t_star'] = sub.apply(compute_tstar, axis=1)
min_g = data['gest_day'].min()
sub['t_star_clean'] = sub['t_star'].where((sub['t_star']>=min_g) & (sub['t_star']<=300), np.nan)
sub.to_csv(os.path.join(OUTDIR,"per_patient_tstar.csv"), index=False)
log("已保存 per_patient_tstar.csv（含 t_star_clean 和 bmi_group）")

group_rows = []
for g, grp in sub.groupby('bmi_group'):
    vals = grp['t_star_clean'].dropna().values
    n = grp.shape[0]
    median_t = float(np.nanmedian(vals)) if vals.size>0 else np.nan
    p90_t = float(np.nanpercentile(vals, 90)) if vals.size>0 else np.nan
    group_rows.append({'bmi_group':g, 'n':int(n), 'median_tstar':median_t, 'p90_tstar':p90_t})
pd.DataFrame(group_rows).sort_values('bmi_group').to_csv(os.path.join(OUTDIR,"group_recommendations.csv"), index=False)
log("已保存 group_recommendations.csv")
log("分组推荐（打印）:")
log(str(pd.DataFrame(group_rows).sort_values('bmi_group')))


first_hit = data[data['yconc']>=THRESH].groupby('patient')['gest_day'].min().reset_index().rename(columns={'gest_day':'first_hit_day'})
last_obs = data.groupby('patient')['gest_day'].max().reset_index().rename(columns={'gest_day':'last_day'})
sub = sub.merge(first_hit, on='patient', how='left').merge(last_obs, on='patient', how='left')
sub['event'] = (~sub['first_hit_day'].isna()).astype(int)
sub['time_to_event'] = sub.apply(lambda r: r['first_hit_day'] if r['event']==1 else r['last_day'], axis=1)
sub[['patient','bmi','bmi_group','t_star_clean','time_to_event','event']].to_csv(os.path.join(OUTDIR,"per_patient_survival.csv"), index=False)
log("已保存 per_patient_survival.csv")

# 按 bmi_group 绘制 KM 曲线
try:
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8,6))
    for name, grp in sub.groupby('bmi_group'):
        if len(grp) < 5:
            log(f"跳过组 {name} (n={len(grp)}) <5")
            continue
        kmf.fit(grp['time_to_event'], event_observed=grp['event'], label=f"{name} (n={len(grp)})")
        kmf.plot_survival_function(ci_show=True)
    plt.title(f"KM: 尚未达到 Yconc >= {THRESH}")
    plt.xlabel("孕天数")
    plt.ylabel("生存概率（未达阈值）")
    plt.grid(True)
    plt.savefig(os.path.join(OUTDIR,"km_by_bmi_group.png"), dpi=150)
    plt.close()
    log("已保存 km_by_bmi_group.png")
except Exception as e:
    log("KM 绘图失败: " + str(e))
    traceback.print_exc(file=logf)


try:
    res_lr = multivariate_logrank_test(sub['time_to_event'], sub['bmi_group'], sub['event'])

    lr_text = f"multivariate_logrank_test: test_statistic={res_lr.test_statistic}, p_value={res_lr.p_value}, df={res_lr.degrees_of_freedom}"
    save_text(os.path.join(OUTDIR,"logrank_summary.txt"), lr_text)
    log("Log-rank 结果: " + lr_text)
except Exception as e:
    log("Log-rank 检验失败: " + str(e))
    traceback.print_exc(file=logf)

# Cox 回归
cox_df = sub[['time_to_event','event','bmi']].copy()
# 检查 Cox 数据质量
n_events = cox_df['event'].sum()
n_nonzero_var = cox_df['bmi'].var() > 0
log(f"Cox 诊断: n_events={n_events}, bmi 方差>0? {n_nonzero_var}")

cox_success = False
if n_events >= 10 and n_nonzero_var:
    try:
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col='time_to_event', event_col='event')
        cph.summary.to_csv(os.path.join(OUTDIR,"cox_summary.csv"))
        log("Cox 拟合成功（无惩罚项）。已保存摘要。")
        cox_success = True
    except Exception as e:
        log("Cox 初始拟合失败: " + str(e))
        traceback.print_exc(file=logf)
        # 尝试带惩罚项
        for pen in [0.1, 0.01, 0.001]:
            try:
                cph = CoxPHFitter(penalizer=pen)
                cph.fit(cox_df, duration_col='time_to_event', event_col='event')
                cph.summary.to_csv(os.path.join(OUTDIR,f"cox_summary_penalized_{pen}.csv"))
                log(f"Cox 拟合成功（惩罚项={pen}）。已保存摘要。")
                cox_success = True
                break
            except Exception as e2:
                log(f"Cox 带惩罚项拟合（pen={pen}）失败: {e2}")
                traceback.print_exc(file=logf)
else:
    log("Cox 未尝试，因事件数不足或 BMI 方差为零。")

if not cox_success:
    log("Cox 回归未成功。详见 run_log。跳过 Cox 输出。")

# 测量误差敏感性分析
log(f"开始测量误差敏感性分析: N_MC={N_MC}, SIM_TAUS={SIM_TAUS}")
rng = np.random.default_rng(123456)
sim_records = []
for tau in SIM_TAUS:
    log(f"模拟 tau={tau}")
    per_sub_med = []
    for idx, row in sub.iterrows():
        t_true = row['t_star']
        slope = row['fixed_slope'] + row['b1j']
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
    # 分组 p90
    grp_p90 = sub.groupby('bmi_group')[f'tstar_tau_{tau}'].apply(lambda x: np.nanpercentile(x.dropna(), 100*PCT_FOR_RECOMMEND) if x.dropna().size>0 else np.nan)
    for g, v in grp_p90.items():
        sim_records.append({'tau':tau, 'bmi_group':g, 'p90_tstar':float(v) if not pd.isna(v) else np.nan})
    log(f"完成 tau={tau}")

pd.DataFrame(sim_records).to_csv(os.path.join(OUTDIR,"measurement_error_sim_results.csv"), index=False)
sub.to_csv(os.path.join(OUTDIR,"per_patient_results_full.csv"), index=False)
log("已保存 measurement_error_sim_results.csv 和 per_patient_results_full.csv")

# 绘制敏感性分析图
try:
    sim_df = pd.DataFrame(sim_records)
    plt.figure(figsize=(8,6))
    for g, grp in sim_df.groupby('bmi_group'):
        plt.plot(grp['tau'], grp['p90_tstar'], marker='o', label=g)
    plt.xlabel("测量噪声 sigma (tau)")
    plt.ylabel(f"P{int(100*PCT_FOR_RECOMMEND)} t* (天)")
    plt.title("推荐 t* 对测量噪声的敏感性")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTDIR,"tstar_vs_tau.png"), dpi=150)
    plt.close()
    log("已保存 tstar_vs_tau.png")
except Exception as e:
    log("绘图 tstar_vs_tau 失败: " + str(e))
    traceback.print_exc(file=logf)

# 保存运行总结
summary = {
    'total_obs': int(len(data)),
    'n_groups': int(n_groups),
    'pct_ge3': float(pct_ge3),
    'pct_ge5': float(pct_ge5),
    'use_random_slope': bool(use_random_slope),
    'group_method': group_method,
    'N_MC': int(N_MC),
    'SIM_TAUS': SIM_TAUS,
    'n_events_first_hit': int(sub['event'].sum())
}
pd.Series(summary).to_csv(os.path.join(OUTDIR,"run_summary.csv"))
log("已保存 run_summary.csv: " + str(summary))

log("运行结束。请检查输出目录: " + OUTDIR)
logf.close()



print("666666666666666666666666666\n")
# 检查custom bins分组重复
data['bmi_custom'] = data['bmi'].apply(lambda b: '<28' if b<28 else ('28-30' if b<30 else ('30-32' if b<32 else ('32-35' if b<35 else '>=35'))))
custom_counts = data.groupby('bmi_custom')['patient'].nunique()
print("custom bins unique patient counts (观测级别统计，患者可能被计入多组):")
print(custom_counts.to_dict(), " sum=", sum(custom_counts))

patient_bins = data[['patient','bmi_custom']].drop_duplicates().groupby('patient')['bmi_custom'].nunique()
multi_bin_patients = (patient_bins > 1).sum()
print(f"有 {multi_bin_patients} 位孕妇至少被分到 2 个不同的 custom bins（因此会重复计数）。")
# 列出若干示例患者及其所在 bins
sample_multi = patient_bins[patient_bins>1].head(10)
print("示例（patient : bins_count）", sample_multi.to_dict())


# 计算每位患者的平均 BMI（也可改为 first BMI 或 pre-preg BMI 若可用）
patient_bmi = data.groupby('patient').agg(
    bmi_mean=('bmi','mean'),
    bmi_first=('bmi','first'),
    yconc_mean=('yconc','mean'),
    n_obs=('bmi','size')
).reset_index()

patient_bmi['bmi_for_group'] = patient_bmi['bmi_mean']

# 临床分组
def clinical_group(b):
    if b < 18.5: return 'Under18.5'
    if b < 24: return '18.5-24'
    if b < 28: return '24-28'
    if b < 32: return '28-32'
    return '>=32'
patient_bmi['group_clinical'] = patient_bmi['bmi_for_group'].apply(clinical_group)

# custom bins
def custom_group(b):
    if b < 28: return '<28'
    if b < 30: return '28-30'
    if b < 32: return '30-32'
    if b < 35: return '32-35'
    return '>=35'
patient_bmi['group_custom'] = patient_bmi['bmi_for_group'].apply(custom_group)

# quantile
patient_bmi['group_q'] = pd.qcut(patient_bmi['bmi_for_group'], 4, labels=['Q1','Q2','Q3','Q4'])
print("55555555555555555555555555\n")

# 打印每组的 BMI 范围
group_ranges = patient_bmi.groupby('group_q')['bmi_for_group'].agg(min_bmi='min', max_bmi='max').reset_index()

print("\n每组（Q1..Q4）的 BMI 范围（基于 patient-level 的 bmi_for_group）:")
for _, row in group_ranges.iterrows():
    # 若 group_q 为 category，会显示标签 'Q1'...'Q4'
    label = row['group_q']
    print(f"{label}: [{row['min_bmi']:.2f}, {row['max_bmi']:.2f}] (min, max)")

# 计算并打印 quantile 切点
qpoints = patient_bmi['bmi_for_group'].quantile([0.0, 0.25, 0.5, 0.75, 1.0])
print("\nQuantile 切点 (0%,25%,50%,75%,100%):")
print(qpoints.to_string())

# 保存 group_ranges 到 CSV
group_ranges.to_csv(os.path.join(OUTDIR, "patient_level_q_ranges.csv"), index=False)
print(f"\n已将 patient-level Q 范围保存到: {os.path.join(OUTDIR, 'patient_level_q_ranges.csv')}")

print("55555555555555555555555555\n")
# 统计每种方法下的 patient-level 分组计数
print("\npatient-level group counts (clinical):\n", patient_bmi['group_clinical'].value_counts().to_dict())
print("patient-level group counts (custom):\n", patient_bmi['group_custom'].value_counts().to_dict())
print("patient-level group counts (quantile Q1..Q4):\n", patient_bmi['group_q'].value_counts().to_dict())
print("总孕妇数:", patient_bmi.shape[0])

# 打印 quantile 切点
qtiles = patient_bmi['bmi_for_group'].quantile([0,0.25,0.5,0.75,1.0]).to_dict()
print("\npatient-level BMI quantile cut points (0,25%,50%,75%,100%):")
print(qtiles)

# 按 quantile 分组计算统计量
grouped = patient_bmi.groupby('group_q').agg(
    n=('patient','count'),
    bmi_min=('bmi_for_group','min'),
    bmi_q25=('bmi_for_group', lambda x: np.percentile(x,25)),
    bmi_median=('bmi_for_group','median'),
    bmi_q75=('bmi_for_group', lambda x: np.percentile(x,75)),
    bmi_max=('bmi_for_group','max'),
    yconc_mean=('yconc_mean','mean'),
    yconc_median=('yconc_mean','median'),
    yconc_min=('yconc_mean','min'),
    yconc_max=('yconc_mean','max')
).reset_index()
print("\nPer-group summary (quantile groups, patient-level):")
print(grouped.to_string(index=False))
print(grouped.to_string(index=False))
