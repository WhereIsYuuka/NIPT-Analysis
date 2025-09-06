import matplotlib
matplotlib.use('Agg')  
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

DATA_PATH = r"C:\Users\admin\Desktop\国赛论文\修改后附件男.xlsx"   
OUTDIR = r"C:\Users\admin\PycharmProjects\pythonProject1\2025数模国赛\问题一模型附件"     
os.makedirs(OUTDIR, exist_ok=True)

# 读取
df = pd.read_excel(DATA_PATH, engine='openpyxl')
# 选列
data = df[['孕妇代码','检测孕周','孕妇BMI','Y染色体浓度']].copy()
data.columns = ['patient','gest_week','bmi','yconc']

# 保留数值有效行
for c in ['gest_week','bmi','yconc']:
    data[c] = pd.to_numeric(data[c], errors='coerce')
data = data.dropna(subset=['patient','gest_week','bmi','yconc']).reset_index(drop=True)

# 全局中心化
data['gest_week_c'] = data['gest_week'] - data['gest_week'].mean()
data['bmi_c'] = data['bmi'] - data['bmi'].mean()

# data['gest_week_s'] = (data['gest_week'] - data['gest_week'].mean())/data['gest_week'].std()
# data['bmi_s'] = (data['bmi'] - data['bmi'].mean())/data['bmi'].std()

# 随机截距 + 随机斜率
formula = "yconc ~ gest_week_c + bmi_c"
re_formula = "~gest_week_c"   
model_ok = None
results = {}

try:
    md = smf.mixedlm(formula, data, groups=data['patient'], re_formula=re_formula)
    try:
        mdf = md.fit(reml=False, method='lbfgs', maxiter=2000)
    except Exception:
        mdf = md.fit(reml=False, method='powell', maxiter=2000)
    model_ok = mdf
    print("Random-slope (gest_week) model fitted. Converged:", getattr(mdf, 'converged', None))
except Exception as e:
    print("Random-slope model failed:", e)
    model_ok = None

# 如果失败，退回随机截距模型并用 REML + 不同优化器重拟合
if model_ok is None or not getattr(model_ok, 'converged', True):
    try:
        md0 = smf.mixedlm(formula, data, groups=data['patient'])
        mdf0 = md0.fit(reml=True, method='lbfgs', maxiter=2000)
        model_ok = mdf0
        print("Random-intercept model fitted with REML. Converged:", getattr(mdf0,'converged',None))
    except Exception as e:
        print("Random-intercept REML fit failed:", e)
        # 最后退回 to ML fit without re_formula
        md0b = smf.mixedlm(formula, data, groups=data['patient'])
        mdf0b = md0b.fit(reml=False, method='nm', maxiter=2000)
        model_ok = mdf0b
        print("Fallback random-intercept model (nm) fitted. Converged:", getattr(mdf0b,'converged',None))

# 输出并保存所有关键参数
if model_ok is None:
    raise RuntimeError("所有模型拟合都失败，请检查数据或与我沟通进一步策略。")

# 打印收敛标志
print("模型收敛状态:", getattr(model_ok, 'converged', None))
results['converged'] = getattr(model_ok, 'converged', None)

# 固定效应
fe = model_ok.fe_params
fe_se = model_ok.bse_fe if hasattr(model_ok, 'bse_fe') else model_ok.bse
print("Fixed effects:\n", fe)
results['fe_params'] = fe
results['fe_se'] = fe_se

# 随机效应协方差矩阵
cov_re = model_ok.cov_re
print("Random effects covariance (cov_re):\n", cov_re)
results['cov_re'] = cov_re

# 残差方差
resid_var = model_ok.scale
print("Residual variance (scale):", resid_var)
results['resid_var'] = resid_var

results['llf'] = model_ok.llf
results['aic'] = model_ok.aic
results['bic'] = model_ok.bic
print("llf, AIC, BIC:", model_ok.llf, model_ok.aic, model_ok.bic)

re_df = pd.DataFrame.from_dict(model_ok.random_effects, orient='index').reset_index().rename(columns={'index':'patient'})
re_df.to_csv(os.path.join(OUTDIR, "random_effects_per_patient_refit.csv"), index=False)
print("Saved BLUPs to random_effects_per_patient_refit.csv")

X_fe = sm.add_constant(data[['gest_week_c','bmi_c']], has_constant='add')
pred_fixed = np.dot(X_fe, fe.values)
var_fixed = np.var(pred_fixed, ddof=0)

rand_pred = []
for i,row in data.iterrows():
    pid = row['patient']
    r = model_ok.random_effects.get(pid, None)
    rp = 0.0
    if r is not None:
        if 'Intercept' in r.index:
            rp += r['Intercept']
        if 'gest_week_c' in r.index:
            rp += r['gest_week_c'] * row['gest_week_c']
        if 'bmi_c' in r.index:
            rp += r['bmi_c'] * row['bmi_c']
    rand_pred.append(rp)
rand_pred = np.array(rand_pred)
var_random = np.var(rand_pred, ddof=0)

marginal_R2 = var_fixed / (var_fixed + var_random + resid_var)
conditional_R2 = (var_fixed + var_random) / (var_fixed + var_random + resid_var)
results.update({'var_fixed':var_fixed,'var_random':var_random,'marginal_R2':marginal_R2,'conditional_R2':conditional_R2})
print("marginal R2:", marginal_R2, "conditional R2:", conditional_R2)

try:
    tmp = np.array(cov_re)
    intercept_var = tmp[0,0] if tmp.size>0 else float(cov_re)
    ICC = intercept_var / (intercept_var + resid_var)
    results['ICC'] = ICC
    print("Approx ICC:", ICC)
except Exception as e:
    print("计算 ICC 出错：", e)

pred_mixed = pred_fixed + rand_pred
rmse = mean_squared_error(data['yconc'], pred_mixed, squared=False)
mae = mean_absolute_error(data['yconc'], pred_mixed)
print("On-sample RMSE, MAE:", rmse, mae)
results['rmse'] = rmse
results['mae'] = mae

# 保存
with open(os.path.join(OUTDIR,"mixed_model_key_results.txt"), "w", encoding='utf-8') as f:
    f.write("Converged: %s\n\n" % str(results['converged']))
    f.write("Fixed effects:\n%s\n\n" % str(fe))
    f.write("Fixed effects SE:\n%s\n\n" % str(fe_se))
    f.write("Random effects cov (cov_re):\n%s\n\n" % str(cov_re))
    f.write("Residual variance (scale): %s\n" % str(resid_var))
    f.write("llf, AIC, BIC: %s, %s, %s\n" % (results['llf'], results['aic'], results['bic']))
    f.write("var_fixed, var_random, marginal_R2, conditional_R2: %s\n" % str((var_fixed,var_random,marginal_R2,conditional_R2)))
    f.write("ICC (approx): %s\n" % str(results.get('ICC',None)))
    f.write("RMSE, MAE: %s, %s\n" % (rmse, mae))

print("Saved summary to:", os.path.join(OUTDIR,"mixed_model_key_results.txt"))
