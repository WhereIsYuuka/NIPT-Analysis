# -*- coding: utf-8 -*-
"""
按孕妇（个体）计算 Y染色体浓度 与其它数值特征 的相关性（Pearson/Spearman/Kendall）。
说明：请在下方 file_path 中填写你的 Excel 文件绝对路径（建议使用 r"..." 原始字符串）。
运行前请确保已安装： pandas, numpy, scipy, matplotlib
如未安装：pip install pandas numpy scipy matplotlib
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import sys

file_path = r"C:\Users\admin\Desktop\国赛论文\修改后附件男.xlsx"
sheet_index_or_name = 0   # 0表示第一个 sheet
mother_col = "孕妇代码"
y_col = "Y染色体浓度"

# 排除不做数值分析的列
exclude_cols = {"序号", "孕妇代码", "染色体的非整倍体"}
min_pairs = 3   # 计算个体内相关所需的最少配对数
out_dir = "y_conc_results"
os.makedirs(out_dir, exist_ok=True)


def to_numeric_col(s):
    # 尝试把Series转为数值，去掉千分号和百分号
    if s.dtype == object:
        s = s.astype(str).str.replace(',', '', regex=False).str.replace('%', '', regex=False).str.strip()
        s = s.replace({'': np.nan})
    return pd.to_numeric(s, errors='coerce')

def robust_read_excel(path, sheet):
    # 读取指定sheet，如果失败则读取第一个 sheet
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件：{path}")
    try:
        df = pd.read_excel(path, sheet_name=sheet)
        if isinstance(df, dict):
            first = list(df.keys())[0]
            print(f"警告:pd.read_excel 返回 dict,使用第一个 sheet: {first}")
            df = df[first]
        return df
    except Exception as e:
        try:
            all_sheets = pd.read_excel(path, sheet_name=None)
            first = list(all_sheets.keys())[0]
            print(f"读取指定 sheet 失败，改为读取第一个 sheet: {first} (错误: {e})")
            df = all_sheets[first]
            return df
        except Exception as e2:
            raise RuntimeError(f"读取 Excel 失败：{e2}")

def main():
    print("开始读取文件：", file_path)
    df = robust_read_excel(file_path, sheet_index_or_name)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    print("读取完成，表格形状:", df.shape)
    print("列名预览：")
    print(df.columns.tolist())

    if mother_col not in df.columns:
        raise ValueError(f"找不到孕妇 ID 列：'{mother_col}'。请确认列名。")
    if y_col not in df.columns:
        raise ValueError(f"找不到目标列：'{y_col}'。请确认列名。")

    df_num = df.copy()
    cols_try = [c for c in df.columns if c not in exclude_cols]
    for c in cols_try:
        try:
            df_num[c] = to_numeric_col(df_num[c])
        except Exception:
            print(f"警告：转换列为数值失败（保留原样）：{c}")

    if df_num[y_col].dtype.kind not in 'fiu':
        df_num[y_col] = to_numeric_col(df_num[y_col])
        if df_num[y_col].dtype.kind not in 'fiu':
            raise ValueError(f"目标列 {y_col} 不能转换为数值，请检查数据。")

    numeric_cols = df_num.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude_cols and c != y_col]
    print("\n将用于逐孕妇计算相关的数值特征（共 {} 个）：".format(len(features)))
    print(features)
    if len(features) == 0:
        raise ValueError("未识别到任何用于分析的数值特征。当前数值列为：\n" + str(numeric_cols))

    per_mother_rows = []
    mother_ids = df_num[mother_col].dropna().unique()
    print("\n检测到孕妇数:", len(mother_ids))
    for i, mom in enumerate(mother_ids):
        sub = df_num[df_num[mother_col] == mom]
        for feat in features:
            a = sub[[feat, y_col]].dropna()
            n = len(a)
            if n < min_pairs:
                per_mother_rows.append({
                    "mother_id": mom, "feature": feat, "n_pairs": n,
                    "pearson_r": np.nan, "pearson_p": np.nan,
                    "spearman_r": np.nan, "spearman_p": np.nan,
                    "kendall_r": np.nan, "kendall_p": np.nan
                })
                continue
            try:
                pr, pp = stats.pearsonr(a[feat], a[y_col])
            except Exception:
                pr, pp = np.nan, np.nan
            try:
                sr, sp = stats.spearmanr(a[feat], a[y_col])
            except Exception:
                sr, sp = np.nan, np.nan
            try:
                kr, kp = stats.kendalltau(a[feat], a[y_col])
            except Exception:
                kr, kp = np.nan, np.nan

            per_mother_rows.append({
                "mother_id": mom, "feature": feat, "n_pairs": n,
                "pearson_r": pr, "pearson_p": pp,
                "spearman_r": sr, "spearman_p": sp,
                "kendall_r": kr, "kendall_p": kp
            })
        if (i+1) % 100 == 0:
            print(f"已处理 {i+1} / {len(mother_ids)} 位孕妇")

    per_mother_df = pd.DataFrame(per_mother_rows)
    per_mother_csv = os.path.join(out_dir, "per_mother_correlations.csv")
    per_mother_df.to_csv(per_mother_csv, index=False, encoding='utf-8-sig')
    print("\n已保存:", per_mother_csv)

    summary_rows = []
    for feat in features:
        sub = per_mother_df[per_mother_df['feature'] == feat]
        valid = sub.dropna(subset=['pearson_r'])
        count_valid = len(valid)
        pear_mean = valid['pearson_r'].mean() if count_valid>0 else np.nan
        pear_med = valid['pearson_r'].median() if count_valid>0 else np.nan
        pear_sig_prop = (valid['pearson_p'] < 0.05).sum() / count_valid if count_valid>0 else np.nan

        sp_valid = sub.dropna(subset=['spearman_r'])
        sp_mean = sp_valid['spearman_r'].mean() if len(sp_valid)>0 else np.nan
        sp_med = sp_valid['spearman_r'].median() if len(sp_valid)>0 else np.nan
        sp_sig_prop = (sp_valid['spearman_p'] < 0.05).sum() / len(sp_valid) if len(sp_valid)>0 else np.nan

        kd_valid = sub.dropna(subset=['kendall_r'])
        kd_mean = kd_valid['kendall_r'].mean() if len(kd_valid)>0 else np.nan
        kd_med = kd_valid['kendall_r'].median() if len(kd_valid)>0 else np.nan
        kd_sig_prop = (kd_valid['kendall_p'] < 0.05).sum() / len(kd_valid) if len(kd_valid)>0 else np.nan

        summary_rows.append({
            "feature": feat,
            "n_mothers_with_valid_pearson": count_valid,
            "pearson_mean_r": pear_mean,
            "pearson_median_r": pear_med,
            "pearson_prop_p_lt_0_05": pear_sig_prop,
            "n_mothers_with_valid_spearman": len(sp_valid),
            "spearman_mean_r": sp_mean,
            "spearman_median_r": sp_med,
            "spearman_prop_p_lt_0_05": sp_sig_prop,
            "n_mothers_with_valid_kendall": len(kd_valid),
            "kendall_mean_r": kd_mean,
            "kendall_median_r": kd_med,
            "kendall_prop_p_lt_0_05": kd_sig_prop
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by='pearson_mean_r', key=lambda s: s.abs(), ascending=False)
    summary_csv = os.path.join(out_dir, "per_feature_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print("已保存：", summary_csv)

    agg_median = df_num.groupby(mother_col)[[y_col] + features].median().reset_index()
    agg_mean = df_num.groupby(mother_col)[[y_col] + features].mean().reset_index()
    agg_med_csv = os.path.join(out_dir, "per_mother_aggregated_median.csv")
    agg_mean_csv = os.path.join(out_dir, "per_mother_aggregated_mean.csv")
    agg_median.to_csv(agg_med_csv, index=False, encoding='utf-8-sig')
    agg_mean.to_csv(agg_mean_csv, index=False, encoding='utf-8-sig')
    print("已保存按母体汇总的数据(median / mean):", agg_med_csv, agg_mean_csv)

    group_level_rows = []
    for feat in features:
        a = agg_median[[feat, y_col]].dropna()
        if len(a) >= 3:
            pr, pp = stats.pearsonr(a[feat], a[y_col])
            sr, sp = stats.spearmanr(a[feat], a[y_col])
        else:
            pr, pp, sr, sp = [np.nan]*4
        group_level_rows.append({"feature": feat, "group_pearson_r": pr, "group_pearson_p": pp,
                                 "group_spearman_r": sr, "group_spearman_p": sp, "n_mothers": len(a)})
    group_level_df = pd.DataFrame(group_level_rows).sort_values(by='group_pearson_r', key=lambda s: s.abs(), ascending=False)
    group_level_csv = os.path.join(out_dir, "group_level_correlations_by_median.csv")
    group_level_df.to_csv(group_level_csv, index=False, encoding='utf-8-sig')
    print("已保存群体层面(按母体median)相关结果：", group_level_csv)

    plot_targets = []
    if '孕妇BMI' in features:
        plot_targets.append('孕妇BMI')
    if '检测孕周' in features:
        plot_targets.append('检测孕周')

    for feat in plot_targets:
        sub = per_mother_df[per_mother_df['feature'] == feat]
        vals = sub['pearson_r'].dropna()
        if len(vals) > 0:
            plt.figure(figsize=(6,4))
            plt.hist(vals, bins=20)
            plt.title(f"Every pregnant woman Pearson r: Y vs test_time")
            plt.xlabel("Pearson r")
            plt.ylabel("Number of pregnant women")
            fname = os.path.join(out_dir, f"per_mother_pearson_hist_{feat}.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            print("已保存图：", fname)
            plt.close()
        else:
            print(f"没有足够的每母体 Pearson 数据来绘制 {feat} 的分布图")

        a = agg_median[[feat, y_col]].dropna()
        if len(a) >= 3:
            plt.figure(figsize=(6,4))
            plt.scatter(a[feat], a[y_col], alpha=0.7, s=30)
            slope, intercept, r_value, p_value, std_err = stats.linregress(a[feat], a[y_col])
            xs = np.linspace(a[feat].min(), a[feat].max(), 100)
            plt.plot(xs, intercept + slope*xs, linestyle='--')
            plt.xlabel(feat)
            plt.ylabel("Number of pregnant women")
            plt.title(f"按母体median: c(y) vs {feat}\n Pearson r={r_value:.3f}, p={p_value:.3g}")
            fname2 = os.path.join(out_dir, f"per_mother_scatter_median_{feat}.png")
            plt.tight_layout()
            plt.savefig(fname2, dpi=150)
            print("已保存图：", fname2)
            plt.close()
        else:
            print(f"母体中位数层面样本太少，跳过 {feat} 的散点图")

    print("\n全部任务完成，结果保存在目录：", os.path.abspath(out_dir))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("脚本执行出错：", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
