import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
    HAVE_KDE = True
except Exception:
    HAVE_KDE = False

# 配置参数
DEFAULT_INPUT = Path(r"C:/Users/admin/Desktop/国赛论文/修改后附件女.xlsx")
TOLERANCE = 0.1
MIN_HEIGHT_M = 1.20
MAX_HEIGHT_M = 2.20
MIN_WEIGHT_KG = 30
MAX_WEIGHT_KG = 200
MIN_BMI = 10
MAX_BMI = 60
INSERT_AFTER_POS = 10  # 在原始列索引10（即J列之后）插入新列


def to_float_safe(x):
    try:
        if pd.isna(x):
            return float('nan')
        if isinstance(x, str):
            for u in ['cm','厘米','㎝','kg','KG','公斤','千克']:
                x = x.replace(u,'')
            x = x.strip()
        return float(x)
    except:
        return float('nan')

def find_all_cols_by_keywords(cols, keywords):
    res = []
    for i, c in enumerate(cols):
        low = str(c).lower()
        for kw in keywords:
            if kw in low:
                res.append(i)
                break
    return res

def compute_per_occurrence(df, height_indices, weight_indices, bmi_indices):
    """按出现序号逐次配对身高/体重/提供BMI 并计算 BMI 与差值。"""
    maxn = max(len(height_indices), len(weight_indices), len(bmi_indices))
    results = {}
    notes = []
    for i in range(maxn):
        h_idx = height_indices[i] if i < len(height_indices) else None
        w_idx = weight_indices[i] if i < len(weight_indices) else None
        b_idx = bmi_indices[i] if i < len(bmi_indices) else None
        nrows = len(df)
        height_cm = pd.Series([float('nan')]*nrows)
        weight_kg = pd.Series([float('nan')]*nrows)
        provided_bmi = pd.Series([float('nan')]*nrows)
        if h_idx is not None:
            height_cm = df.iloc[:, h_idx].apply(to_float_safe).reset_index(drop=True)
        if w_idx is not None:
            weight_kg = df.iloc[:, w_idx].apply(to_float_safe).reset_index(drop=True)
        if b_idx is not None:
            provided_bmi = df.iloc[:, b_idx].apply(to_float_safe).reset_index(drop=True)
        height_m = height_cm / 100.0
        bmi_raw = []
        for hh, ww in zip(height_m, weight_kg):
            try:
                if math.isnan(hh) or math.isnan(ww) or hh <= 0:
                    bmi_raw.append(float('nan'))
                else:
                    bmi_raw.append(ww / (hh * hh))
            except:
                bmi_raw.append(float('nan'))
        bmi_raw = pd.Series(bmi_raw)
        bmi_rounded = bmi_raw.round(ROUND_DIGITS)
        comp_col = f"自主测试BMI_{i+1}"
        diff_col = f"自主评测差值_{i+1}"
        diff_series = bmi_rounded - provided_bmi
        results[comp_col] = bmi_rounded
        results[diff_col] = diff_series
        if (h_idx is None) or (w_idx is None) or (b_idx is None):
            notes.append(f"measurement #{i+1} pairing incomplete (h:{h_idx is not None}, w:{w_idx is not None}, bmi:{b_idx is not None})")
    return results, notes

def insert_columns_after_position(df, insert_pos, new_cols_dict):
    """在指定索引后插入新列（保持新列顺序），允许重复列名。"""
    df_new = df.copy()
    for name, series in new_cols_dict.items():
        df_new[name] = series.values if isinstance(series, (pd.Series, np.ndarray)) else series
    cols = list(df_new.columns)
    original_col_count = len(cols) - len(new_cols_dict)
    if insert_pos >= original_col_count:
        return df_new
    new_names = list(new_cols_dict.keys())
    original_cols = list(df.columns)
    left = original_cols[:insert_pos]
    right = original_cols[insert_pos:]
    new_order = left + new_names + right
    df_new = df_new.reindex(columns=new_order)
    return df_new

def compute_check_status_for_all(df, comp_cols_prefix="自主测试BMI_", diff_prefix="自主评测差值_"):
    """对每个差值列判定行级状态并汇总。"""
    comp_cols = [c for c in df.columns if str(c).startswith(comp_cols_prefix)]
    diff_cols = [c for c in df.columns if str(c).startswith(diff_prefix)]
    per_measure_status = {}
    for comp, diff in zip(comp_cols, diff_cols):
        comp_series = df[comp]
        diff_series = df[diff]
        status = []
        for cval, dval in zip(comp_series, diff_series):
            if pd.isna(dval) and pd.isna(cval):
                status.append('Missing_height_or_weight')
                continue
            if pd.isna(dval) and not pd.isna(cval):
                status.append('Missing_provided_BMI')
                continue
            if pd.isna(cval):
                status.append('Missing_height_or_weight')
                continue
            if cval < MIN_BMI or cval > MAX_BMI:
                status.append('Implausible_BMI')
                continue
            status.append('OK' if abs(dval) <= TOLERANCE else 'Mismatch')
        per_measure_status[comp] = pd.Series(status, index=df.index)
    agg_counts = {}
    for s in per_measure_status.values():
        for k, v in s.value_counts().to_dict().items():
            agg_counts[k] = agg_counts.get(k, 0) + v
    per_row_summary = []
    for i in df.index:
        row_counts = {}
        for comp, s in per_measure_status.items():
            val = s.iloc[i]
            row_counts[val] = row_counts.get(val, 0) + 1
        parts = [f"{k}:{v}" for k, v in row_counts.items()]
        per_row_summary.append(";".join(parts) if parts else "")
    df['per_row_status_summary'] = per_row_summary
    return agg_counts, per_measure_status

def make_overview_plot_all(diffs_all, comps_all, prov_all, out_png_path):
    """绘制2x2汇总图：六边形密度/Bland-Altman/差值分布/状态计数。"""
    diffs = np.array(diffs_all)
    calc = np.array(comps_all)
    prov = np.array(prov_all)
    mean_vals = (calc + prov) / 2.0

    fig, axes = plt.subplots(2,2, figsize=(13,10))
    ax1 = axes[0,0]; ax2 = axes[0,1]; ax3 = axes[1,0]; ax4 = axes[1,1]

    # 六边形密度图
    try:
        hb = ax1.hexbin(calc, prov, gridsize=60, mincnt=1, cmap='Blues', bins='log')
        fig.colorbar(hb, ax=ax1)
    except Exception:
        ax1.scatter(calc, prov, s=6, alpha=0.5)
    mn = np.nanmin(np.concatenate([calc, prov])); mx = np.nanmax(np.concatenate([calc, prov])); pad = 1.0
    ax1.plot([mn-pad, mx+pad], [mn-pad, mx+pad], linestyle='--', linewidth=1, label='y = x')
    ax1.plot([mn-pad, mx+pad], [mn-pad + TOLERANCE, mx+pad + TOLERANCE], linestyle=':', linewidth=1, alpha=0.8, label=f'±{TOLERANCE} tol')
    ax1.plot([mn-pad, mx+pad], [mn-pad - TOLERANCE, mx+pad - TOLERANCE], linestyle=':', linewidth=1, alpha=0.8)
    ax1.set_xlabel('Calculated BMI')
    ax1.set_ylabel('Provided BMI')
    ax1.set_title('Density hexbin: Calculated vs Provided BMI')
    ax1.legend(loc='upper left')

    # 标记绝对差值最大的前若干个点
    absdiff_idx = np.argsort(np.abs(diffs))[::-1] if len(diffs)>0 else np.array([])
    topk = min(10, len(absdiff_idx))
    if topk > 0:
        ax1.scatter(calc[absdiff_idx[:topk]], prov[absdiff_idx[:topk]], facecolors='none', edgecolors='red', s=90, linewidths=1.2, label='Top mismatches')
        ax1.legend()

    # 聚合图
    mean_diff = np.nanmean(diffs) if len(diffs)>0 else 0.0
    sd_diff = np.nanstd(diffs, ddof=1) if len(diffs)>1 else 0.0
    loa_upper = mean_diff + 1.96*sd_diff
    loa_lower = mean_diff - 1.96*sd_diff
    ax2.scatter(mean_vals, diffs, s=8, alpha=0.6)
    ax2.axhline(mean_diff, color='black', linestyle='--', linewidth=1, label=f'Mean diff = {mean_diff:.2f}')
    ax2.axhline(loa_upper, color='red', linestyle=':', linewidth=1, label=f'+1.96σ = {loa_upper:.2f}')
    ax2.axhline(loa_lower, color='red', linestyle=':', linewidth=1, label=f'-1.96σ = {loa_lower:.2f}')
    ax2.axhline(TOLERANCE, color='grey', linestyle='-.', linewidth=1, label=f'±Tolerance = {TOLERANCE}')
    ax2.axhline(-TOLERANCE, color='grey', linestyle='-.', linewidth=1)
    ax2.set_xlabel('(Calculated + Provided)/2')
    ax2.set_ylabel('Difference (Calc - Prov)')
    ax2.set_title('Bland–Altman (aggregated)')
    ax2.legend(loc='upper right', fontsize='small')
    outside_tol = np.sum(np.abs(diffs) > TOLERANCE)
    ax2.text(0.02, 0.95, f'Entries = {len(diffs)}\nOutside tolerance = {int(outside_tol)}', transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))

    # 差值直方图
    ax3.hist(diffs, bins=60, density=True, alpha=0.6)
    if HAVE_KDE and len(diffs[~np.isnan(diffs)])>2:
        try:
            kde = gaussian_kde(diffs[~np.isnan(diffs)])
            xs = np.linspace(np.nanmin(diffs)-0.5, np.nanmax(diffs)+0.5, 300)
            ax3.plot(xs, kde(xs), linestyle='-', linewidth=1.2)
        except:
            pass
    ax3.axvline(TOLERANCE, color='grey', linestyle='-.', linewidth=1)
    ax3.axvline(-TOLERANCE, color='grey', linestyle='-.', linewidth=1)
    within_pct = np.sum(np.abs(diffs) <= TOLERANCE) / len(diffs) * 100 if len(diffs)>0 else 0.0
    ax3.set_title(f'Difference distribution: {within_pct:.1f}% within ±{TOLERANCE}')
    ax3.set_xlabel('Difference (Calc - Prov)')
    ax3.set_ylabel('Density')

    # 差值分类计数柱状图
    statuses = []
    for d in diffs:
        if np.isnan(d):
            statuses.append('Missing')
        elif abs(d) <= TOLERANCE:
            statuses.append('OK')
        else:
            statuses.append('Mismatch')
    import collections
    st_counts = collections.Counter(statuses)
    labels = list(st_counts.keys())
    vals = [st_counts[k] for k in labels]
    ax4.bar(labels, vals, color='tab:blue')
    for i, v in enumerate(vals):
        ax4.text(i, v + max(1, int(len(diffs)*0.01)), str(int(v)), ha='center')
    ax4.set_title('Aggregated status counts')
    ax4.set_ylabel('Count')

    plt.tight_layout()
    fig.savefig(out_png_path, dpi=200)
    plt.close(fig)

def main(input_path: Path):
    # 读取表
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    sheets = pd.read_excel(input_path, sheet_name=None, engine='openpyxl')
    sheet_names = list(sheets.keys())
    first = sheet_names[0]
    df = sheets[first].copy()

    cols = list(df.columns)
    height_idxs = find_all_cols_by_keywords(cols, ['身高','height'])
    weight_idxs = find_all_cols_by_keywords(cols, ['体重','weight'])
    bmi_idxs = find_all_cols_by_keywords(cols, ['孕妇bmi','孕期bmi','孕妇 bmi','bmi'])

    if len(height_idxs)==0 and len(weight_idxs)==0 and len(bmi_idxs)==0:
        raise RuntimeError("No height/weight/bmi-like columns found. Columns: " + str(cols))

    # 逐序号计算BMI与差值
    newcols_dict, pairing_notes = compute_per_occurrence(df, height_idxs, weight_idxs, bmi_idxs)

    # 插入新列
    df_with_new = insert_columns_after_position(df, INSERT_AFTER_POS, newcols_dict)

    # 保留表
    base = input_path.with_suffix('')
    out_xlsx_path = input_path.with_name(input_path.stem + "_with_checks" + input_path.suffix)
    with pd.ExcelWriter(out_xlsx_path, engine='openpyxl') as writer:
        df_with_new.to_excel(writer, sheet_name=first, index=False)
    # 新增review工作表
        diff_cols = [c for c in df_with_new.columns if str(c).startswith("自主评测差值_")]
        if diff_cols:
            mask_list = []
            for dcol in diff_cols:
                s = df_with_new[dcol]
                mask = s.isna() | (s.abs() > TOLERANCE)
                mask_list.append(mask)
            # 合并掩码
            combined_mask = pd.Series(False, index=df_with_new.index)
            for m in mask_list:
                combined_mask = combined_mask | m
            review_df = df_with_new.loc[combined_mask].copy()
        else:
            review_df = pd.DataFrame(columns=df_with_new.columns)
        review_df.to_excel(writer, sheet_name='review_rows', index=False)

        for name in sheet_names:
            if name == first: continue
            sheets[name].to_excel(writer, sheet_name=name, index=False)
    print("Wrote annotated workbook to:", out_xlsx_path)

    # 汇总数组
    diffs_all = []
    comps_all = []
    prov_all = []
    comp_cols = [c for c in df_with_new.columns if str(c).startswith("自主测试BMI_")]
    diff_cols = [c for c in df_with_new.columns if str(c).startswith("自主评测差值_")]

    prov_candidates = [c for c in df_with_new.columns if str(c) == '孕妇BMI_provided' or '孕妇bmi' in str(c).lower() or str(c).lower()=='bmi']

    for comp_col, diff_col in zip(comp_cols, diff_cols):
        comp_vals = df_with_new[comp_col].to_numpy()
        diff_vals = df_with_new[diff_col].to_numpy()
        prov_vals = comp_vals - diff_vals 

        mask = (~np.isnan(comp_vals)) & (~np.isnan(prov_vals))
        if np.sum(mask) > 0:
            comps_all.extend(comp_vals[mask].tolist())
            prov_all.extend(prov_vals[mask].tolist())
            diffs_all.extend(diff_vals[mask].tolist())

    # 若无有效数据则跳过绘图
    if len(diffs_all) == 0:
        print("No computed vs provided pairs found for plotting (all missing). Skipping plot generation.")
        return

    out_png_path = input_path.with_name(input_path.stem + "_bmi_overview.png")
    make_overview_plot_all(diffs_all, comps_all, prov_all, out_png_path)
    print("Saved overview PNG to:", out_png_path)

    if pairing_notes:
        print("Pairing notes:")
        for n in pairing_notes:
            print(" -", n)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inp = Path(sys.argv[1])
    else:
        inp = DEFAULT_INPUT
    main(inp)
