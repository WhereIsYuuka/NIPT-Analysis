import pandas as pd
from datetime import datetime

def parse_date(date_str):
    """
    解析日期字符串
    返回datetime对象或None
    """
    if pd.isna(date_str):
        return None
    # 如果是数字格式，先转换为字符串
    if isinstance(date_str, (int, float)):
        date_str = str(int(date_str))
    formats = [
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%Y%m%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except ValueError:
            continue
    return None

# 读取Excel文件
file_path = r"C:\Users\admin\Desktop\国赛论文\修改后附件.xlsx"

df = pd.read_excel(file_path, sheet_name="女胎检测数据")
# 解析日期列
df['F_dates'] = df.iloc[:, 5].apply(parse_date)  # F列是第6列，索引为5
df['H_dates'] = df.iloc[:, 7].apply(parse_date)  # H列是第8列，索引为7

# 计算天数差
df['时间差'] = (df['H_dates'] - df['F_dates']).dt.days

# 替换F列并删除H列
df.iloc[:, 5] = df['时间差']  # 用天数差替换F列
df.drop(df.columns[7], axis=1, inplace=True)  # 删除H列
df.drop(['F_dates', 'H_dates', '时间差'], axis=1, inplace=True)  # 清理临时列

# 重命名F列
df.rename(columns={df.columns[5]: "检测日期同最后一次月经的时间差"}, inplace=True)

# 保存回Excel文件
df.to_excel(file_path, index=False)