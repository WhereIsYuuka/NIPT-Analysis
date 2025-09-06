import openpyxl
import re

# 文件路径
file_path = r"C:\Users\admin\Desktop\国赛论文\修改后附件.xlsx"

# 加载工作簿和工作表
wb = openpyxl.load_workbook(file_path)
ws = wb.active

# 遍历第J列，从第2行开始
for row in ws.iter_rows(min_row=2, min_col=10, max_col=10):
    cell = row[0]
    if cell.value:
        original = str(cell.value).strip()
        match = re.match(r'(\d+)w\s*\+?\s*(\d*)', original, re.I)
        if match:
            weeks = int(match.group(1))
            days = int(match.group(2)) if match.group(2) else 0
            total_days = weeks * 7 + days
            cell.value = total_days

# 保存回原文件
wb.save(file_path)
print("处理完成，已保存回原文件。")