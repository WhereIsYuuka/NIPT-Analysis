import openpyxl

file_path = r"C:\Users\admin\Desktop\国赛论文\修改后附件.xlsx"
wb = openpyxl.load_workbook(file_path)
ws = wb.active

for row in ws.iter_rows(min_row=2, min_col=29, max_col=29):
    cell = row[0]
    if cell.value == '≥3':
        cell.value = 3

wb.save(file_path)
print('AC 列 ≥3 已替换为 3，保存完成。')