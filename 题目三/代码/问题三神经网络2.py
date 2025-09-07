# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random, numpy as np, pandas as pd, torch, torch.nn as nn, math
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix

seed_everything = lambda seed: (random.seed(seed), np.random.seed(seed),
                                torch.manual_seed(seed), torch.cuda.manual_seed_all(seed))
seed_everything(42)


DATA_PATH   = r"C:/Users/admin/Desktop/国赛论文/修改后附件男.xlsx"
BATCH_SIZE  = 256
EPOCHS      = 1000
LR          = 1e-3
WEIGHT_DECAY= 1e-4
PATIENCE    = 30
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GROUP_NUM   = 4


def load_data(path: str):
    df = pd.read_excel(path)
    use_cols = ['检测孕周', '生产次数', '年龄', '孕妇BMI', '身高', 'Y染色体浓度']
    df = df[use_cols].dropna()
    df['nipt_day'] = df['检测孕周']         
    df['group_id'] = pd.qcut(df['孕妇BMI'], q=GROUP_NUM, labels=False, duplicates='drop')
    edges = df.groupby('group_id')['孕妇BMI'].agg(['min', 'max']).reset_index()
    edges.to_csv('bmi_edges.csv', index=False)
    return df, edges

print('Loading data...')
df_all, bmi_edges = load_data(DATA_PATH) 
print('Data shape:', df_all.shape)
print('BMI edges\n', bmi_edges)


X = df_all[['检测孕周', '生产次数', '年龄', '孕妇BMI', '身高']].values
y_reg = df_all['nipt_day'].values.reshape(-1, 1)
y_clf = df_all['group_id'].astype(int).values

x_scaler = StandardScaler()
X = x_scaler.fit_transform(X)


X_train, X_temp, y_reg_tr, y_reg_te, y_clf_tr, y_clf_te = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
X_val, X_test, y_reg_val, y_reg_test, y_clf_val, y_clf_test = train_test_split(
    X_temp, y_reg_te, y_clf_te, test_size=0.5, random_state=42, stratify=y_clf_te)

# 划分结果
class NIPTDataset(Dataset):
    def __init__(self, x, y_reg, y_clf):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        self.y_clf = torch.tensor(y_clf, dtype=torch.long)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y_reg[idx], self.y_clf[idx]

train_set = NIPTDataset(X_train, y_reg_tr, y_clf_tr)
val_set   = NIPTDataset(X_val,   y_reg_val, y_clf_val)
test_set  = NIPTDataset(X_test,  y_reg_test, y_clf_test)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)


class MLPRegClf(nn.Module):
    def __init__(self, input_dim=5, hidden=64, n_groups=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU()
        )
        self.tower_reg = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        self.tower_clf = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, n_groups))

    def forward(self, x):
        feat = self.shared(x)
        return self.tower_reg(feat), self.tower_clf(feat)

model = MLPRegClf(n_groups=GROUP_NUM).to(DEVICE)
print('\n--- Model architecture ---')
print(model)
print('Params:', sum(p.numel() for p in model.parameters()))


reg_loss_fn = nn.MSELoss()
clf_loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

best_val, patience_cnt = 1e9, 0

# 存储历史用于可视化
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': [],
    'train_mae': [], 'val_mae': []
}

for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    model.train()
    epoch_tr_loss = 0.
    tr_true_cls = []
    tr_pred_cls = []
    tr_true_reg = []
    tr_pred_reg = []

    for xb, y_regb, y_clfb in train_loader:
        xb, y_regb, y_clfb = xb.to(DEVICE), y_regb.to(DEVICE), y_clfb.to(DEVICE)
        pred_reg, pred_clf = model(xb)
        loss = reg_loss_fn(pred_reg, y_regb) + clf_loss_fn(pred_clf, y_clfb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        epoch_tr_loss += loss.item() * xb.size(0)


        tr_true_cls.append(y_clfb.cpu())
        tr_pred_cls.append(torch.argmax(pred_clf, dim=1).cpu())
        tr_true_reg.append(y_regb.cpu())
        tr_pred_reg.append(pred_reg.detach().cpu())

    avg_tr_loss = epoch_tr_loss / len(train_set)

    tr_true_cls = torch.cat(tr_true_cls).numpy()
    tr_pred_cls = torch.cat(tr_pred_cls).numpy()
    tr_true_reg = torch.cat(tr_true_reg).numpy().ravel()
    tr_pred_reg = torch.cat(tr_pred_reg).numpy().ravel()
    train_acc = accuracy_score(tr_true_cls, tr_pred_cls)
    train_mae = mean_absolute_error(tr_true_reg, tr_pred_reg)

    history['train_loss'].append(avg_tr_loss)
    history['train_acc'].append(train_acc)
    history['train_mae'].append(train_mae)

    model.eval()
    epoch_val_loss = 0.
    val_true_cls = []
    val_pred_cls = []
    val_true_reg = []
    val_pred_reg = []
    with torch.no_grad():
        for xb, y_regb, y_clfb in val_loader:
            xb, y_regb, y_clfb = xb.to(DEVICE), y_regb.to(DEVICE), y_clfb.to(DEVICE)
            pred_reg, pred_clf = model(xb)
            epoch_val_loss += (reg_loss_fn(pred_reg, y_regb) + clf_loss_fn(pred_clf, y_clfb)).item() * xb.size(0)

            val_true_cls.append(y_clfb.cpu())
            val_pred_cls.append(torch.argmax(pred_clf, dim=1).cpu())
            val_true_reg.append(y_regb.cpu())
            val_pred_reg.append(pred_reg.detach().cpu())

    avg_val_loss = epoch_val_loss / len(val_set)
    val_true_cls = torch.cat(val_true_cls).numpy()
    val_pred_cls = torch.cat(val_pred_cls).numpy()
    val_true_reg = torch.cat(val_true_reg).numpy().ravel()
    val_pred_reg = torch.cat(val_pred_reg).numpy().ravel()
    val_acc = accuracy_score(val_true_cls, val_pred_cls)
    val_mae = mean_absolute_error(val_true_reg, val_pred_reg)

    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(val_acc)
    history['val_mae'].append(val_mae)

    scheduler.step(avg_val_loss)
    epoch_time = time.time() - t0
    print(f'Epoch {epoch:03d} | train Loss: {avg_tr_loss:.4f} | val Loss: {avg_val_loss:.4f} ' \
          f'| train Acc: {train_acc:.4f} | val Acc: {val_acc:.4f} ' \
          f'| train MAE: {train_mae:.2f} 天 | val MAE: {val_mae:.2f} 天 | time: {epoch_time:.1f}s')

    if avg_val_loss < best_val:
        best_val = avg_val_loss
        torch.save(model.state_dict(), 'best_weight.pt')
        patience_cnt = 0
    else:
        patience_cnt += 1
    if patience_cnt >= PATIENCE:
        print(f'Early stop at epoch {epoch}'); break

# 测试评估
model.load_state_dict(torch.load('best_weight.pt', map_location=DEVICE))
model.eval()
y_reg_true, y_reg_pred = [], []
y_clf_true, y_clf_pred = [], []
with torch.no_grad():
    for xb, y_regb, y_clfb in test_loader:
        xb = xb.to(DEVICE)
        pred_reg, pred_clf = model(xb)
        y_reg_pred.append(pred_reg.detach().cpu())
        y_clf_pred.append(torch.argmax(pred_clf, dim=1).cpu())
        y_reg_true.append(y_regb)
        y_clf_true.append(y_clfb)

y_reg_pred = torch.cat(y_reg_pred).numpy().ravel()
y_reg_true = torch.cat(y_reg_true).numpy().ravel()
y_clf_pred = torch.cat(y_clf_pred).numpy().ravel()
y_clf_true = torch.cat(y_clf_true).numpy().ravel()

mae  = mean_absolute_error(y_reg_true, y_reg_pred)
rmse = math.sqrt(mean_squared_error(y_reg_true, y_reg_pred))
acc  = accuracy_score(y_clf_true, y_clf_pred)
print('\n>>> Test Regression  MAE = %.2f 天  RMSE = %.2f 天' % (mae, rmse))
print('>>> Test Classification Accuracy = %.3f' % acc)

# 训练集真实分组规则
train_rules = (df_all.iloc[:len(y_reg_tr)]
               .groupby('group_id', as_index=False)
               .agg(bmi_min=('孕妇BMI', 'min'),
                    bmi_max=('孕妇BMI', 'max'),
                    bmi_mean=('孕妇BMI', 'mean'),
                    nipt_true_day=('nipt_day', 'mean'),
                    nipt_std_day=('nipt_day', 'std'))
               .round(1))
print('>>> 训练集真实分组规则（单位：天）')
print(train_rules.set_index('group_id'))
print()  

# 测试集预测分组规则
def bmi_to_group_fixed(bmi, edges_df):
    for _, row in edges_df.iterrows():
        if row['min'] <= bmi <= row['max']:
            return int(row['group_id'])
    return int(edges_df.iloc[-1]['group_id'])

# 取测试集长度对应的尾部样本用于比较（与原逻辑保持一致）
df_test = df_all.iloc[-len(y_reg_pred):].copy()
df_test['pred_group'] = df_test['孕妇BMI'].apply(lambda x: bmi_to_group_fixed(x, bmi_edges))
df_test['pred_nipt']  = y_reg_pred

group_rules = (df_test.groupby('pred_group', as_index=False)
                      .agg(bmi_min=('孕妇BMI', 'min'),
                           bmi_max=('孕妇BMI', 'max'),
                           bmi_mean=('孕妇BMI', 'mean'),
                           nipt_mean=('pred_nipt', 'mean'),
                           nipt_std=('pred_nipt', 'std'))
                      .round(1))
print('>>> 推荐分组规则（测试集，无重叠，单位：天）')
print(group_rules.set_index('pred_group'))

# 蒙特卡洛噪声测试
def monte_carlo_noise(loader, model, noise_std=0.01, n_round=100):
    errs = []
    with torch.no_grad():
        for _ in range(n_round):
            y_tmp = []
            for xb, _, _ in loader:
                xb_noisy = xb + torch.randn_like(xb) * noise_std
                pred_reg, _ = model(xb_noisy.to(DEVICE))
                y_tmp.append(pred_reg.detach().cpu())
            y_tmp = torch.cat(y_tmp).numpy().ravel()
            errs.append(mean_absolute_error(y_reg_true, y_tmp))
    return np.array(errs)

mc_err = monte_carlo_noise(test_loader, model, noise_std=0.01)
print('\n>>> 检测误差(±1 %%) 带来的 MAE 波动：%.2f ± %.2f 天' % (mc_err.mean(), mc_err.std()))

# 可视化
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

epochs_range = range(1, len(history['train_loss'])+1)
plt.figure(figsize=(12, 9))

# Loss
plt.subplot(3, 1, 1)
plt.plot(epochs_range, history['train_loss'], label='Train Loss', color=colors[0], linewidth=2)
plt.plot(epochs_range, history['val_loss'], label='Val Loss', color=colors[1], linewidth=2)
plt.title('Loss / Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(3, 1, 2)
plt.plot(epochs_range, history['train_acc'], label='Train Acc', color=colors[2], linewidth=2)
plt.plot(epochs_range, history['val_acc'], label='Val Acc', color=colors[3], linewidth=2)
plt.title('Accuracy / Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# MAE
plt.subplot(3, 1, 3)
plt.plot(epochs_range, history['train_mae'], label='Train MAE', color=colors[4], linewidth=2)

plt.plot(epochs_range, history['val_mae'], label='Val MAE', color='tab:brown' if 'tab:brown' in plt.colormaps() else 'black', linewidth=2)
plt.title('MAE (days) / Epoch')
plt.xlabel('Epoch')
plt.ylabel('MAE (days)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('metrics_curves.png', dpi=200)
print("Saved metrics plot to 'metrics_curves.png'")
plt.close()

# 混淆矩阵可视化
cm = confusion_matrix(y_clf_true, y_clf_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Test set)')
plt.colorbar()
classes = [str(i) for i in range(GROUP_NUM)]
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=200)
print("Saved confusion matrix to 'confusion_matrix.png'")
plt.close()


torch.save(model.state_dict(), 'final_weight.pt')
group_rules.to_csv('bmi_group_rules.csv', encoding='utf-8-sig', index=False)
pd.DataFrame({'mean': x_scaler.mean_, 'scale': x_scaler.scale_},
             index=['检测孕周', '生产次数', '年龄', '孕妇BMI', '身高']).to_csv('x_scaler.csv')
print('\n>>> 模型、规则、标准化系数、图像已落盘')
