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
EPOCHS      = 300
LR          = 1e-3
WEIGHT_DECAY= 1e-4
PATIENCE    = 30
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GROUP_NUM   = 4
ALPHA_SUBJ  = 0.1
EMB_DIM     = 1


def load_data(path: str):
    # 读取并选取必要列
    use_cols = ['孕妇代码', '检测孕周', '生产次数', '年龄', '孕妇BMI', '身高', 'Y染色体浓度']
    df = pd.read_excel(path)
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}，请检查Excel列名是否正确")
    df = df[use_cols].dropna(subset=['孕妇代码','检测孕周','孕妇BMI'])

    # 统一标签 nipt_day
    df['nipt_day'] = df['检测孕周']

    # BMI 分组
    df['group_id'] = pd.qcut(df['孕妇BMI'], q=GROUP_NUM, labels=False, duplicates='drop')

    # 计算个体级汇总特征：测量次数、BMI 均值/std、nipt slope
    grp = df.groupby('孕妇代码')
    df['sub_count'] = grp['孕妇代码'].transform('count')
    df['sub_bmi_mean'] = grp['孕妇BMI'].transform('mean')
    df['sub_bmi_std']  = grp['孕妇BMI'].transform('std').fillna(0)
    df['sub_nipt_mean'] = grp['nipt_day'].transform('mean')
    df['sub_nipt_std']  = grp['nipt_day'].transform('std').fillna(0)

    # 计算每个个体的线性拟合斜率
    def slope_fn(g):
        if len(g) < 2:
            return 0.0
        x = g['检测孕周'].values
        y = g['nipt_day'].values
        return float(np.polyfit(x, y, 1)[0])
    slopes = grp.apply(slope_fn).to_dict()
    df['sub_nipt_slope'] = df['孕妇代码'].map(slopes).fillna(0.0)

    # 保存 BMI 边界
    edges = df.groupby('group_id')['孕妇BMI'].agg(['min', 'max']).reset_index()
    edges.to_csv('bmi_edges.csv', index=False)

    return df, edges

print('Loading data...')
df_all, bmi_edges = load_data(DATA_PATH)
print('Data shape:', df_all.shape)
print('BMI edges', bmi_edges)

# 特征与标签
feature_cols = ['检测孕周', '生产次数', '年龄', '孕妇BMI', '身高',
                'sub_count', 'sub_bmi_mean', 'sub_bmi_std', 'sub_nipt_mean', 'sub_nipt_std', 'sub_nipt_slope']
X = df_all[feature_cols].values
y_reg = df_all['nipt_day'].values.reshape(-1, 1)
y_clf = df_all['group_id'].astype(int).values


unique_subjects = df_all['孕妇代码'].unique()
subj2idx = {s:i for i,s in enumerate(unique_subjects)}
df_all['subj_idx'] = df_all['孕妇代码'].map(subj2idx)
subj_idx_all = df_all['subj_idx'].values

# 标准化
x_scaler = StandardScaler()
X = x_scaler.fit_transform(X)


X_train, X_temp, y_reg_tr, y_reg_te, y_clf_tr, y_clf_te, subj_tr_idx, subj_temp_idx = train_test_split(
    X, y_reg, y_clf, subj_idx_all, test_size=0.2, random_state=42, stratify=y_clf)
X_val, X_test, y_reg_val, y_reg_test, y_clf_val, y_clf_test, subj_val_idx, subj_test_idx = train_test_split(
    X_temp, y_reg_te, y_clf_te, subj_temp_idx, test_size=0.5, random_state=42, stratify=y_clf_te)


class NIPTDataset(Dataset):
    def __init__(self, x, y_reg, y_clf, subj_idx):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        self.y_clf = torch.tensor(y_clf, dtype=torch.long)
        self.subj_idx = torch.tensor(subj_idx, dtype=torch.long)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y_reg[idx], self.y_clf[idx], self.subj_idx[idx]

train_set = NIPTDataset(X_train, y_reg_tr, y_clf_tr, subj_tr_idx)
val_set   = NIPTDataset(X_val,   y_reg_val, y_clf_val, subj_val_idx)
test_set  = NIPTDataset(X_test,  y_reg_test, y_clf_test, subj_test_idx)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)


class MLPRegClfWithSub(nn.Module):
    def __init__(self, input_dim=11, hidden=64, n_groups=4, n_subjects=None, emb_dim=1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU()
        )
        self.tower_reg = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        self.tower_clf = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, n_groups))
        assert n_subjects is not None, 'n_subjects 需要指定'
        self.sub_emb = nn.Embedding(n_subjects, emb_dim)

    def forward(self, x, subj_idx):
        feat = self.shared(x)
        reg = self.tower_reg(feat)
        emb = self.sub_emb(subj_idx)
        if emb.shape[1] == 1:
            reg = reg + emb.view(-1,1)
        else:
            # 若 emb_dim > 1，则把 emb 连接到 feat 再通过一个小 MLP。目前保持 emb_dim=1
            reg = reg + emb[:, :1].view(-1,1)
        clf = self.tower_clf(feat)
        return reg, clf

model = MLPRegClfWithSub(input_dim=len(feature_cols), n_groups=GROUP_NUM,
                         n_subjects=len(unique_subjects), emb_dim=EMB_DIM).to(DEVICE)
print('--- Model architecture ---')
print(model)
print('Params:', sum(p.numel() for p in model.parameters()))

# 训练
reg_loss_fn = nn.MSELoss()
clf_loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

best_val, patience_cnt = 1e9, 0

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

    for xb, y_regb, y_clfb, subj_idx in train_loader:
        xb = xb.to(DEVICE); y_regb = y_regb.to(DEVICE); y_clfb = y_clfb.to(DEVICE); subj_idx = subj_idx.to(DEVICE)
        pred_reg, pred_clf = model(xb, subj_idx)

        loss_sample = reg_loss_fn(pred_reg, y_regb) + clf_loss_fn(pred_clf, y_clfb)


        unique_subj, inv_idx = torch.unique(subj_idx, return_inverse=True)
        s_loss = torch.tensor(0.0, device=DEVICE)
        if len(unique_subj) > 0:
            for u in range(len(unique_subj)):
                mask = (inv_idx == u)
                if mask.sum() < 1: continue
                pred_mean = pred_reg[mask].mean()
                y_mean = y_regb[mask].mean()
                s_loss = s_loss + (pred_mean - y_mean).pow(2)
            s_loss = s_loss / float(len(unique_subj))

        loss = loss_sample + ALPHA_SUBJ * s_loss

        optimizer.zero_grad(); loss.backward(); optimizer.step()
        epoch_tr_loss += loss.item() * xb.size(0)


        tr_true_cls.append(y_clfb.cpu())
        tr_pred_cls.append(torch.argmax(pred_clf.detach(), dim=1).cpu())
        tr_true_reg.append(y_regb.detach().cpu())
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

    # validation
    model.eval()
    epoch_val_loss = 0.
    val_true_cls = []
    val_pred_cls = []
    val_true_reg = []
    val_pred_reg = []
    with torch.no_grad():
        for xb, y_regb, y_clfb, subj_idx in val_loader:
            xb = xb.to(DEVICE); y_regb = y_regb.to(DEVICE); y_clfb = y_clfb.to(DEVICE); subj_idx = subj_idx.to(DEVICE)
            pred_reg, pred_clf = model(xb, subj_idx)
            epoch_val_loss += (reg_loss_fn(pred_reg, y_regb) + clf_loss_fn(pred_clf, y_clfb)).item() * xb.size(0)

            val_true_cls.append(y_clfb.cpu())
            val_pred_cls.append(torch.argmax(pred_clf.detach(), dim=1).cpu())
            val_true_reg.append(y_regb.detach().cpu())
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
    print(f'Epoch {epoch:03d} | train Loss: {avg_tr_loss:.4f} | val Loss: {avg_val_loss:.4f} '
          f'| train Acc: {train_acc:.4f} | val Acc: {val_acc:.4f} '
          f'| train MAE: {train_mae:.2f} 天 | val MAE: {val_mae:.2f} 天 | time: {epoch_time:.1f}s')

    if avg_val_loss < best_val:
        best_val = avg_val_loss
        torch.save(model.state_dict(), 'best_weight.pt')
        patience_cnt = 0
    else:
        patience_cnt += 1
    if patience_cnt >= PATIENCE:
        print(f'Early stop at epoch {epoch}'); break

# 测试
model.load_state_dict(torch.load('best_weight.pt', map_location=DEVICE))
model.eval()
y_reg_true, y_reg_pred = [], []
y_clf_true, y_clf_pred = [], []
with torch.no_grad():
    for xb, y_regb, y_clfb, subj_idx in test_loader:
        xb = xb.to(DEVICE); subj_idx = subj_idx.to(DEVICE)
        pred_reg, pred_clf = model(xb, subj_idx)
        y_reg_pred.append(pred_reg.detach().cpu())
        y_clf_pred.append(torch.argmax(pred_clf.detach(), dim=1).cpu())
        y_reg_true.append(y_regb)
        y_clf_true.append(y_clfb)

y_reg_pred = torch.cat(y_reg_pred).numpy().ravel()
y_reg_true = torch.cat(y_reg_true).numpy().ravel()
y_clf_pred = torch.cat(y_clf_pred).numpy().ravel()
y_clf_true = torch.cat(y_clf_true).numpy().ravel()

mae  = mean_absolute_error(y_reg_true, y_reg_pred)
rmse = math.sqrt(mean_squared_error(y_reg_true, y_reg_pred))
acc  = accuracy_score(y_clf_true, y_clf_pred)
print('>>> Test Regression  MAE = %.2f 天  RMSE = %.2f 天' % (mae, rmse))
print('>>> Test Classification Accuracy = %.3f' % acc)

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

# df_test 使用原始 df_all 的尾部样本（与原逻辑一致）
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

# 测试集上加入 1% 浮动的蒙特卡洛实验
def monte_carlo_noise(loader, model, noise_std=0.01, n_round=100):
    errs = []
    with torch.no_grad():
        for _ in range(n_round):
            y_tmp = []
            for xb, _, _, subj_idx in loader:
                xb_noisy = xb + torch.randn_like(xb) * noise_std
                pred_reg, _ = model(xb_noisy.to(DEVICE), subj_idx.to(DEVICE))
                y_tmp.append(pred_reg.detach().cpu())
            y_tmp = torch.cat(y_tmp).numpy().ravel()
            errs.append(mean_absolute_error(y_reg_true, y_tmp))
    return np.array(errs)

mc_err = monte_carlo_noise(test_loader, model, noise_std=0.01)
print('>>> 检测误差(±1 %%) 带来的 MAE 波动：%.2f ± %.2f 天' % (mc_err.mean(), mc_err.std()))

# 可视化
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

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
plt.plot(epochs_range, history['val_mae'], label='Val MAE', color=colors[5], linewidth=2)
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
             index=feature_cols).to_csv('x_scaler.csv')
print('>>> 模型、规则、标准化系数、图像已落盘')
