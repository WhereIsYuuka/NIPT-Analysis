import matplotlib
matplotlib.use('Agg')
import os, sys, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    brier_score_loss, accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")  # 屏蔽警告


# 配置
DATA_PATH = r"C:\Users\admin\Desktop\国赛论文\修改后附件女.xlsx"  # 数据文件路径
SHEET = 0                              # Excel 工作表索引或名称
RANDOM_SEED = 42                      # 随机种子，保证可复现
BATCH_SIZE = 64                       # 训练批大小
EPOCHS = 120                          # 最大训练轮数
LR = 1e-3                             # 学习率
WEIGHT_DECAY = 1e-5                   # L2 正则
EARLY_STOPPING_PATIENCE = 12          # 早停耐心
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Z_THRESHOLD = 3.0                     # Z 值阈值（用于构造标签的兜底策略）
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05]  # 特征乘性噪声幅度列表（敏感性分析）
MC_RUNS = 50                          # 蒙特卡洛重复次数
MODEL_SAVE = "mlp_params.pth"         # 最佳模型参数保存文件
META_SAVE = "meta.joblib"             # 元信息保存文件（缩放器/特征等）


df = pd.read_excel(DATA_PATH, sheet_name=SHEET, engine="openpyxl")  # 读取 Excel
df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]  # 去除列名空格

# 选取可能存在的数值特征列
numeric_cols = [c for c in ["年龄","身高","体重","检测孕周","孕妇BMI",
    "原始读段数","在参考基因组上比对的比例","重复读段的比例","唯一比对的读段数",
    "GC含量","13号染色体的Z值","18号染色体的Z值","21号染色体的Z值","X染色体的Z值",
    "X染色体浓度","13号染色体的GC含量","18号染色体的GC含量","21号染色体的GC含量",
    "被过滤掉读段数的比例"] if c in df.columns]

patient_col = "孕妇代码" if "孕妇代码" in df.columns else None
label_col = "胎儿是否健康" if "胎儿是否健康" in df.columns else None
non_disomy_col = "染色体的非整倍体" if "染色体的非整倍体" in df.columns else None

# 标签构造策略
if label_col and df[label_col].nunique()>1:
    y = (df[label_col]==0).astype(int).values
elif non_disomy_col:
    y = df[non_disomy_col].notnull().astype(int).values  # 存在异常记录视为 1
else:
    # 兜底：利用 13/18/21 号染色体 Z 值绝对值是否 >= 阈值 判定
    zc = [c for c in ["13号染色体的Z值","18号染色体的Z值","21号染色体的Z值"] if c in df.columns]
    if len(zc)==0:
        raise RuntimeError("无法构造标签：请确保 Z 值列或非整倍体列存在")
    y = (df[zc].abs().max(axis=1) >= Z_THRESHOLD).astype(int).values

df['_label'] = y
print("label counts:", np.bincount(y))

# 特征矩阵与分组信息
if patient_col is None:
    df['_pid'] = np.arange(len(df))
    patient_col = '_pid'
X = df[numeric_cols].copy()
groups = df[patient_col].values

# 缺失值填补（中位数）+ 标准化
num_imp = SimpleImputer(strategy="median")
X_num = pd.DataFrame(num_imp.fit_transform(X), columns=numeric_cols)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=numeric_cols)

# 简单衍生特征：唯一比对读段 / 原始读段
X_scaled['unique_ratio'] = (X_num.get("唯一比对的读段数",0) / (X_num.get("原始读段数",1))).fillna(0)

feature_cols = X_scaled.columns.tolist()
X_arr = X_scaled.values.astype(np.float32)
y_arr = y.astype(np.float32)

# 保存元信息
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(X_arr, y_arr, groups))
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.111111, random_state=RANDOM_SEED)
train_idx, val_idx = next(gss2.split(X_arr[train_idx], y_arr[train_idx], groups[train_idx]))
#val_idx = train_idx[val_idx]  # correct indices

X_train, y_train = X_arr[train_idx], y_arr[train_idx]
X_val, y_val = X_arr[val_idx], y_arr[val_idx]
X_test, y_test = X_arr[test_idx], y_arr[test_idx]

print("Sizes:", X_train.shape, X_val.shape, X_test.shape)
print("Train pos:", y_train.sum(), "Val pos:", y_val.sum(), "Test pos:", y_test.sum())

# 多层感知机
class MLP(nn.Module):
    def __init__(self, input_dim, hidden=[128,64], dropout=0.3):
        super().__init__()
        layers=[]
        prev=input_dim
        for h in hidden:
            layers.append(nn.Linear(prev,h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev=h
        layers.append(nn.Linear(prev,1))
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)

model = MLP(input_dim=X_train.shape[1]).to(DEVICE)
# 计算正样本权重（缓解类别不平衡）
pos = y_train.sum(); neg = len(y_train)-pos
pos_weight = torch.tensor(max(1.0, neg/(pos+1e-8))).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# 数据加载器封装
class TabDataset(Dataset):
    def __init__(self,X,y):
        self.X=torch.tensor(X)
        self.y=torch.tensor(y).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self,idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(TabDataset(X_train,y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TabDataset(X_val,y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TabDataset(X_test,y_test), batch_size=BATCH_SIZE, shuffle=False)

# 训练与验证
train_losses=[]; val_aucs=[]
best_val_auc=-1; patience=0
for epoch in range(1,EPOCHS+1):
    model.train()
    total_loss=0.0
    for xb,yb in train_loader:
        xb=xb.to(DEVICE).float(); yb=yb.to(DEVICE).float()
        opt.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward(); opt.step()
        total_loss += loss.item()*xb.size(0)
    avg_loss = total_loss/len(train_loader.dataset)
    train_losses.append(avg_loss)
    # val
    model.eval()
    ys=[]; preds=[]
    with torch.no_grad():
        for xb,yb in val_loader:
            xb=xb.to(DEVICE).float()
            logits=model(xb)
            probs=torch.sigmoid(logits).cpu().numpy().ravel()
            preds.extend(probs.tolist()); ys.extend(yb.numpy().ravel().tolist())
    try:
        val_auc = roc_auc_score(ys,preds)
    except:
        val_auc=0.5
    val_aucs.append(val_auc)
    if val_auc>best_val_auc+1e-4:
        best_val_auc=val_auc; patience=0
        torch.save(model.state_dict(), MODEL_SAVE)
    else:
        patience+=1
    print(f"Epoch {epoch} loss {avg_loss:.4f} val_auc {val_auc:.4f} best {best_val_auc:.4f} pat {patience}")
    if patience>=EARLY_STOPPING_PATIENCE:
        print("Early stop."); break

# 加载最佳模型参数
model.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE))
model.eval()

# 辅助：批量预测函数
def predict_array(X_array):
    ds = DataLoader(TabDataset(X_array, np.zeros(len(X_array))), batch_size=256, shuffle=False)
    preds=[]
    with torch.no_grad():
        for xb,_ in ds:
            xb=xb.to(DEVICE).float()
            logits=model(xb)
            probs=torch.sigmoid(logits).cpu().numpy().ravel()
            preds.extend(probs.tolist())
    return np.array(preds)

y_test_pred = predict_array(X_test)
y_val_pred = predict_array(X_val)
y_train_pred = predict_array(X_train)

# 通过验证集 ROC 曲线使用 Youden 指数选择最佳阈值
fpr,tpr,thr = roc_curve(y_val, y_val_pred)
youden_idx = np.argmax(tpr-fpr)
th_best = thr[youden_idx]
print("Youden thr:", th_best)

# 结果可视化与评估
os.makedirs("figs", exist_ok=True)
# 1) 训练损失曲线
plt.figure()
plt.plot(np.arange(1,len(train_losses)+1), train_losses)
plt.xlabel("Epoch"); plt.ylabel("Train loss"); plt.title("Training loss curve")
plt.savefig("figs/train_loss.png"); plt.close()

# 2) 验证集 AUC 曲线
plt.figure()
plt.plot(np.arange(1,len(val_aucs)+1), val_aucs)
plt.xlabel("Epoch"); plt.ylabel("Validation AUC"); plt.title("Validation AUC curve")
plt.savefig("figs/val_auc.png"); plt.close()

# 3) 测试集 ROC 曲线
from sklearn.metrics import roc_curve, auc
fpr_t,tpr_t,_ = roc_curve(y_test, y_test_pred)
auc_t = auc(fpr_t,tpr_t)
plt.figure()
plt.plot(fpr_t,tpr_t, label=f"AUC={auc_t:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC - Test"); plt.legend()
plt.savefig("figs/roc_test.png"); plt.close()

# 4) 测试集 PR 曲线
prec, rec, prt = precision_recall_curve(y_test, y_test_pred)
# emulate a single plot per rule: precision vs recall
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall - Test")
plt.savefig("figs/pr_test.png"); plt.close()

# 5) 校准曲线
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, y_test_pred, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1],"--")
plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency"); plt.title("Calibration plot (Test)")
plt.savefig("figs/calibration_test.png"); plt.close()
brier = brier_score_loss(y_test, y_test_pred)
print("Brier score:", brier)

# 6) 按真实类别的预测概率分布
plt.figure()
plt.hist(y_test_pred[y_test==0], bins=30, alpha=0.7)
plt.hist(y_test_pred[y_test==1], bins=30, alpha=0.7)
plt.xlabel("Predicted prob"); plt.title("Predicted probability dist by true class")
plt.savefig("figs/pred_dist.png"); plt.close()

# 7) 模型参数直方图与统计摘要
weights=[]
for name,param in model.named_parameters():
    if param.requires_grad:
        arr = param.detach().cpu().numpy().ravel()
        weights.append(arr)
        # save histogram per param
        plt.figure()
        plt.hist(arr, bins=50)
        plt.title(f"Weight distribution: {name}")
        plt.xlabel("value"); plt.ylabel("count")
        safe = name.replace(".","_")
        plt.savefig(f"figs/weight_hist_{safe}.png"); plt.close()
# summary stats
param_stats = {}
for name,param in model.named_parameters():
    arr = param.detach().cpu().numpy()
    param_stats[name] = {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max()), "shape": arr.shape}
import json
with open("figs/param_stats.json","w",encoding="utf8") as f:
    json.dump(param_stats, f, ensure_ascii=False, indent=2)

# 8) 置换重要度评估特征对 AUC 的影响
def perm_importance(X_base, y_base, feature_names, metric_fn, n_repeats=30):
    base_pred = predict_array(X_base)
    base_score = metric_fn(y_base, base_pred)
    importances = []
    rng = np.random.RandomState(RANDOM_SEED)
    for j,fn in enumerate(feature_names):
        scores=[]
        for r in range(n_repeats):
            Xp = X_base.copy()
            idx = rng.permutation(len(Xp))
            Xp[:,j] = X_base[idx,j]
            pred = predict_array(Xp)
            sc = metric_fn(y_base, pred)
            scores.append(base_score - sc)
        importances.append(np.mean(scores))
    return np.array(importances)

def auc_metric(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except:
        return 0.5

perm_imp = perm_importance(X_val, y_val, feature_cols, auc_metric, n_repeats=30)
# plot
plt.figure()
xpos = np.arange(len(feature_cols))
plt.bar(xpos, perm_imp)
plt.xticks(xpos, feature_cols, rotation=90)
plt.title("Permutation importance (AUC drop)")
plt.tight_layout()
plt.savefig("figs/perm_importance.png"); plt.close()
# write csv
imp_df = pd.DataFrame({"feature":feature_cols, "perm_drop":perm_imp})
imp_df.to_csv("figs/perm_importance.csv", index=False)

# 9) 蒙特卡洛噪声敏感性：模拟特征乘性噪声考察鲁棒性
def mc_noise_eval(X_base, y_base, noise_level, runs=MC_RUNS):
    rng = np.random.RandomState(RANDOM_SEED)
    res=[]
    for r in range(runs):
        noise = 1.0 + rng.normal(0, noise_level, size=X_base.shape)
        Xn = X_base * noise
        pred = predict_array(Xn)
        try:
            aucv = roc_auc_score(y_base, pred)
        except:
            aucv = np.nan
        # compute metrics at threshold th_best
        y_pred_label = (pred >= th_best).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_base, y_pred_label, average='binary', zero_division=0)
        res.append((aucv, prec, rec, f1))
    res = np.array(res)
    return {"auc_mean":np.nanmean(res[:,0]), "auc_std":np.nanstd(res[:,0]),
            "prec_mean":np.nanmean(res[:,1]), "rec_mean":np.nanmean(res[:,2]), "f1_mean":np.nanmean(res[:,3])}

mc_results = {}
for nl in NOISE_LEVELS:
    mc_results[nl] = mc_noise_eval(X_test, y_test, nl, runs=MC_RUNS)
import json
with open("figs/mc_results.json","w",encoding="utf8") as f:
    json.dump(mc_results, f, ensure_ascii=False, indent=2)

# 保存元数据
joblib.dump({"scaler":scaler, "feature_cols":feature_cols, "param_stats":param_stats, "th_best":th_best}, META_SAVE)
print("Saved meta to", META_SAVE)
print("All figures saved to ./figs/")


# 模型结构导出
import json, os, math
import torch
import torch.nn as nn
from collections import OrderedDict

def get_activation_name(module):
    """返回激活层名称（若 module 为常见激活函数实例）。"""
    act_map = {
        nn.ReLU: "ReLU",
        nn.LeakyReLU: "LeakyReLU",
        nn.Sigmoid: "Sigmoid",
        nn.Tanh: "Tanh",
        nn.ELU: "ELU",
        nn.Softmax: "Softmax",
        nn.LogSoftmax: "LogSoftmax",
        nn.GELU: "GELU"
    }
    for k, v in act_map.items():
        if isinstance(module, k):
            return v
    return None

def save_model_structure(model: nn.Module, input_dim:int, save_prefix="model_structure"):
    """检查 PyTorch 模型结构并导出 JSON + Markdown 摘要。

    参数:
        model: 已训练或初始化的 nn.Module
        input_dim: 输入特征维度（用于构造假输入进行前向推理获取形状）
        save_prefix: 输出文件前缀（不含后缀）
    """
    # 收集层信息
    layer_list = []
    total_params = 0
    total_trainable = 0

    # 获取各层输入/输出形状
    hooks = []
    activation_names = {}

    module_input_shapes = {}
    module_output_shapes = {}

    def make_hook(name):
        def hook(module, inp, out):
            try:
                module_input_shapes[name] = [tuple(x.size()) for x in inp if hasattr(x, 'size')]
            except Exception:
                module_input_shapes[name] = None
            try:
                module_output_shapes[name] = [tuple(out.size())] if hasattr(out, 'size') else None
            except Exception:
                module_output_shapes[name] = None
        return hook

    # 为所有非顶层模块注册 hook
    for name, mod in model.named_modules():

        if name == "":
            continue
        try:
            hooks.append((name, mod.register_forward_hook(make_hook(name))))
        except Exception:
            pass

    # 使用 batch=2 的零张量做一次前向传播
    model_eval = model.eval()
    with torch.no_grad():
        try:
            dummy = torch.zeros((2, input_dim)).float()
            _ = model_eval(dummy.to(next(model.parameters()).device) if any(p.requires_grad for p in model.parameters()) else dummy)
        except Exception:
            pass

    # 移除 hook
    for _, h in hooks:
        h.remove()

    # 按拓扑顺序整理成层级列表
    idx = 0
    for name, module in model.named_modules():
        if name == "":
            continue
        if len(list(module.children())) > 0:
            continue
        idx += 1
        param_count = 0
        trainable_count = 0
        param_shapes = []
        for pname, p in module.named_parameters(recurse=False):
            s = tuple(p.size())
            param_shapes.append({ "name": pname, "shape": s, "numel": p.numel(), "requires_grad": p.requires_grad })
            param_count += p.numel()
            if p.requires_grad:
                trainable_count += p.numel()
        total_params += param_count
        total_trainable += trainable_count

        act_name = get_activation_name(module)
        mod_type = module.__class__.__name__
        in_shapes = module_input_shapes.get(name, None)
        out_shapes = module_output_shapes.get(name, None)
        layer_info = {
            "index": idx,
            "name": name,
            "type": mod_type,
            "activation": act_name,
            "param_count": param_count,
            "trainable_param_count": trainable_count,
            "param_shapes": param_shapes,
            "input_shapes": in_shapes,
            "output_shapes": out_shapes
        }
        layer_list.append(layer_info)

    # 保存 JSON
    out = {
        "model_class": model.__class__.__name__,
        "total_params": total_params,
        "total_trainable_params": total_trainable,
        "layers": layer_list
    }
    os.makedirs(os.path.dirname(save_prefix) if os.path.dirname(save_prefix) else ".", exist_ok=True)
    json_path = f"{save_prefix}.json"
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 生成 Markdown 表格
    md_lines = []
    md_lines.append(f"# Model structure: {model.__class__.__name__}\n")
    md_lines.append(f"- Total params: **{total_params:,}**\n")
    md_lines.append(f"- Total trainable params: **{total_trainable:,}**\n")
    md_lines.append("\n| # | Layer name | Type | Activation | Params | Trainable | Input shape -> Output shape |\n")
    md_lines.append("|---:|---|---|---|---:|---:|---|\n")
    for L in layer_list:
        inp = L['input_shapes'][0] if L['input_shapes'] else "unknown"
        out = L['output_shapes'][0] if L['output_shapes'] else "unknown"
        md_lines.append(f"| {L['index']} | {L['name']} | {L['type']} | {L['activation'] or '-'} | {L['param_count']:,} | {L['trainable_param_count']:,} | `{inp}` → `{out}` |\n")
    md_text = "\n".join(md_lines)
    md_path = f"{save_prefix}.md"
    with open(md_path, "w", encoding="utf8") as f:
        f.write(md_text)

    print(f"Model structure saved to {json_path} and {md_path}")
    return out

# model = MLP(...)
# info = save_model_structure(model, input_dim=X_train.shape[1])
info = save_model_structure(model, input_dim=X_train.shape[1], save_prefix="model_structure")
