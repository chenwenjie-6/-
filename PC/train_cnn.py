"""
多手势 CNN 训练脚本
双流架构: Conv2d(谱图) + Conv1d(包络) → 拼接 → 分类
"""
import numpy as np
import os, json, random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# 配置
# =========================================================
PROJECT_DIR = r"C:\Users\86132\Documents\手势毕设"
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
MODEL_DIR   = os.path.join(PROJECT_DIR, "models")
GESTURES    = ["Push", "Pull", "Sweep", "Slide", "Fist_bump"]
NUM_CLASSES = len(GESTURES)

SPEC_TIME  = 26   # 谱图时间维度统一
SPEC_FREQ  = 32   # 谱图频率维度
ENV_LEN    = 100  # 包络长度统一

BATCH_SIZE    = 8
EPOCHS        = 80
LR            = 0.001
N_FOLDS       = 5
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# 数据加载
# =========================================================
def pad_to(x, target_len):
    """居中填充/裁剪到目标长度"""
    cur = len(x) if x.ndim == 1 else x.shape[-1]
    if cur == target_len:
        return x
    if cur < target_len:
        pad_len = target_len - cur
        pad_left = pad_len // 2
        pad_right = pad_len - pad_left
        if x.ndim == 1:
            return np.pad(x, (pad_left, pad_right), mode="constant")
        else:
            return np.pad(x, ((0,0), (pad_left, pad_right)), mode="constant")
    else:
        start = (cur - target_len) // 2
        if x.ndim == 1:
            return x[start:start+target_len]
        else:
            return x[:, start:start+target_len]

def load_data():
    """加载所有样本，返回 X_spec, X_env, y, sample_ids"""
    X_spec, X_env, y, ids = [], [], [], []
    for label, gesture in enumerate(GESTURES):
        npy_dir = os.path.join(DATASET_DIR, gesture, "npy")
        for f in os.listdir(npy_dir):
            if f.endswith("_spectrogram.npy"):
                stem = f.replace("_spectrogram.npy", "")
                spec = np.load(os.path.join(npy_dir, f))
                env  = np.load(os.path.join(npy_dir, f"{stem}_envelope.npy"))
                X_spec.append(pad_to(spec, SPEC_TIME))
                X_env.append(pad_to(env, ENV_LEN))
                y.append(label)
                ids.append(f"{gesture}/{stem}")
    return np.array(X_spec), np.array(X_env), np.array(y), ids

# =========================================================
# 数据增强
# =========================================================
def augment(spec, env):
    """对单个样本做随机增强，返回增强后的 (spec, env)"""
    # 加性高斯噪声
    if random.random() < 0.5:
        spec = spec + np.random.normal(0, 0.02, spec.shape).astype(np.float32)
    if random.random() < 0.5:
        env = env + np.random.normal(0, 0.02, env.shape).astype(np.float32)

    # 时间方向随机平移（模拟动作起始时间不同）
    if random.random() < 0.5:
        shift = random.randint(-3, 3)
        spec = np.roll(spec, shift, axis=1)
        env  = np.roll(env, shift, axis=0)

    # 幅度缩放
    if random.random() < 0.3:
        scale = np.random.uniform(0.85, 1.15)
        spec = spec * scale
        env  = env * scale

    return spec, env

# =========================================================
# Dataset
# =========================================================
class GestureDataset(Dataset):
    def __init__(self, X_spec, X_env, y, ids, train=True):
        self.X_spec = torch.FloatTensor(X_spec).unsqueeze(1)  # (N, 1, 32, 26)
        self.X_env  = torch.FloatTensor(X_env).unsqueeze(1)   # (N, 1, 100)
        self.y      = torch.LongTensor(y)
        self.ids    = ids
        self.train  = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        spec = self.X_spec[idx].clone().numpy().squeeze(0)
        env  = self.X_env[idx].clone().numpy().squeeze(0)
        if self.train:
            spec, env = augment(spec, env)
        spec = torch.FloatTensor(spec).unsqueeze(0)
        env  = torch.FloatTensor(env).unsqueeze(0)
        return spec, env, self.y[idx], self.ids[idx]

# =========================================================
# 双流 CNN 模型
# =========================================================
class DualStreamCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        # 谱图分支: Conv2d (1, 32, 26)
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # (16, 16, 13)

            nn.Conv2d(16, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # (32, 8, 6)

            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 3)),       # (64, 4, 3)
        )
        self.fc2d = nn.Sequential(
            nn.Flatten(),                        # 64*4*3 = 768
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # 包络分支: Conv1d (1, 100)
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),                    # (16, 50)

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),                    # (32, 25)

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(12),           # (64, 12)
        )
        self.fc1d = nn.Sequential(
            nn.Flatten(),                        # 64*12 = 768
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # 融合分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, spec, env):
        f2d = self.fc2d(self.conv2d(spec))
        f1d = self.fc1d(self.conv1d(env))
        fused = torch.cat([f2d, f1d], dim=1)
        return self.classifier(fused)

# =========================================================
# 训练一个 fold
# =========================================================
def train_fold(fold, train_loader, val_loader):
    model = DualStreamCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for spec, env, labels, _ in train_loader:
            spec, env, labels = spec.to(DEVICE), env.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(spec, env), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for spec, env, labels, _ in val_loader:
                spec, env, labels = spec.to(DEVICE), env.to(DEVICE), labels.to(DEVICE)
                outputs = model(spec, env)
                val_loss += criterion(outputs, labels).item()
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d} | Val Acc={val_acc:.3f}")

    model.load_state_dict(best_state)
    return model, best_val_acc, history

# =========================================================
# 主流程
# =========================================================
def main():
    print(f"设备: {DEVICE}")
    print("加载数据...")
    X_spec, X_env, y, ids = load_data()
    counts = ", ".join([f"{g}={sum(y==i)}" for i, g in enumerate(GESTURES)])
    print(f"  总样本: {len(y)}  ({counts})")
    print(f"  谱图形状: {X_spec.shape}, 包络形状: {X_env.shape}")

    # 标准化（按样本做 min-max）
    for i in range(len(X_spec)):
        X_spec[i] = (X_spec[i] - X_spec[i].min()) / (X_spec[i].max() - X_spec[i].min() + 1e-8)
        X_env[i]  = (X_env[i] - X_env[i].min()) / (X_env[i].max() - X_env[i].min() + 1e-8)

    # K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_accs = []
    all_val_preds  = []
    all_val_labels = []
    all_val_ids    = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_spec, y)):
        print(f"\n{'='*40}\nFold {fold+1}/{N_FOLDS}")
        print(f"  Train={len(train_idx)}, Val={len(val_idx)}")

        train_ds = GestureDataset(X_spec[train_idx], X_env[train_idx], y[train_idx],
                                   [ids[i] for i in train_idx], train=True)
        val_ds   = GestureDataset(X_spec[val_idx], X_env[val_idx], y[val_idx],
                                   [ids[i] for i in val_idx], train=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model, best_acc, history = train_fold(fold, train_loader, val_loader)
        fold_accs.append(best_acc)

        # 收集验证集预测
        model.eval()
        with torch.no_grad():
            for spec, env, labels, batch_ids in val_loader:
                spec, env = spec.to(DEVICE), env.to(DEVICE)
                outputs = model(spec, env)
                all_val_preds.extend(outputs.argmax(1).cpu().numpy())
                all_val_labels.extend(labels.numpy())
                all_val_ids.extend(batch_ids)

        print(f"  Fold {fold+1} Best Val Acc = {best_acc:.4f}")

    # 汇总
    print(f"\n{'='*40}")
    print(f"{N_FOLDS}-Fold 平均准确率: {np.mean(fold_accs):.4f} (±{np.std(fold_accs):.4f})")
    print(f"\n分类报告:")
    print(classification_report(all_val_labels, all_val_preds,
                                target_names=GESTURES, digits=4, zero_division=0))

    # 混淆矩阵
    cm = confusion_matrix(all_val_labels, all_val_preds)
    nc = NUM_CLASSES
    fig, ax = plt.subplots(figsize=(nc+2, nc+1))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(nc)); ax.set_xticklabels(GESTURES)
    ax.set_yticks(range(nc)); ax.set_yticklabels(GESTURES)
    ax.set_xlabel("预测"); ax.set_ylabel("真实")
    for i in range(nc):
        for j in range(nc):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=16,
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_title(f"混淆矩阵 ({N_FOLDS}-Fold CV, Acc={np.mean(fold_accs):.3f})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=100)
    plt.close(fig)

    # ========== 标签打乱验证 ==========
    print(f"\n{'='*40}")
    print("标签打乱验证（若仍100%则有数据泄漏bug）...")
    y_shuffled = y.copy()
    np.random.seed(7)
    np.random.shuffle(y_shuffled)
    shuffle_accs = []
    skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=99)
    for fold, (tr, va) in enumerate(skf2.split(X_spec, y_shuffled)):
        tr_ds = GestureDataset(X_spec[tr], X_env[tr], y_shuffled[tr],
                                [ids[i] for i in tr], train=True)
        va_ds = GestureDataset(X_spec[va], X_env[va], y_shuffled[va],
                                [ids[i] for i in va], train=False)
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
        va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)
        m = DualStreamCNN(num_classes=NUM_CLASSES).to(DEVICE)
        opt = optim.Adam(m.parameters(), lr=LR, weight_decay=1e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        for ep in range(EPOCHS):
            m.train()
            for sp, en, lb, _ in tr_ld:
                sp, en, lb = sp.to(DEVICE), en.to(DEVICE), lb.to(DEVICE)
                opt.zero_grad()
                loss = nn.CrossEntropyLoss()(m(sp, en), lb)
                loss.backward(); opt.step()
            sch.step()
        m.eval()
        preds, labs = [], []
        with torch.no_grad():
            for sp, en, lb, _ in va_ld:
                sp, en = sp.to(DEVICE), en.to(DEVICE)
                preds.extend(m(sp, en).argmax(1).cpu().numpy())
                labs.extend(lb.numpy())
        shuffle_accs.append(accuracy_score(labs, preds))
    print(f"打乱标签 5-Fold 平均准确率: {np.mean(shuffle_accs):.4f} (±{np.std(shuffle_accs):.4f})")
    print(f"  → {'真实特征' if np.mean(shuffle_accs) < 0.5 else '存在数据泄漏!'}")

    # 保存模型
    # 用全部数据训练最终模型
    print("\n用全部数据训练最终模型...")
    full_ds = GestureDataset(X_spec, X_env, y, ids, train=True)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)

    final_model = DualStreamCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        final_model.train()
        total_loss = 0
        for spec, env, labels, _ in full_loader:
            spec, env, labels = spec.to(DEVICE), env.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(final_model(spec, env), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

    model_name = "vs_".join(GESTURES) + ".pth"
    torch.save(final_model.state_dict(), os.path.join(MODEL_DIR, model_name))
    print(f"最终模型已保存: {os.path.join(MODEL_DIR, model_name)}")

    # 保存预测详情
    with open(os.path.join(MODEL_DIR, "cv_predictions.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": i, "true": int(t), "pred": int(p)}
                   for i, t, p in zip(all_val_ids, all_val_labels, all_val_preds)],
                  f, ensure_ascii=False, indent=2)

    print("预测详情已保存")

if __name__ == "__main__":
    main()
