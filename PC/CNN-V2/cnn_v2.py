"""
cnn_v2.py -- 双流CNN + SE注意力 + 早停
用法: python cnn_v2.py
配置: 修改 GESTURES 列表控制分类数
"""
import numpy as np
import os, json, random, copy
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

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# 配置
# =========================================================
PROJECT_DIR = r"C:\Users\86132\Documents\手势毕设"
MODEL_DIR   = os.path.join(PROJECT_DIR, "CNN-V2", "models")
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
GESTURES    = ["Push", "Pull", "Sweep", "Slide", "Fist_bump"]
NUM_CLASSES = len(GESTURES)

SPEC_TIME  = 26
SPEC_FREQ  = 32
ENV_LEN    = 100

BATCH_SIZE    = 8
EPOCHS        = 120
LR            = 0.001
WEIGHT_DECAY  = 1e-4
N_FOLDS       = 5
PATIENCE      = 15
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# 数据加载
# =========================================================
def pad_to(x, target_len):
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
    X_spec, X_env, y, ids = [], [], [], []
    for label, gesture in enumerate(GESTURES):
        npy_dir = os.path.join(DATASET_DIR, gesture, "npy")
        if not os.path.isdir(npy_dir):
            print(f"  WARNING: {npy_dir} 不存在, 跳过 {gesture}")
            continue
        for f in sorted(os.listdir(npy_dir)):
            if f.endswith("_spectrogram.npy"):
                stem = f.replace("_spectrogram.npy", "")
                spec = np.load(os.path.join(npy_dir, f))
                env  = np.load(os.path.join(npy_dir, f"{stem}_envelope.npy"))
                X_spec.append(pad_to(spec, SPEC_TIME))
                X_env.append(pad_to(env, ENV_LEN))
                y.append(label)
                ids.append(f"{gesture}/{stem}")
    if not X_spec:
        raise RuntimeError("无数据, 请先运行 process_pcm.py 生成 npy")
    return np.array(X_spec), np.array(X_env), np.array(y), ids

# =========================================================
# 数据增强
# =========================================================
def augment(spec, env):
    if random.random() < 0.5:
        spec = spec + np.random.normal(0, 0.02, spec.shape).astype(np.float32)
    if random.random() < 0.5:
        env = env + np.random.normal(0, 0.02, env.shape).astype(np.float32)
    if random.random() < 0.5:
        shift = random.randint(-3, 3)
        spec = np.roll(spec, shift, axis=1)
        env  = np.roll(env, shift, axis=0)
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
        self.X_spec = X_spec
        self.X_env  = X_env
        self.y      = y
        self.ids    = ids
        self.train  = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        spec = self.X_spec[idx].copy()
        env  = self.X_env[idx].copy()
        if self.train:
            spec, env = augment(spec, env)
        return (torch.FloatTensor(spec).unsqueeze(0),
                torch.FloatTensor(env).unsqueeze(0),
                torch.LongTensor([self.y[idx]])[0],
                self.ids[idx])

# =========================================================
# SE 注意力模块
# =========================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.gap(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

# =========================================================
# 双流 CNN + SE
# =========================================================
class DualStreamCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 3)),
        )
        self.se = SEBlock(64, reduction=16)
        self.fc2d = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.5),
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(12),
        )
        self.fc1d = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.5),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, spec, env):
        f2d = self.conv2d(spec)
        f2d = self.se(f2d)
        f2d = self.fc2d(f2d)
        f1d = self.fc1d(self.conv1d(env))
        return self.classifier(torch.cat([f2d, f1d], dim=1))

# =========================================================
# 早停
# =========================================================
class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

# =========================================================
# 训练一个 fold
# =========================================================
def train_fold(fold, train_loader, val_loader, fold_dir):
    model = DualStreamCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    early_stop = EarlyStopping(patience=PATIENCE)

    best_state = None
    best_val_acc = 0
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
            best_state = copy.deepcopy(model.state_dict())

        improved = early_stop(val_loss / len(val_loader), epoch)
        if improved:
            torch.save(model.state_dict(), os.path.join(fold_dir, "fold_best.pth"))

        if (epoch + 1) % 20 == 0 or early_stop.should_stop:
            print(f"    Epoch {epoch+1:3d} | Val Loss={val_loss/len(val_loader):.4f}  Acc={val_acc:.4f}")

        if early_stop.should_stop:
            print(f"    早停于 epoch {epoch+1}, 最佳 epoch={early_stop.best_epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, best_val_acc, history

# =========================================================
# 主流程
# =========================================================
def main():
    print(f"设备: {DEVICE}")
    print(f"手势: {GESTURES} ({NUM_CLASSES} 分类)")
    print(f"SE注意力: 已启用")
    print(f"早停: patience={PATIENCE}")
    print()

    print("加载数据...")
    try:
        X_spec, X_env, y, ids = load_data()
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return

    counts = ", ".join([f"{g}={sum(y==i)}" for i, g in enumerate(GESTURES)])
    print(f"  总样本: {len(y)}  ({counts})")
    print(f"  谱图形状: {X_spec.shape}, 包络形状: {X_env.shape}")

    print("标准化...")
    for i in range(len(X_spec)):
        X_spec[i] = (X_spec[i] - X_spec[i].min()) / (X_spec[i].max() - X_spec[i].min() + 1e-8)
        X_env[i]  = (X_env[i] - X_env[i].min()) / (X_env[i].max() - X_env[i].min() + 1e-8)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_accs = []
    fold_histories = []
    all_val_preds  = []
    all_val_labels = []
    all_val_ids    = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_spec, y)):
        print(f"\n{'='*40}\nFold {fold+1}/{N_FOLDS}")
        print(f"  Train={len(train_idx)}, Val={len(val_idx)}")

        fold_dir = os.path.join(MODEL_DIR, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)

        train_ds = GestureDataset(X_spec[train_idx], X_env[train_idx], y[train_idx],
                                   [ids[i] for i in train_idx], train=True)
        val_ds   = GestureDataset(X_spec[val_idx], X_env[val_idx], y[val_idx],
                                   [ids[i] for i in val_idx], train=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model, best_acc, history = train_fold(fold, train_loader, val_loader, fold_dir)
        fold_accs.append(best_acc)
        fold_histories.append(history)

        torch.save(model.state_dict(), os.path.join(fold_dir, "model.pth"))

        model.eval()
        with torch.no_grad():
            for spec, env, labels, batch_ids in val_loader:
                spec, env = spec.to(DEVICE), env.to(DEVICE)
                outputs = model(spec, env)
                all_val_preds.extend(outputs.argmax(1).cpu().numpy())
                all_val_labels.extend(labels.numpy())
                all_val_ids.extend(batch_ids)

        print(f"  Fold {fold+1} Best Val Acc = {best_acc:.4f}")

    print(f"\n{'='*40}")
    print(f"{N_FOLDS}-Fold 平均准确率: {np.mean(fold_accs):.4f} (+-{np.std(fold_accs):.4f})")
    print(f"\n分类报告:")
    print(classification_report(all_val_labels, all_val_preds,
                                target_names=GESTURES, digits=4, zero_division=0))

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
    ax.set_title(f"混淆矩阵 (5-Fold CV, Acc={np.mean(fold_accs):.3f})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "confusion_matrix_v2.png"), dpi=100)
    plt.close(fig)

    fig2, axes = plt.subplots(N_FOLDS, 2, figsize=(14, 3*N_FOLDS))
    if N_FOLDS == 1:
        axes = axes.reshape(1, -1)
    for f, hist in enumerate(fold_histories):
        epochs = range(1, len(hist["train_loss"])+1)
        axes[f,0].plot(epochs, hist["train_loss"], label="Train Loss", color="steelblue")
        axes[f,0].plot(epochs, hist["val_loss"], label="Val Loss", color="coral")
        axes[f,0].set_title(f"Fold {f+1} Loss"); axes[f,0].grid(alpha=0.3); axes[f,0].legend()
        axes[f,1].plot(epochs, hist["val_acc"], label="Val Acc", color="seagreen")
        axes[f,1].axhline(y=np.mean(fold_accs), color="gray", ls="--", lw=0.8)
        axes[f,1].set_title(f"Fold {f+1} Accuracy"); axes[f,1].grid(alpha=0.3); axes[f,1].legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(MODEL_DIR, "training_curves_v2.png"), dpi=100)
    plt.close(fig2)

    print(f"\n{'='*40}")
    print("标签打乱验证...")
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
        m = DualStreamCNN().to(DEVICE)
        opt = optim.Adam(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
        for ep in range(80):
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
    random_level = 100/NUM_CLASSES
    mean_shuffle = np.mean(shuffle_accs)
    print(f"打乱标签 准确率: {mean_shuffle:.4f} (+-{np.std(shuffle_accs):.4f})")
    print(f"  -> {'真实特征' if mean_shuffle < random_level*1.5 else 'WARNING: 可能存在数据泄漏!'}")

    print("\n用全部数据训练最终模型...")
    full_ds = GestureDataset(X_spec, X_env, y, ids, train=True)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)

    final_model = DualStreamCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss={total_loss/len(full_loader):.4f}")

    model_path = os.path.join(MODEL_DIR, "cnn_v2.pth")
    torch.save(final_model.state_dict(), model_path)
    print(f"最终模型已保存: {model_path}")

    with open(os.path.join(MODEL_DIR, "cv_predictions_v2.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": i, "true": int(t), "pred": int(p)}
                   for i, t, p in zip(all_val_ids, all_val_labels, all_val_preds)],
                  f, ensure_ascii=False, indent=2)

    with open(os.path.join(MODEL_DIR, "config_v2.json"), "w", encoding="utf-8") as f:
        json.dump({
            "gestures": GESTURES,
            "num_classes": NUM_CLASSES,
            "se_attention": True,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "fold_accs": [float(a) for a in fold_accs],
            "mean_acc": float(np.mean(fold_accs)),
            "std_acc": float(np.std(fold_accs)),
            "shuffle_mean_acc": float(mean_shuffle),
        }, f, ensure_ascii=False, indent=2)

    print("全部完成")

if __name__ == "__main__":
    main()