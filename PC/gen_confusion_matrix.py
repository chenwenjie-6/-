"""混淆矩阵图生成脚本"""
import numpy as np, matplotlib, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 中文
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LABELS = ["前推", "后拉", "横扫", "滑动", "碰拳"]

def parse(s):
    return np.array([[int(x.strip()) for x in r.split(",")] for r in s.strip().split(";")])

def plot(cm, out, title, acc):
    n = 5
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(LABELS)
    ax.set_yticks(range(n)); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predict"); ax.set_ylabel("True")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=16,
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    t = f"{title} (Acc={acc})" if acc else title
    ax.set_title(t)
    fig.colorbar(ax.images[0], ax=ax)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)

if __name__ == "__main__":
    cm = parse(sys.argv[1])
    title = sys.argv[2] if len(sys.argv) > 2 else "Confusion Matrix"
    acc = sys.argv[3] if len(sys.argv) > 3 else None
    plot(cm, "confusion_matrix.png", title, acc)
