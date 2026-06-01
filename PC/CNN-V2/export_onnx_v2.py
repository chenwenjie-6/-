"""导出 ONNX 模型 (V2, 含 SE 注意力), 供 ONNX Runtime Android 加载。"""
import torch
import torch.nn as nn
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn_v2 import DualStreamCNN, SEBlock, GESTURES, NUM_CLASSES, SPEC_FREQ, SPEC_TIME, ENV_LEN, MODEL_DIR

os.makedirs(MODEL_DIR, exist_ok=True)


class DualStreamCNNExport(nn.Module):
    """与 cnn_v2.DualStreamCNN 相同, 但 AdaptiveAvgPool1d(12) → AvgPool1d(3,2)
    对 25 长度输入二者输出均为 12, 权重完全兼容。"""
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
            nn.AvgPool1d(kernel_size=3, stride=2),  # 25→12, 替代 AdaptiveAvgPool1d(12)
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


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 创建导出模型 + 加载训练权重
    export_model = DualStreamCNNExport()
    pth_path = os.path.join(MODEL_DIR, "cnn_v2.pth")
    if not os.path.exists(pth_path):
        print(f"ERROR: 未找到 {pth_path}, 请先运行 cnn_v2.py 训练")
        return

    state = torch.load(pth_path, map_location="cpu")
    export_model.load_state_dict(state)
    export_model.eval()

    # 加载原模型对比
    orig = DualStreamCNN()
    orig.load_state_dict(state)
    orig.eval()

    # 导出
    spec = torch.randn(1, 1, SPEC_FREQ, SPEC_TIME)
    env  = torch.randn(1, 1, ENV_LEN)

    out_path = os.path.join(MODEL_DIR, "model.onnx")
    torch.onnx.export(
        export_model, (spec, env), out_path,
        input_names=["spectrogram", "envelope"],
        output_names=["logits"],
        opset_version=17,
    )
    print(f"ONNX 已导出: {out_path}")
    print(f"分类: {GESTURES} ({NUM_CLASSES} classes)")

    # 验证
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path)
    for inp in sess.get_inputs():
        print(f"  输入: {inp.name} shape={inp.shape}")

    spec_np = spec.numpy().astype("float32")
    env_np  = env.numpy().astype("float32")
    ort_out = sess.run(None, {"spectrogram": spec_np, "envelope": env_np})[0]
    with torch.no_grad():
        torch_out = export_model(spec, env).numpy()

    diff_export = abs(ort_out - torch_out).max()
    print(f"ONNX vs 导出模型 最大差异: {diff_export:.2e} (应为 < 1e-5)")

    # 对比原模型 (检查 AvgPool1d 替代无影响)
    with torch.no_grad():
        orig_out = orig(spec, env).numpy()
    diff_orig = abs(orig_out - torch_out).max()
    print(f"原模型 vs 导出模型 最大差异: {diff_orig:.2e} (AvgPool1d 替代的影响, 应为 0)")

    # 复制到 assets 路径 (方便部署)
    assets_path = r"D:\AndroidProjects\UltraGesture\app\src\main\assets\model.onnx"
    import shutil
    shutil.copy2(out_path, assets_path)
    print(f"已复制到: {assets_path}")


if __name__ == "__main__":
    main()