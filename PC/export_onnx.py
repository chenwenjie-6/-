"""导出 ONNX 模型，供 ONNX Runtime Android 加载。"""
import torch
import torch.nn as nn
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from train_cnn import NUM_CLASSES, GESTURES, SPEC_FREQ, SPEC_TIME, ENV_LEN

MODEL_DIR = "models"


class DualStreamCNNExport(nn.Module):
    """与原版 DualStreamCNN 相同，但 AdaptiveAvgPool1d(12) 替换为 AvgPool1d(3,2)
    对 25 长度输入二者输出均为 12，权重完全兼容。"""
    def __init__(self, num_classes=3):
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
        self.fc2d = nn.Sequential(
            nn.Flatten(), nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.5),
        )
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            # 25 → 12: AvgPool1d(kernel=3, stride=2) = floor((25+0-3)/2)+1 = 12
            nn.AvgPool1d(kernel_size=3, stride=2),
        )
        self.fc1d = nn.Sequential(
            nn.Flatten(), nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, spec, env):
        f2d = self.fc2d(self.conv2d(spec))
        f1d = self.fc1d(self.conv1d(env))
        return self.classifier(torch.cat([f2d, f1d], dim=1))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 创建模型 + 加载训练好的权重
    model = DualStreamCNNExport(num_classes=NUM_CLASSES)
    model_name = "vs_".join(GESTURES) + ".pth"
    state = torch.load(os.path.join(MODEL_DIR, model_name),
                       map_location="cpu")
    # AvgPool1d 无参数，权重 struct 一样，直接加载
    model.load_state_dict(state)
    model.eval()

    spec = torch.randn(1, 1, SPEC_FREQ, SPEC_TIME)
    env  = torch.randn(1, 1, ENV_LEN)

    out_path = os.path.join(MODEL_DIR, "model.onnx")
    torch.onnx.export(
        model, (spec, env), out_path,
        input_names=["spectrogram", "envelope"],
        output_names=["logits"],
        opset_version=17,
    )
    print(f"ONNX 已导出: {out_path}")
    print(f"分类: {GESTURES}")

    # 验证一致性
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path)
    for inp in sess.get_inputs():
        print(f"  输入: {inp.name} shape={inp.shape}")

    spec_np = spec.numpy().astype("float32")
    env_np  = env.numpy().astype("float32")
    ort_out = sess.run(None, {"spectrogram": spec_np, "envelope": env_np})[0]
    with torch.no_grad():
        torch_out = model(spec, env).numpy()
    diff = abs(ort_out - torch_out).max()
    print(f"PyTorch  vs ONNX 最大差异: {diff:.2e} (应为 < 1e-5)")

    # ----- 对比原模型 -----
    from train_cnn import DualStreamCNN
    orig = DualStreamCNN(num_classes=NUM_CLASSES)
    orig.load_state_dict(state)
    orig.eval()
    with torch.no_grad():
        orig_out = orig(spec, env).numpy()
    orig_diff = abs(orig_out - torch_out).max()
    print(f"原模型 vs 导出模型 最大差异: {orig_diff:.2e} (AvgPool1d替代AdaptiveAvgPool1d的影响)")


if __name__ == "__main__":
    main()
