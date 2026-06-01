"""
导出训练好的 PyTorch 模型为 TorchScript 格式，供 Android LibTorch 加载。
用法: python export_model.py
"""
import torch
import os, sys

# 复用 train_cnn.py 中的模型定义
sys.path.insert(0, os.path.dirname(__file__))
from train_cnn import DualStreamCNN, GESTURES, NUM_CLASSES, SPEC_FREQ, SPEC_TIME, ENV_LEN

MODEL_DIR = "models"
MODEL_FILE = "Pushvs_Pullvs_Sweep.pth"

def main():
    # 切换工作目录以确保中文路径不出问题
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 1. 创建模型结构并加载训练好的权重
    model = DualStreamCNN(num_classes=NUM_CLASSES)
    state = torch.load(os.path.join(MODEL_DIR, MODEL_FILE), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # 2. 用 torch.jit.script 导出（比 trace 好，保留控制流）
    scripted = torch.jit.script(model)
    out_path = os.path.join(MODEL_DIR, "model.pt")
    scripted.save(out_path)
    print(f"模型已导出: {out_path}")

    # 3. 验证：导出的模型和原模型输出一致
    spec = torch.randn(1, 1, SPEC_FREQ, SPEC_TIME)
    env  = torch.randn(1, 1, ENV_LEN)
    with torch.no_grad():
        out_orig = model(spec, env)
        out_pt   = scripted(spec, env)
    diff = (out_orig - out_pt).abs().max().item()
    print(f"输入 shape: spec={list(spec.shape)}, env={list(env.shape)}")
    print(f"输出 shape: {list(out_pt.shape)}")
    print(f"分类手势: {GESTURES}")
    print(f"原模型与导出模型最大差异: {diff:.2e} (应为0)")

    # 4. 保存输入输出规格，C++ 端需要
    print(f"\nC++ 端参考尺寸:")
    print(f"  谱图输入: (1, {SPEC_FREQ}, {SPEC_TIME})  float32")
    print(f"  包络输入: (1, {ENV_LEN})               float32")
    print(f"  输出:     ({NUM_CLASSES},)               float32 (logits)")

if __name__ == "__main__":
    main()
