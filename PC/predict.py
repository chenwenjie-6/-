"""识别 PCM 手势 — 用法: python predict.py <pcm文件路径>"""
import numpy as np
import onnxruntime as ort
import sys, os

# 加入当前目录以导入 process_pcm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process_pcm import _process_core

TARGET_TIME = 26
TARGET_ENV  = 100
NAMES = ['Push', 'Pull', 'Sweep', 'Slide', 'Fist_bump']


def pad_to(x, target):
    cur = len(x) if x.ndim == 1 else x.shape[-1]
    if cur == target:
        return x
    if cur < target:
        pad = target - cur
        if x.ndim == 1:
            return np.pad(x, (pad // 2, pad - pad // 2))
        return np.pad(x, ((0, 0), (pad // 2, pad - pad // 2)))
    start = (cur - target) // 2
    return x[start:start + target] if x.ndim == 1 else x[:, start:start + target]


def predict(pcm_path):
    # 1. 信号处理（返回 8 元组）
    data = _process_core(pcm_path)
    if data is None:
        print("信号处理失败，无有效信号")
        return None, None
    result, sig_amp, amp_smooth, t_axis, spec, fd, td, vmask = data
    env = amp_smooth.copy()

    # 3. 居中裁剪/填充
    spec = pad_to(spec, TARGET_TIME)
    env = pad_to(env, TARGET_ENV)

    # 4. minMaxNorm
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    env = (env - env.min()) / (env.max() - env.min() + 1e-8)

    # 5. ONNX 推理
    session = ort.InferenceSession(
        os.path.join(os.path.dirname(__file__), 'models', 'model.onnx'),
        providers=['CPUExecutionProvider']
    )
    inputs = {
        'spectrogram': spec.reshape(1, 1, *spec.shape).astype(np.float32),
        'envelope': env.reshape(1, 1, -1).astype(np.float32),
    }
    logits = session.run(None, inputs)[0][0]

    # 6. softmax
    logits = logits - logits.max()
    probs = np.exp(logits) / np.sum(np.exp(logits))
    pred = int(np.argmax(probs))

    print(f'\n预测: {NAMES[pred]}')
    print(f'logits: {logits}')
    for i, (name, p) in enumerate(zip(NAMES, probs)):
        bar = '#' * int(p * 50)
        marker = ' <--' if i == pred else ''
        print(f'  {name:7s}: {p:.4f} {bar}{marker}')
    print()

    return NAMES[pred], probs


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('用法: python predict.py <pcm文件路径>')
        sys.exit(1)
    predict(sys.argv[1])
