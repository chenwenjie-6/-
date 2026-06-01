# =========================================================
# UltraGesture — V4 数据集模式
# 用法: python process_pcm.py                     # 默认:拉手机数据+处理所有手势
#       python process_pcm.py --gesture Push       # 只处理一个手势
#       python process_pcm.py --no-pull            # 跳过 adb pull
#       python process_pcm.py --file xxx.pcm       # 单文件(旧模式)
# V3: 无重叠提取, 慢时间 50Hz
# =========================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import savgol_filter
import scipy.ndimage as ndimage
import os, sys, time, argparse, json, subprocess, re

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# 路径配置
# =========================================================
PROJECT_DIR    = r"C:\Users\86132\Documents\手势毕设"
DATASET_DIR    = os.path.join(PROJECT_DIR, "dataset")
ADB_PATH       = r"C:\Users\86132\AppData\Local\Android\Sdk\platform-tools\adb.exe"
PHONE_PCM_DIR  = "/storage/emulated/0/Android/data/com.cwj.ultragesture/files/pcm_data"

# 旧模式默认路径
DEFAULT_DATA_DIR   = r"D:\UltraGesture\pcm_data"
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

# 7 种手势
GESTURES = ["Push", "Pull", "Sweep", "Slide", "Fist_bump", "Grab", "Tap"]

# =========================================================
# 雷达参数
# =========================================================
FS = 48000
START_FREQ = 17000.0
END_FREQ   = 23000.0
BANDWIDTH  = END_FREQ - START_FREQ
CHIRP_TIME = 0.02
CHIRP_SAMPLES = int(FS * CHIRP_TIME)   # 960
SOUND_SPEED   = 343
RANGE_RES     = SOUND_SPEED / (2 * BANDWIDTH)
CHIRP_SLOPE   = BANDWIDTH / CHIRP_TIME

CHIRP_STRIDE = CHIRP_SAMPLES
FS_SLOW      = FS / CHIRP_STRIDE  # 50 Hz

MAX_RANGE_CM  = 100
DOPPLER_LIMIT = 80

# 全局滤波器
from scipy.signal import sosfiltfilt
b_bp, a_bp = signal.butter(6, [START_FREQ/(FS/2), END_FREQ/(FS/2)], btype="band")
b_lp, a_lp = signal.butter(6, 2500/(FS/2), btype="low")
sos_lp = signal.tf2sos(b_lp, a_lp)

# =========================================================
# 参考信号
# =========================================================
def make_ref_chirp_real():
    t = np.arange(CHIRP_SAMPLES) / FS
    phase = 2*np.pi * (START_FREQ*t + 0.5*CHIRP_SLOPE*t*t)
    win = np.hanning(CHIRP_SAMPLES)
    return 0.9 * np.sin(phase) * win

def make_ref_chirp_complex():
    t = np.arange(CHIRP_SAMPLES) / FS
    phase = 2*np.pi * (START_FREQ*t + 0.5*CHIRP_SLOPE*t*t)
    win = np.hanning(CHIRP_SAMPLES)
    return 0.9 * np.exp(1j*phase) * win

ref_real = make_ref_chirp_real()
ref_cplx = make_ref_chirp_complex()

# =========================================================
# 信号处理
# =========================================================
def find_global_offset(rx, search_len=None):
    if search_len is None:
        search_len = CHIRP_SAMPLES * 5
    search_len = min(search_len, len(rx) - CHIRP_SAMPLES)
    corr = np.correlate(rx[:search_len], ref_real, mode="valid")
    return int(np.argmax(np.abs(corr)))

def freq_to_range(f):
    return f * SOUND_SPEED / (2 * CHIRP_SLOPE)

def detect_gesture(fname):
    """从文件名检测手势: Push_001.pcm -> Push"""
    m = re.match(r"([A-Z][a-z_]*)", fname)
    return m.group(1) if m else None

# =========================================================
# adb pull（同名文件跳过，只拉新文件）
# =========================================================
def adb_pull(gesture, pcm_dir):
    """只拉本地没有的 PCM 文件，同名跳过。返回新增文件名列表"""
    os.makedirs(pcm_dir, exist_ok=True)

    # 获取手机端文件列表
    phone_dir = f"{PHONE_PCM_DIR}/{gesture}/"
    try:
        ls_result = subprocess.run(
            [ADB_PATH, "shell", "ls", phone_dir],
            capture_output=True, text=True, timeout=10)
        if ls_result.returncode != 0:
            print(f"  [{gesture}] 手机目录为空或不存在")
            return []
        phone_files = [f.strip() for f in ls_result.stdout.splitlines()
                       if f.strip().endswith(".pcm")]
    except Exception as e:
        print(f"  [{gesture}] adb ls 失败: {e}")
        return []

    if not phone_files:
        print(f"  [{gesture}] 手机无 PCM 文件")
        return []

    # 已有文件（含 .pcm.pcm 兼容）
    existing = set()
    for f in os.listdir(pcm_dir):
        if f.endswith(".pcm"):
            existing.add(f)
            # 兼容：已有 Push_001.pcm 时，手机同名也算已存在
            if f.endswith(".pcm.pcm"):
                existing.add(f[:-4])
            else:
                existing.add(f + ".pcm")

    new_files = [f for f in phone_files if f not in existing]
    if not new_files:
        print(f"  [{gesture}] 无新文件")
        return []

    # 逐个拉新文件
    pulled = []
    for f in new_files:
        try:
            ret = subprocess.run(
                [ADB_PATH, "pull", phone_dir + f, os.path.join(pcm_dir, f)],
                capture_output=True, text=True, timeout=15)
            if ret.returncode == 0:
                pulled.append(f)
        except Exception:
            pass

    print(f"  [{gesture}] 拉了 {len(pulled)} 个新文件")
    return sorted(pulled)

# =========================================================
# 单文件处理（内部函数，返回数据供 dataset 模式使用）
# =========================================================
def _process_core(fpath):
    """处理一个 PCM，返回 (result_dict, sig_amp, amp_smooth, t_axis, mspec_vmask) 或 None"""
    fname = os.path.basename(fpath)
    stem = os.path.splitext(fname)[0]
    if stem.endswith(".pcm"):
        stem = stem[:-4]

    rx = np.fromfile(fpath, dtype=np.float32)
    duration = len(rx) / FS

    if len(rx) < CHIRP_SAMPLES * 5:
        print(f"  [{stem}] 数据太短 ({duration:.1f}s)，跳过")
        return None

    rx_filt = signal.filtfilt(b_bp, a_bp, rx)

    offset0 = find_global_offset(rx_filt)
    n_chirps = (len(rx_filt) - offset0) // CHIRP_STRIDE
    if n_chirps < 20:
        print(f"  [{stem}] 帧太少 ({n_chirps})，跳过")
        return None

    idx = offset0 + np.arange(n_chirps) * CHIRP_STRIDE
    chirps_td = np.lib.stride_tricks.sliding_window_view(
        rx_filt, CHIRP_SAMPLES)[idx[:n_chirps]]

    static_bg = np.median(chirps_td, axis=0)
    chirps_td = chirps_td - static_bg

    hann_win = np.hanning(CHIRP_SAMPLES)
    beats = chirps_td * np.conj(ref_cplx)[np.newaxis, :]
    beats = sosfiltfilt(sos_lp, beats, axis=1)
    spec_all = np.fft.fft(beats * hann_win[np.newaxis, :], axis=1)[:, :CHIRP_SAMPLES // 2]

    range_bins = CHIRP_SAMPLES // 2
    range_mag  = np.abs(spec_all).T.astype(np.float32)
    range_cplx = spec_all.T.astype(np.complex64)

    f_axis = np.fft.fftfreq(CHIRP_SAMPLES, d=1/FS)[:range_bins]
    r_axis = freq_to_range(f_axis)
    r_mask = r_axis <= MAX_RANGE_CM / 100
    r_axis  = r_axis[r_mask]
    range_mag  = range_mag[r_mask, :]
    range_cplx = range_cplx[r_mask, :]

    r_max = np.max(range_mag)
    if r_max < 1e-9:
        print(f"  [{stem}] 无有效信号，跳过")
        return None
    range_mag /= r_max

    t_axis = np.arange(n_chirps) * CHIRP_STRIDE / FS

    energy   = np.sum(range_mag, axis=1)
    best_bin = np.argmax(energy)
    best_cm  = r_axis[best_bin] * 100

    sig_cplx = range_cplx[best_bin, :]
    sig_amp  = np.abs(sig_cplx)

    # 相位差分 → 微多普勒
    dphase  = np.angle(sig_cplx[1:] * np.conj(sig_cplx[:-1]))
    dop_raw = dphase / (2 * np.pi * CHIRP_STRIDE / FS)

    # SG 平滑多普勒
    win_len = min(11, len(dop_raw) // 2 * 2 + 1)
    if win_len >= 5:
        dop_smooth = savgol_filter(dop_raw, window_length=win_len, polyorder=min(3, win_len - 2))
    else:
        dop_smooth = dop_raw.copy()

    # SG 平滑幅度包络
    sg_win = min(25, len(sig_amp) // 2 * 2 + 1)
    if sg_win >= 5:
        amp_smooth = savgol_filter(sig_amp, window_length=sg_win, polyorder=min(3, sg_win - 2))
    else:
        amp_smooth = sig_amp.copy()

    # 微多普勒谱图
    lo = max(0, best_bin - 5); hi = min(range_cplx.shape[0], best_bin + 6)
    micro = np.mean(range_cplx[lo:hi, :], axis=0)
    nperseg = 32; noverlap = 28
    fd, td, Zd = signal.stft(micro, fs=FS_SLOW, window="hann",
                              nperseg=nperseg, noverlap=noverlap, padded=False)
    mspec = np.abs(Zd); mspec /= mspec.max() + 1e-10
    # per-frequency median subtraction 去除硬件固定频率干扰
    mspec = mspec - np.median(mspec, axis=1, keepdims=True)
    mspec = np.clip(mspec, 0, None)
    vmask = np.abs(fd) <= DOPPLER_LIMIT
    mspec = np.power(mspec, 0.5)
    mspec = ndimage.gaussian_filter(mspec, sigma=0.6)
    mspec_vmask = mspec[vmask, :]

    result = {
        "file": fname,
        "stem": stem,
        "duration_s": round(duration, 2),
        "n_chirps": n_chirps,
        "best_range_cm": round(best_cm, 1),
        "max_doppler_hz": round(float(np.max(np.abs(dop_smooth))), 1),
    }

    return (result, sig_amp, amp_smooth, t_axis, mspec_vmask, fd, td, vmask)


def save_dataset_outputs(data, gesture, dataset_dir):
    """将处理结果存到 dataset/<gesture>/ 的平铺子文件夹"""
    result, sig_amp, amp_smooth, t_axis, mspec_vmask, fd, td, vmask = data
    stem = result["stem"]

    env_dir = os.path.join(dataset_dir, gesture, "幅度包络图")
    dop_dir = os.path.join(dataset_dir, gesture, "微多普勒谱图")
    npy_dir = os.path.join(dataset_dir, gesture, "npy")
    for d in [env_dir, dop_dir, npy_dir]:
        os.makedirs(d, exist_ok=True)

    best_cm = result["best_range_cm"]

    # ---- 图1: 幅度包络 ----
    sg_win = min(25, len(sig_amp) // 2 * 2 + 1)
    if sg_win >= 5:
        amp_smooth_plot = savgol_filter(sig_amp, window_length=sg_win, polyorder=min(3, sg_win - 2))
    else:
        amp_smooth_plot = sig_amp.copy()
    y_lo = max(0, np.percentile(amp_smooth_plot, 5) * 0.6)
    y_hi = np.percentile(amp_smooth_plot, 98) * 1.25

    fig1, ax1 = plt.subplots(figsize=(13, 4))
    ax1.plot(t_axis, sig_amp, color="lightgray", lw=0.6, alpha=0.5, label="原始")
    ax1.plot(t_axis, amp_smooth_plot, color="steelblue", lw=1.8, label="SG平滑 (0.5s窗)")
    ax1.fill_between(t_axis, 0, amp_smooth_plot, alpha=0.12, color="steelblue")
    ax1.set_ylim(y_lo, y_hi)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_xlabel("时间 (s)"); ax1.set_ylabel("幅度")
    ax1.set_title(f"幅度包络 — 距离门 {best_cm:.0f}cm | {stem}")
    fig1.tight_layout()
    fig1.savefig(os.path.join(env_dir, f"{stem}.png"), dpi=100)
    plt.close(fig1)

    # ---- 图2: 微多普勒谱图 ----
    fig2, ax2 = plt.subplots(figsize=(13, 5))
    ax2.imshow(mspec_vmask, aspect="auto", origin="lower",
               extent=[td[0], td[-1], fd[vmask][0], fd[vmask][-1]], cmap="jet")
    ax2.set_ylim(-0.2, -1.4)
    ax2.set_xlabel("时间 (s)"); ax2.set_ylabel("多普勒 (Hz)")
    ax2.set_title(f"微多普勒谱图 (gamma=0.5, gaussian sigma=0.6) | {stem}")
    fig2.tight_layout()
    fig2.savefig(os.path.join(dop_dir, f"{stem}.png"), dpi=100)
    plt.close(fig2)

    # ---- .npy ----
    np.save(os.path.join(npy_dir, f"{stem}_spectrogram.npy"), mspec_vmask)
    np.save(os.path.join(npy_dir, f"{stem}_envelope.npy"), amp_smooth_plot)

    print(f"  [{stem}] 距离={best_cm:.1f}cm  最大多普勒={result['max_doppler_hz']:.1f}Hz  "
          f"帧数={result['n_chirps']}")

    return result


# =========================================================
# 旧模式: 单文件 + 批量处理（保留兼容）
# =========================================================
def process_one_legacy(fpath, out_dir, save_npy=False):
    """旧模式: 输出到 out_dir/<样本名>/ 子目录，包含 03 图"""
    fname = os.path.basename(fpath)
    stem = os.path.splitext(fname)[0]
    if stem.endswith(".pcm"):
        stem = stem[:-4]
    out_subdir = os.path.join(out_dir, stem)
    os.makedirs(out_subdir, exist_ok=True)

    data = _process_core(fpath)
    if data is None:
        return None

    result, sig_amp, amp_smooth, t_axis, mspec_vmask, fd, td, vmask = data
    best_cm = result["best_range_cm"]

    # 幅度包络
    sg_win = min(25, len(sig_amp) // 2 * 2 + 1)
    amp_smooth_plot = savgol_filter(sig_amp, window_length=sg_win, polyorder=min(3, sg_win - 2)) if sg_win >= 5 else sig_amp.copy()
    y_lo = max(0, np.percentile(amp_smooth_plot, 5) * 0.6)
    y_hi = np.percentile(amp_smooth_plot, 98) * 1.25

    fig1, ax1 = plt.subplots(figsize=(13, 4))
    ax1.plot(t_axis, sig_amp, color="lightgray", lw=0.6, alpha=0.5, label="原始")
    ax1.plot(t_axis, amp_smooth_plot, color="steelblue", lw=1.8, label="SG平滑 (0.5s窗)")
    ax1.fill_between(t_axis, 0, amp_smooth_plot, alpha=0.12, color="steelblue")
    ax1.set_ylim(y_lo, y_hi); ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_xlabel("时间 (s)"); ax1.set_ylabel("幅度")
    ax1.set_title(f"幅度包络 — 距离门 {best_cm:.0f}cm | {stem}")
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_subdir, "01_幅度包络.png"), dpi=100)
    plt.close(fig1)

    # 微多普勒谱图
    fig2, ax2 = plt.subplots(figsize=(13, 5))
    ax2.imshow(mspec_vmask, aspect="auto", origin="lower",
               extent=[td[0], td[-1], fd[vmask][0], fd[vmask][-1]], cmap="jet")
    ax2.set_ylim(-0.2, -1.4)
    ax2.set_xlabel("时间 (s)"); ax2.set_ylabel("多普勒 (Hz)")
    ax2.set_title(f"微多普勒谱图 (gamma=0.5, gaussian sigma=0.6) | {stem}")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_subdir, "02_微多普勒谱图.png"), dpi=100)
    plt.close(fig2)

    # 03 诊断图
    dphase = np.angle(data[1][1:] * np.conj(data[1][:-1]))
    dop_raw = dphase / (2 * np.pi * CHIRP_STRIDE / FS)
    dop_time = t_axis[:-1] + (CHIRP_STRIDE / FS) / 2
    win_len = min(11, len(dop_raw) // 2 * 2 + 1)
    dop_smooth = savgol_filter(dop_raw, window_length=win_len, polyorder=min(3, win_len - 2)) if win_len >= 5 else dop_raw.copy()

    fig3, (ax_a, ax_p, ax_d) = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    ax_a.plot(t_axis, sig_amp); ax_a.grid(alpha=.3)
    ax_a.set_ylabel("幅度"); ax_a.set_title(f"距离门 {best_cm:.0f}cm — 幅度")
    ax_p.plot(t_axis, np.angle(data[1]), color="green", alpha=.7)
    ax_p.grid(alpha=.3); ax_p.set_ylabel("相位 (rad)")
    ax_p.set_title(f"缠绕相位 ({FS_SLOW:.0f}Hz 慢时间)")
    ax_d.plot(dop_time, dop_raw,  lw=.5, color="gray", alpha=.4, label="原始")
    ax_d.plot(dop_time, dop_smooth, lw=1.5, color="red", label="SG 平滑")
    ax_d.axhline(0, color="gray", ls="--", lw=.5)
    ax_d.grid(alpha=.3); ax_d.legend()
    ax_d.set_xlabel("时间 (s)"); ax_d.set_ylabel("多普勒 (Hz)")
    ax_d.set_title(f"微多普勒 (Nyquist={FS_SLOW/2:.0f}Hz)")
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_subdir, "03_幅度相位多普勒.png"), dpi=100)
    plt.close(fig3)

    if save_npy:
        np.save(os.path.join(out_subdir, f"{stem}_spectrogram.npy"), mspec_vmask)
        np.save(os.path.join(out_subdir, f"{stem}_envelope.npy"), amp_smooth_plot)

    print(f"  [{stem}] 距离={best_cm:.1f}cm  最大多普勒={result['max_doppler_hz']:.1f}Hz  "
          f"帧数={result['n_chirps']}  ->  {out_subdir}")
    return result


# =========================================================
# 数据集模式: 拉数据 + 处理
# =========================================================
def process_dataset(dataset_dir, gesture=None, do_pull=True):
    """数据集模式: adb pull + 处理。只处理被更新的文件。"""
    gestures = [gesture] if gesture else GESTURES
    pull_changed = {}

    if do_pull:
        print("--- 从手机拉取数据 ---")
        for g in gestures:
            pcm_dir = os.path.join(dataset_dir, g, "pcm")
            changed = adb_pull(g, pcm_dir)
            pull_changed[g] = changed

    print("\n--- 处理 PCM ---")
    t0 = time.time()
    total_ok = 0

    for g in gestures:
        pcm_dir = os.path.join(dataset_dir, g, "pcm")
        if not os.path.isdir(pcm_dir):
            print(f"  [{g}] 无 PCM 目录，跳过")
            continue

        # 有 pull 结果则只处理被更新的文件，否则处理全部
        if do_pull and g in pull_changed:
            files = pull_changed[g]
            if not files:
                print(f"  [{g}] 无文件需要处理")
                continue
            print(f"\n[{g}] 处理 {len(files)} 个更新文件:")
        else:
            files = sorted([f for f in os.listdir(pcm_dir) if f.endswith(".pcm")])
            if not files:
                print(f"  [{g}] 无 PCM 文件")
                continue
            print(f"\n[{g}] {len(files)} 个文件:")

        ok = 0
        for i, fname in enumerate(files, 1):
            print(f"  [{i}/{len(files)}]", end=" ")
            try:
                data = _process_core(os.path.join(pcm_dir, fname))
                if data is None:
                    continue
                save_dataset_outputs(data, g, dataset_dir)
                ok += 1
            except Exception as e:
                print(f"  失败: {e}")
        total_ok += ok

    elapsed = time.time() - t0
    print(f"\n处理完成，耗时 {elapsed:.1f}s | 成功={total_ok}")


# =========================================================
# 命令行入口
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="UltraGesture PCM 处理 (V4 数据集模式)")
    parser.add_argument("--gesture", default=None,
                        help="只处理指定手势 (默认: 全部7种)")
    parser.add_argument("--no-pull", action="store_true",
                        help="跳过 adb pull，直接处理本地 PCM")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"旧模式: PCM 数据目录 (默认: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"旧模式: 输出目录 (默认: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--file", default=None,
                        help="旧模式: 处理单个 PCM 文件")
    parser.add_argument("--watch", action="store_true",
                        help="旧模式: 监控目录自动处理")
    parser.add_argument("--save-npy", action="store_true",
                        help="旧模式: 同时输出 .npy 训练数据")
    parser.add_argument("--legacy", action="store_true",
                        help="使用旧模式 (非 dataset)")

    args = parser.parse_args()

    print("=" * 55)
    print(f"UltraGesture V4 处理器")
    print(f"  FS_SLOW={FS_SLOW:.0f}Hz  Nyquist={FS_SLOW/2:.0f}Hz")
    print(f"  距离分辨率={RANGE_RES*100:.1f}cm")
    print(f"  Doppler 限幅=±{DOPPLER_LIMIT}Hz")
    print("=" * 55)

    if args.legacy or args.watch or args.file:
        # 旧模式
        os.makedirs(args.output_dir, exist_ok=True)
        if args.watch:
            # 简化监控模式，略
            print("监控模式暂不支持，请用 dataset 模式")
        elif args.file:
            fpath = args.file if os.path.isabs(args.file) else os.path.join(args.data_dir, args.file)
            if not os.path.exists(fpath):
                print(f"文件不存在: {fpath}")
                sys.exit(1)
            process_one_legacy(fpath, args.output_dir, save_npy=args.save_npy)
        else:
            # 旧批量模式
            files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".pcm")])
            if not files:
                print("没有找到 .pcm 文件")
                return
            print(f"找到 {len(files)} 个文件\n")
            t0 = time.time()
            results = []; ok = 0
            for i, fname in enumerate(files, 1):
                print(f"[{i}/{len(files)}]", end=" ")
                try:
                    r = process_one_legacy(os.path.join(args.data_dir, fname), args.output_dir, args.save_npy)
                    if r: results.append(r); ok += 1
                except Exception as e:
                    print(f"  失败: {e}")
            elapsed = time.time() - t0
            print(f"\n处理完成，耗时 {elapsed:.1f}s | 成功={ok}")
            if results:
                with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        # 数据集模式（默认）
        process_dataset(DATASET_DIR, gesture=args.gesture, do_pull=not args.no_pull)

if __name__ == "__main__":
    main()
