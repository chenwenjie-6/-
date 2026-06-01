"""
四宫格对比脚本 — 原始 vs 处理后
用法: python sigongge.py <pcm文件路径>
输出: 左列为未经任何处理的原始谱图/包络，右列为完整信号管线处理后的结果
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as signal
from scipy.signal import savgol_filter, sosfiltfilt
import scipy.ndimage as ndimage
import os, sys, argparse

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

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
CHIRP_SLOPE   = BANDWIDTH / CHIRP_TIME
CHIRP_STRIDE  = CHIRP_SAMPLES
FS_SLOW       = FS / CHIRP_STRIDE     # 50 Hz
MAX_RANGE_CM  = 100
DOPPLER_LIMIT = 80

b_bp, a_bp = signal.butter(6, [START_FREQ/(FS/2), END_FREQ/(FS/2)], btype="band")
b_lp, a_lp = signal.butter(6, 2500/(FS/2), btype="low")
sos_lp     = signal.tf2sos(b_lp, a_lp)

# =========================================================
# 参考信号
# =========================================================
def make_ref_chirp_real():
    t = np.arange(CHIRP_SAMPLES) / FS
    phase = 2*np.pi * (START_FREQ*t + 0.5*CHIRP_SLOPE*t*t)
    return 0.9 * np.sin(phase) * np.hanning(CHIRP_SAMPLES)

def make_ref_chirp_complex():
    t = np.arange(CHIRP_SAMPLES) / FS
    phase = 2*np.pi * (START_FREQ*t + 0.5*CHIRP_SLOPE*t*t)
    return 0.9 * np.exp(1j*phase) * np.hanning(CHIRP_SAMPLES)

ref_real = make_ref_chirp_real()
ref_cplx = make_ref_chirp_complex()

def find_global_offset(rx, search_len=None):
    if search_len is None:
        search_len = CHIRP_SAMPLES * 5
    search_len = min(search_len, len(rx) - CHIRP_SAMPLES)
    corr = np.correlate(rx[:search_len], ref_real, mode="valid")
    return int(np.argmax(np.abs(corr)))

def freq_to_range(f):
    return f * SOUND_SPEED / (2 * CHIRP_SLOPE)


def process_raw(fpath):
    """原图：只做带通+互相关对齐+去斜+距离FFT，不做背景扣除/低通/SG等增强"""
    rx = np.fromfile(fpath, dtype=np.float32)
    if len(rx) < CHIRP_SAMPLES * 5:
        return None

    rx_filt = signal.filtfilt(b_bp, a_bp, rx)
    offset0 = find_global_offset(rx_filt)
    n_chirps = (len(rx_filt) - offset0) // CHIRP_STRIDE
    if n_chirps < 20:
        return None

    idx = offset0 + np.arange(n_chirps) * CHIRP_STRIDE
    chirps = np.lib.stride_tricks.sliding_window_view(
        rx_filt, CHIRP_SAMPLES)[idx[:n_chirps]]

    # 去斜 + 距离FFT（无低通、无背景扣除）
    beats = chirps * np.conj(ref_cplx)[np.newaxis, :]
    spec_all = np.fft.fft(beats, axis=1)[:, :CHIRP_SAMPLES//2]

    range_bins = CHIRP_SAMPLES // 2
    range_mag  = np.abs(spec_all).T.astype(np.float32)
    range_cplx = spec_all.T.astype(np.complex64)
    f_axis = np.fft.fftfreq(CHIRP_SAMPLES, d=1/FS)[:range_bins]
    r_axis = freq_to_range(f_axis)
    r_mask = r_axis <= MAX_RANGE_CM / 100
    r_axis = r_axis[r_mask]
    range_mag = range_mag[r_mask, :]
    range_cplx = range_cplx[r_mask, :]

    r_max = np.max(range_mag)
    if r_max < 1e-9:
        return None
    range_mag /= r_max

    t_axis = np.arange(n_chirps) * CHIRP_STRIDE / FS
    energy = np.sum(range_mag, axis=1)
    best_bin = np.argmax(energy)
    sig_amp = range_mag[best_bin, :]

    # 微多普勒谱图：复数幅度跨距离bin平均
    lo = max(0, best_bin - 5)
    hi = min(range_cplx.shape[0], best_bin + 6)
    micro = np.mean(range_cplx[lo:hi, :], axis=0)
    nperseg = 32; noverlap = 28
    fd, td, Zd = signal.stft(micro, fs=FS_SLOW, window="hann",
                              nperseg=nperseg, noverlap=noverlap,
                              padded=False, return_onesided=False)
    mspec = np.abs(Zd)
    mspec /= mspec.max() + 1e-10
    mspec = np.power(mspec, 0.3)
    vmask = np.abs(fd) <= DOPPLER_LIMIT

    return {"mspec": mspec[vmask], "amp": sig_amp, "fd": fd, "td": td,
            "vmask": vmask, "t_axis": t_axis}


def process_full(fpath):
    """处理后：完整信号管线"""
    rx = np.fromfile(fpath, dtype=np.float32)
    if len(rx) < CHIRP_SAMPLES * 5:
        return None

    rx_filt = signal.filtfilt(b_bp, a_bp, rx)
    offset0 = find_global_offset(rx_filt)
    n_chirps = (len(rx_filt) - offset0) // CHIRP_STRIDE
    if n_chirps < 20:
        return None

    idx = offset0 + np.arange(n_chirps) * CHIRP_STRIDE
    chirps = np.lib.stride_tricks.sliding_window_view(
        rx_filt, CHIRP_SAMPLES)[idx[:n_chirps]]

    # 静态背景扣除
    static_bg = np.median(chirps, axis=0)
    chirps = chirps - static_bg

    # 去斜 + 低通 + 距离FFT
    hann_win = np.hanning(CHIRP_SAMPLES)
    beats = chirps * np.conj(ref_cplx)[np.newaxis, :]
    beats = sosfiltfilt(sos_lp, beats, axis=1)
    spec_all = np.fft.fft(beats * hann_win[np.newaxis, :], axis=1)[:, :CHIRP_SAMPLES//2]

    range_bins = CHIRP_SAMPLES // 2
    range_mag  = np.abs(spec_all).T.astype(np.float32)
    range_cplx = spec_all.T.astype(np.complex64)
    f_axis = np.fft.fftfreq(CHIRP_SAMPLES, d=1/FS)[:range_bins]
    r_axis = freq_to_range(f_axis)
    r_mask = r_axis <= MAX_RANGE_CM / 100
    r_axis = r_axis[r_mask]
    range_mag = range_mag[r_mask, :]
    range_cplx = range_cplx[r_mask, :]

    r_max = np.max(range_mag)
    if r_max < 1e-9:
        return None
    range_mag /= r_max

    t_axis = np.arange(n_chirps) * CHIRP_STRIDE / FS
    energy = np.sum(range_mag, axis=1)
    best_bin = np.argmax(energy)

    sig_cplx = range_cplx[best_bin, :]
    sig_amp  = np.abs(sig_cplx)

    # SG 平滑包络
    sg_win = min(25, len(sig_amp) // 2 * 2 + 1)
    amp_smooth = savgol_filter(sig_amp, window_length=sg_win,
                               polyorder=min(3, sg_win - 2)) if sg_win >= 5 else sig_amp.copy()

    # 微多普勒谱图 + per-freq median + gamma + gaussian
    lo = max(0, best_bin - 5)
    hi = min(range_cplx.shape[0], best_bin + 6)
    micro = np.mean(range_cplx[lo:hi, :], axis=0)
    nperseg = 32; noverlap = 28
    fd, td, Zd = signal.stft(micro, fs=FS_SLOW, window="hann",
                              nperseg=nperseg, noverlap=noverlap, padded=False)
    mspec = np.abs(Zd); mspec /= mspec.max() + 1e-10
    mspec = mspec - np.median(mspec, axis=1, keepdims=True)
    mspec = np.clip(mspec, 0, None)
    vmask = np.abs(fd) <= DOPPLER_LIMIT
    mspec = np.power(mspec, 0.5)
    mspec = ndimage.gaussian_filter(mspec, sigma=0.6)

    return {"mspec": mspec[vmask], "amp": amp_smooth, "fd": fd, "td": td,
            "vmask": vmask, "t_axis": t_axis}


def draw_comparison(raw, full, stem, out_path):
    """左列：原始 | 右列：处理后"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    td_full = full["td"]
    fd_v = full["fd"][full["vmask"]]
    fd_raw = raw["fd"]
    fd_raw_v = fd_raw[raw["vmask"]]

    for ax in axes.flat:
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)

    # 左上：原始谱图
    axes[0, 0].imshow(raw["mspec"], aspect="auto", origin="lower",
                      extent=[td_full[0], td_full[-1], fd_raw_v[0], fd_raw_v[-1]],
                      cmap="jet")
    axes[0, 0].set_ylim(-0.2, -1.4)

    # 右上：处理后谱图
    axes[0, 1].imshow(full["mspec"], aspect="auto", origin="lower",
                      extent=[td_full[0], td_full[-1], fd_v[0], fd_v[-1]], cmap="jet")
    axes[0, 1].set_ylim(-0.2, -1.4)

    # 左下：原始包络
    axes[1, 0].plot(raw["amp"], color="gray", lw=1)

    # 右下：处理后包络
    axes[1, 1].plot(full["amp"], color="steelblue", lw=1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"四宫格对比图已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="原始 vs 处理后 四宫格对比")
    parser.add_argument("pcm", help="PCM 文件路径")
    parser.add_argument("-o", "--output", default=None,
                        help="输出路径 (默认: pcm同目录下 <stem>_四宫格.png)")
    args = parser.parse_args()

    fpath = args.pcm
    if not os.path.isfile(fpath):
        print(f"文件不存在: {fpath}")
        sys.exit(1)

    stem = os.path.splitext(os.path.basename(fpath))[0]
    if stem.endswith(".pcm"):
        stem = stem[:-4]

    print(f"处理: {fpath}")
    raw  = process_raw(fpath)
    full = process_full(fpath)
    if raw is None or full is None:
        print("数据不足或无有效信号，跳过")
        sys.exit(1)

    out_path = args.output or os.path.join(os.path.dirname(fpath),
                                           f"{stem}_四宫格.png")
    draw_comparison(raw, full, stem, out_path)


if __name__ == "__main__":
    main()
