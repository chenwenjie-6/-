"""
生成 C++ 头文件：滤波器系数、参考chirp、窗函数等所有预计算常量。
用法: python gen_constants.py
输出: cpp_output/constants.h, cpp_output/biquad_coeffs.h
"""
import numpy as np
import scipy.signal as signal
from scipy.signal import savgol_filter, savgol_coeffs
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp_output")

FS = 48000
START_FREQ = 17000.0
END_FREQ   = 23000.0
CHIRP_TIME = 0.02
CHIRP_SAMPLES = int(FS * CHIRP_TIME)  # 960
CHIRP_SLOPE   = (END_FREQ - START_FREQ) / CHIRP_TIME
CHIRP_STRIDE  = CHIRP_SAMPLES
FS_SLOW = FS / CHIRP_STRIDE  # 50 Hz

def write_constants_h():
    """写 constants.h: 参考chirp、窗函数、SG核、高斯核等"""
    lines = []
    lines.append("// 自动生成，勿手动修改")
    lines.append("#pragma once")
    lines.append("#include <cmath>")
    lines.append("")
    lines.append("// ========== 基础参数 ==========")
    lines.append(f"constexpr int kSampleRate = {FS};")
    lines.append(f"constexpr double kStartFreq = {START_FREQ};")
    lines.append(f"constexpr double kEndFreq = {END_FREQ};")
    lines.append(f"constexpr double kBandwidth = {END_FREQ - START_FREQ};")
    lines.append(f"constexpr double kChirpTime = {CHIRP_TIME};")
    lines.append(f"constexpr int kChirpSamples = {CHIRP_SAMPLES};")
    lines.append(f"constexpr double kChirpSlope = {CHIRP_SLOPE};")
    lines.append(f"constexpr int kChirpStride = {CHIRP_STRIDE};")
    lines.append(f"constexpr double kSlowFS = {FS_SLOW};")
    lines.append(f"constexpr int kMaxRangeCm = 100;")
    lines.append(f"constexpr int kDopplerLimit = 80;")
    lines.append(f"constexpr int kRangeBins = {CHIRP_SAMPLES // 2};")
    lines.append("")

    # --- 参考chirp（实数和复数） ---
    t = np.arange(CHIRP_SAMPLES) / FS
    phase = 2*np.pi * (START_FREQ*t + 0.5*CHIRP_SLOPE*t*t)
    hann = np.hanning(CHIRP_SAMPLES)
    ref_real = 0.9 * np.sin(phase) * hann
    ref_cplx = 0.9 * np.exp(1j*phase) * hann

    lines.append(f"// 参考chirp实数 (len={CHIRP_SAMPLES})")
    lines.append(f"const float kRefChirpReal[{CHIRP_SAMPLES}] = {{")
    vals = ", ".join([f"{v:.10f}f" for v in ref_real])
    for i in range(0, len(ref_real), 8):
        lines.append("    " + ", ".join([f"{v:.10f}f" for v in ref_real[i:i+8]]) + ",")
    lines.append("};")
    lines.append("")

    lines.append(f"// 参考chirp复数实部/虚部 (len={CHIRP_SAMPLES})")
    lines.append(f"const float kRefChirpReal2[{CHIRP_SAMPLES}] = {{")
    for i in range(0, CHIRP_SAMPLES, 8):
        lines.append("    " + ", ".join([f"{v:.10f}f" for v in ref_cplx.real[i:i+8]]) + ",")
    lines.append("};")
    lines.append(f"const float kRefChirpImag[{CHIRP_SAMPLES}] = {{")
    for i in range(0, CHIRP_SAMPLES, 8):
        lines.append("    " + ", ".join([f"{v:.10f}f" for v in ref_cplx.imag[i:i+8]]) + ",")
    lines.append("};")
    lines.append("")

    # --- Hann窗 (960点) ---
    lines.append(f"const float kHannWindow[{CHIRP_SAMPLES}] = {{")
    for i in range(0, CHIRP_SAMPLES, 8):
        lines.append("    " + ", ".join([f"{v:.10f}f" for v in hann[i:i+8]]) + ",")
    lines.append("};")
    lines.append("")

    # --- STFT参数 ---
    nperseg = 32
    noverlap = 28
    n_stft_windows = (100 - nperseg) // (nperseg - noverlap) + 1  # approx from max envelope length
    hann32 = np.hanning(nperseg)
    lines.append(f"constexpr int kStftNperseg = {nperseg};")
    lines.append(f"constexpr int kStftNoverlap = {noverlap};")
    lines.append(f"constexpr int kStftHop = {nperseg - noverlap};")
    lines.append(f"constexpr int kStftFreqBins = {nperseg};")
    lines.append(f"constexpr int kMaxTimeBins = 30;  // 足够覆盖所有样本")
    lines.append("")
    lines.append(f"const float kStftHann32[{nperseg}] = {{")
    lines.append("    " + ", ".join([f"{v:.10f}f" for v in hann32]) + "")
    lines.append("};")
    lines.append("")

    # --- SG平滑系数 (window=25, polyorder=3), 完整 25x25 矩阵 ---
    sg_full = np.array([savgol_coeffs(25, 3, pos=i, use="dot") for i in range(25)])
    lines.append("// SG平滑 25x25 完整系数矩阵 (scipy savgol_coeffs, pos=0..24)")
    lines.append("constexpr int kSgWindow = 25;")
    lines.append("constexpr int kSgHalf = 12;")
    lines.append("const float kSgCoeffsFull[25][25] = {")
    for row in sg_full:
        vals = ", ".join([f"{v:.10f}f" for v in row])
        lines.append(f"    {{{vals}}},")
    lines.append("};")
    lines.append("")

    # --- SG平滑系数 (window=11, polyorder=3), 完整 11x11 矩阵 ---
    sg11 = np.array([savgol_coeffs(11, 3, pos=i, use="dot") for i in range(11)])
    lines.append("// SG平滑 11x11 完整系数矩阵")
    lines.append("constexpr int kSgWindow11 = 11;")
    lines.append("constexpr int kSgHalf11 = 5;")
    lines.append("const float kSgCoeffs11Full[11][11] = {")
    for row in sg11:
        vals = ", ".join([f"{v:.10f}f" for v in row])
        lines.append(f"    {{{vals}}},")
    lines.append("};")
    lines.append("")

    # --- 高斯平滑核 (sigma=0.6, 5x5) ---
    import scipy.ndimage as ndimage
    gauss_kernel = np.zeros((5, 5))
    gauss_kernel[2, 2] = 1.0
    gauss_kernel = ndimage.gaussian_filter(gauss_kernel, sigma=0.6)
    lines.append("// 2D高斯平滑核 (sigma=0.6, 5x5)")
    lines.append("constexpr int kGaussSize = 5;")
    lines.append("constexpr int kGaussHalf = 2;")
    lines.append("const float kGaussKernel[5][5] = {")
    for row in gauss_kernel:
        vals = ", ".join([f"{v:.10f}f" for v in row])
        lines.append(f"    {{{vals}}},")
    lines.append("};")
    lines.append("")

    # --- 距离-频率映射 ---
    SOUND_SPEED = 343.0
    f_axis = np.fft.fftfreq(CHIRP_SAMPLES, d=1/FS)[:CHIRP_SAMPLES//2]
    r_axis = f_axis * SOUND_SPEED / (2 * CHIRP_SLOPE)
    r_mask = r_axis <= 1.0  # MAX_RANGE_CM / 100
    valid_bins = int(np.sum(r_mask))
    lines.append(f"// 距离门有效bin数 (<= {100}cm)")
    lines.append(f"constexpr int kValidRangeBins = {valid_bins};")
    lines.append("")

    # --- 模型输入尺寸 ---
    lines.append("// CNN模型输入输出尺寸")
    lines.append("constexpr int kSpecFreq = 32;")
    lines.append("constexpr int kSpecTime = 26;")
    lines.append("constexpr int kEnvelopeLen = 100;")
    lines.append("constexpr int kNumClasses = 5;")
    lines.append("")

    with open(os.path.join(OUT_DIR, "constants.h"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已生成: constants.h")


def write_biquad_coeffs_h():
    """写 biquad_coeffs.h: 带通和低通滤波器的biquad系数"""
    nyq = FS / 2

    # 6阶带通 (17-23kHz) → 3个biquad
    sos_bp = signal.butter(6, [START_FREQ/nyq, END_FREQ/nyq], btype="band", output="sos")
    # 6阶低通 (2.5kHz) → 3个biquad
    sos_lp = signal.butter(6, 2500/nyq, btype="low", output="sos")

    lines = []
    lines.append("// 自动生成，勿手动修改")
    lines.append("#pragma once")
    lines.append("")
    lines.append("// 单个Biquad节 (Direct Form I)")
    lines.append("struct BiquadSection {")
    lines.append("    float b0, b1, b2;")
    lines.append("    float a1, a2;  // a0隐含为1")
    lines.append("};")
    lines.append("")

    for name, sos in [("kSosBandpass", sos_bp), ("kSosLowpass", sos_lp)]:
        n = sos.shape[0]
        lines.append(f"// {name}: {n}个biquad节")
        lines.append(f"const BiquadSection {name}[{n}] = {{")
        for sec in sos:
            lines.append(f"    {{{sec[0]:.10f}f, {sec[1]:.10f}f, {sec[2]:.10f}f, {sec[4]:.10f}f, {sec[5]:.10f}f}},")
        lines.append("};")
        lines.append(f"constexpr int {name}Count = {n};")
        lines.append("")

    with open(os.path.join(OUT_DIR, "biquad_coeffs.h"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已生成: biquad_coeffs.h")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    write_constants_h()
    write_biquad_coeffs_h()
    print(f"\n输出目录: {OUT_DIR}")

if __name__ == "__main__":
    main()
