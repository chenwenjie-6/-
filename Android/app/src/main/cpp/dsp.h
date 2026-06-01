#pragma once

#include "constants.h"
#include "biquad.h"
#include "fft.h"

#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>

// ======================================================
// 辅助函数
// ======================================================

// 互相关找第一个 chirp 的起始偏移
inline int findChirpOffset(const float* rx, int rxLen) {
    int searchLen = std::min(kChirpSamples * 5, rxLen - kChirpSamples);
    float bestCorr = 0;
    int bestOff = 0;
    for (int off = 0; off < searchLen; ++off) {
        float corr = 0;
        for (int j = 0; j < kChirpSamples; ++j) {
            corr += rx[off + j] * kRefChirpReal[j];
        }
        corr = std::abs(corr);
        if (corr > bestCorr) {
            bestCorr = corr;
            bestOff = off;
        }
    }
    return bestOff;
}

// 中值：原地排序取中间值
inline float median(std::vector<float>& v) {
    size_t n = v.size();
    size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    return v[mid];
}

// 1D SG 卷积, 使用完整系数矩阵匹配 scipy savgol_filter
// kSgCoeffsFull[row] 对应 scipy savgol_coeffs(25, 3, pos=row)
inline std::vector<float> sgSmooth(const std::vector<float>& x) {
    int n = (int)x.size();
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) {
        int row, lo;
        if (i < kSgHalf) {
            row = i; lo = 0;                      // 左边界, 窗口对齐信号起始
        } else if (i >= n - kSgHalf) {
            row = kSgWindow - n + i; lo = n - kSgWindow;  // 右边界, 窗口对齐信号末尾
        } else {
            row = kSgHalf; lo = i - kSgHalf;      // 中间, 窗口居中
        }
        const float* coeffs = kSgCoeffsFull[row];
        float sum = 0;
        for (int k = 0; k < kSgWindow; ++k) {
            sum += x[lo + k] * coeffs[k];
        }
        out[i] = sum;
    }
    return out;
}

// 2D 高斯平滑 (same mode, 反射边界)
inline std::vector<float> gauss2D(const std::vector<float>& img,
                                   int rows, int cols) {
    std::vector<float> out(rows * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float sum = 0;
            for (int dr = -kGaussHalf; dr <= kGaussHalf; ++dr) {
                for (int dc = -kGaussHalf; dc <= kGaussHalf; ++dc) {
                    int rr = r + dr, cc = c + dc;
                    if (rr < 0) rr = -rr;
                    if (rr >= rows) rr = 2 * rows - 2 - rr;
                    if (cc < 0) cc = -cc;
                    if (cc >= cols) cc = 2 * cols - 2 - cc;
                    sum += img[rr * cols + cc]
                         * kGaussKernel[dr + kGaussHalf][dc + kGaussHalf];
                }
            }
            out[r * cols + c] = sum;
        }
    }
    return out;
}

// 线性插值 resize: 将 (inRows, inCols) 矩阵缩放到 (outRows, outCols)
inline std::vector<float> resize2D(const std::vector<float>& src,
                                    int inRows, int inCols,
                                    int outRows, int outCols) {
    std::vector<float> dst(outRows * outCols);
    float rowScale = (float)inRows / outRows;
    float colScale = (float)inCols / outCols;
    for (int r = 0; r < outRows; ++r) {
        float sr = (r + 0.5f) * rowScale - 0.5f;
        if (sr < 0) sr = 0;
        if (sr >= inRows - 1) sr = (float)(inRows - 2);
        int sr0 = (int)sr;
        float wr = sr - sr0;
        for (int c = 0; c < outCols; ++c) {
            float sc = (c + 0.5f) * colScale - 0.5f;
            if (sc < 0) sc = 0;
            if (sc >= inCols - 1) sc = (float)(inCols - 2);
            int sc0 = (int)sc;
            float wc = sc - sc0;
            float v00 = src[sr0     * inCols + sc0];
            float v01 = src[sr0     * inCols + sc0 + 1];
            float v10 = src[(sr0+1) * inCols + sc0];
            float v11 = src[(sr0+1) * inCols + sc0 + 1];
            dst[r * outCols + c] =
                v00 * (1-wr) * (1-wc) + v01 * (1-wr) * wc +
                v10 * wr * (1-wc)      + v11 * wr * wc;
        }
    }
    return dst;
}

// 1D 线性插值 resize
inline std::vector<float> resize1D(const std::vector<float>& src,
                                    int outLen) {
    int n = (int)src.size();
    std::vector<float> dst(outLen);
    float scale = (float)(n - 1) / (outLen - 1);
    for (int i = 0; i < outLen; ++i) {
        float pos = i * scale;
        int idx = (int)pos;
        float frac = pos - idx;
        if (idx >= n - 1) {
            dst[i] = src[n - 1];
        } else {
            dst[i] = src[idx] * (1 - frac) + src[idx + 1] * frac;
        }
    }
    return dst;
}

// 频率→距离
inline float freqToRange(float f) {
    return f * 343.0f / (2.0f * kChirpSlope);
}

// ======================================================
// ChirpProcessor — 2秒录音 → 谱图+包络
// ======================================================

struct ProcessResult {
    std::vector<float> spectrogram;  // specRows * specCols, row-major
    std::vector<float> envelope;     // envLen
    int specRows = 0;
    int specCols = 0;
    int envLen = 0;
    int nChirps = 0;
    float bestRangeCm = 0;
    float maxDopplerHz = 0;
};

class ChirpProcessor {
public:
    ChirpProcessor() {
        bp_.init(kSosBandpass);
    }

    // 处理 2 秒 PCM 数据，返回 CNN 就绪的谱图和包络
    ProcessResult process(const float* rawPcm, int numSamples) {
        ProcessResult result;

        if (numSamples < kChirpSamples * 5) return result;

        // ---- 1. 带通滤波 (filtfilt, 匹配 Python) ----
        std::vector<float> rx(numSamples);
        bp_.processFiltFilt(rawPcm, rx.data(), numSamples);

        // ---- 2. 互相关对齐 ----
        int offset0 = findChirpOffset(rx.data(), (int)rx.size());
        int nChirps = ((int)rx.size() - offset0) / kChirpStride;
        if (nChirps < 20) return result;
        result.nChirps = nChirps;

        // ---- 3. 提取 chirp 帧 ----
        // chirps[i][s] = rx[offset0 + i*kChirpStride + s]
        std::vector<std::vector<float>> chirps(nChirps,
                                                std::vector<float>(kChirpSamples));
        for (int i = 0; i < nChirps; ++i) {
            int start = offset0 + i * kChirpStride;
            for (int s = 0; s < kChirpSamples; ++s) {
                chirps[i][s] = rx[start + s];
            }
        }

        // ---- 4. 静态背景去除 (每列中值) ----
        std::vector<float> bg(kChirpSamples);
        std::vector<float> colVals(nChirps);
        for (int s = 0; s < kChirpSamples; ++s) {
            for (int i = 0; i < nChirps; ++i) {
                colVals[i] = chirps[i][s];
            }
            bg[s] = median(colVals);
        }
        for (int i = 0; i < nChirps; ++i) {
            for (int s = 0; s < kChirpSamples; ++s) {
                chirps[i][s] -= bg[s];
            }
        }

        // ---- 5. Dechirp (复数乘法) ----
        // beats[i][s] = chirps[i][s] * conj(ref_cplx[s])
        // 预计算每个chirp去斜后的复信号
        std::vector<std::vector<std::complex<float>>> beats(
            nChirps, std::vector<std::complex<float>>(kChirpSamples));
        for (int i = 0; i < nChirps; ++i) {
            for (int s = 0; s < kChirpSamples; ++s) {
                float refReal = kRefChirpReal2[s];
                float refImag = kRefChirpImag[s];
                beats[i][s] = std::complex<float>(
                    chirps[i][s] * refReal,
                    chirps[i][s] * (-refImag));
            }
        }

        // ---- 6. 低通滤波 (filtfilt, 分离实部/虚部) ----
        for (int i = 0; i < nChirps; ++i) {
            std::vector<float> realPart(kChirpSamples), imagPart(kChirpSamples);
            for (int s = 0; s < kChirpSamples; ++s) {
                realPart[s] = beats[i][s].real();
                imagPart[s] = beats[i][s].imag();
            }
            LowpassFilter lpR, lpI;
            lpR.init(kSosLowpass);
            lpI.init(kSosLowpass);
            std::vector<float> realFilt(kChirpSamples), imagFilt(kChirpSamples);
            lpR.processFiltFilt(realPart.data(), realFilt.data(), kChirpSamples);
            lpI.processFiltFilt(imagPart.data(), imagFilt.data(), kChirpSamples);
            for (int s = 0; s < kChirpSamples; ++s) {
                beats[i][s] = std::complex<float>(realFilt[s], imagFilt[s]);
            }
        }

        // ---- 7. 距离 FFT (960点, 匹配 Python) ----
        int rangeFFT = kChirpSamples;  // 960
        int halfRange = rangeFFT / 2;   // 480
        std::vector<std::vector<std::complex<float>>> specAll(
            nChirps, std::vector<std::complex<float>>(halfRange));
        for (int i = 0; i < nChirps; ++i) {
            std::vector<std::complex<float>> fftIn(rangeFFT);
            for (int s = 0; s < kChirpSamples; ++s) {
                fftIn[s] = beats[i][s] * kHannWindow[s];
            }
            fft960_.fft(fftIn.data());
            for (int k = 0; k < halfRange; ++k) {
                specAll[i][k] = fftIn[k];
            }
        }

        // ---- 8. 距离门选择 ----
        int validBins = std::min(halfRange, kValidRangeBins);
        std::vector<float> energy(validBins, 0);
        for (int k = 0; k < validBins; ++k) {
            for (int i = 0; i < nChirps; ++i) {
                energy[k] += std::norm(specAll[i][k]);
            }
        }
        int bestBin = 0;
        float bestEnergy = 0;
        for (int k = 0; k < validBins; ++k) {
            if (energy[k] > bestEnergy) {
                bestEnergy = energy[k];
                bestBin = k;
            }
        }
        result.bestRangeCm = freqToRange(
            (float)bestBin * kSampleRate / rangeFFT) * 100;

        // 提取最强距离门的复信号
        std::vector<std::complex<float>> sigCplx(nChirps);
        for (int i = 0; i < nChirps; ++i) {
            sigCplx[i] = specAll[i][bestBin];
        }

        // ---- 9. 相位差分 → 微多普勒 ----
        std::vector<float> sigAmp(nChirps);
        sigAmp[0] = std::abs(sigCplx[0]);
        for (int i = 1; i < nChirps; ++i) {
            sigAmp[i] = std::abs(sigCplx[i]);
        }

        // ---- 10. SG 平滑幅度包络 ----
        std::vector<float> ampSmooth = sgSmooth(sigAmp);

        // 微多普勒: phase difference / (2*pi*dt)
        int sgWin = std::min(11, (nChirps - 1) / 2 * 2 + 1);
        float dt = kChirpStride / (float)kSampleRate;
        std::vector<float> dopRaw(nChirps - 1);
        for (int i = 1; i < nChirps; ++i) {
            float dphase = std::arg(sigCplx[i] * std::conj(sigCplx[i - 1]));
            dopRaw[i - 1] = dphase / (2.0f * M_PI * dt);
        }

        // ---- 11. STFT 微多普勒谱图 ----
        // 距离门平均: bestBin ±5
        int lo = std::max(0, bestBin - 5);
        int hi = std::min(validBins, bestBin + 6);
        std::vector<std::complex<float>> microCplx(nChirps);
        for (int i = 0; i < nChirps; ++i) {
            float sumReal = 0, sumImag = 0;
            for (int k = lo; k < hi; ++k) {
                sumReal += specAll[i][k].real();
                sumImag += specAll[i][k].imag();
            }
            microCplx[i] = std::complex<float>(
                sumReal / (hi - lo), sumImag / (hi - lo));
        }

        // STFT
        int hop = kStftNperseg - kStftNoverlap;  // 4
        int nTime = (nChirps - kStftNperseg) / hop + 1;
        if (nTime < 1) nTime = 1;
        // 存储为 freq-major: mspec[f * nTime + t]，与 Python Zd[freq][time] 一致
        std::vector<float> mspec(kStftFreqBins * nTime);
        for (int t = 0; t < nTime; ++t) {
            int start = t * hop;
            std::vector<std::complex<float>> frame(fft32_.size(), {0, 0});
            for (int s = 0; s < kStftNperseg; ++s) {
                frame[s] = microCplx[start + s] * kStftHann32[s];
            }
            fft32_.fft(frame.data());
            for (int f = 0; f < kStftFreqBins; ++f) {
                mspec[f * nTime + t] = std::abs(frame[f]);
            }
        }

        // Normalize
        float specMax = 0;
        for (float& v : mspec) specMax = std::max(specMax, v);
        if (specMax < 1e-10f) specMax = 1;
        for (float& v : mspec) v /= specMax;

        // per-frequency median subtraction 去除硬件固定频率干扰
        for (int f = 0; f < kStftFreqBins; ++f) {
            std::vector<float> rowVals(nTime);
            for (int t = 0; t < nTime; ++t) rowVals[t] = mspec[f * nTime + t];
            float fmed = median(rowVals);
            for (int t = 0; t < nTime; ++t) {
                float& v = mspec[f * nTime + t];
                v -= fmed;
                if (v < 0) v = 0;
            }
        }

        // 幂压缩 + 高斯平滑
        for (float& v : mspec) v = std::sqrt(v);
        mspec = gauss2D(mspec, kStftFreqBins, nTime);

        // ---- 12. 输出 (原始尺寸，不 resize) ----
        result.spectrogram = std::move(mspec);
        result.specRows = kStftFreqBins;
        result.specCols = nTime;

        result.envelope = std::move(ampSmooth);
        result.envLen = (int)result.envelope.size();

        // 多普勒统计
        result.maxDopplerHz = 0;
        for (float v : dopRaw) {
            result.maxDopplerHz = std::max(result.maxDopplerHz, std::abs(v));
        }

        return result;
    }

private:
    BluesteinFFT fft960_{960};
    R2FFT fft32_{32};
    BandpassFilter bp_;
};
