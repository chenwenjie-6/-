#pragma once

#include <cmath>
#include <complex>
#include <vector>

// 轻量 radix-2 DIT FFT，用于 power-of-2 长度
class R2FFT {
public:
    explicit R2FFT(int n) : N_(n) {
        // 预计算旋转因子
        twiddles_.resize(N_ / 2);
        for (int i = 0; i < N_ / 2; ++i) {
            float angle = -2.0f * M_PI * i / N_;
            twiddles_[i] = std::complex<float>(cosf(angle), sinf(angle));
        }
        // 预计算 bit-reversal 表
        rev_.resize(N_);
        int bits = 0;
        while ((1 << bits) < N_) bits++;
        for (int i = 0; i < N_; ++i) {
            int r = 0;
            for (int b = 0; b < bits; ++b) {
                if (i & (1 << b)) r |= (1 << (bits - 1 - b));
            }
            rev_[i] = r;
        }
    }

    void fft(std::complex<float>* data) {
        // bit-reversal permutation
        for (int i = 0; i < N_; ++i) {
            if (i < rev_[i]) {
                std::swap(data[i], data[rev_[i]]);
            }
        }
        // Cooley-Tukey butterfly
        for (int len = 2; len <= N_; len <<= 1) {
            int half = len >> 1;
            int step = N_ / len;
            for (int i = 0; i < N_; i += len) {
                for (int j = 0; j < half; ++j) {
                    std::complex<float> t = twiddles_[j * step] * data[i + j + half];
                    data[i + j + half] = data[i + j] - t;
                    data[i + j] += t;
                }
            }
        }
    }

    void ifft(std::complex<float>* data) {
        // conjugate input
        for (int i = 0; i < N_; ++i) data[i] = std::conj(data[i]);
        fft(data);
        // conjugate + scale
        for (int i = 0; i < N_; ++i) data[i] = std::conj(data[i]) / (float)N_;
    }

    int size() const { return N_; }

private:
    int N_;
    std::vector<std::complex<float>> twiddles_;
    std::vector<int> rev_;
};

// Bluestein (chirp Z-transform) FFT — 支持任意长度 N
// 内部通过 M≥2N-1 的 radix-2 FFT 做卷积实现
class BluesteinFFT {
public:
    explicit BluesteinFFT(int n) : N_(n) {
        // M = 下一个 2 的幂，≥ 2N-1
        M_ = 1;
        while (M_ < 2 * N_ - 1) M_ <<= 1;

        fftM_ = new R2FFT(M_);

        // chirp[n] = exp(-j*pi*n²/N)
        chirp_.resize(N_);
        for (int i = 0; i < N_; ++i) {
            float angle = -M_PI * i * i / N_;
            chirp_[i] = std::complex<float>(cosf(angle), sinf(angle));
        }

        // 卷积核 b[m] = exp(j*pi*m²/N), 填满 M 长
        b_.resize(M_, {0, 0});
        b_[0] = {1, 0};
        for (int i = 1; i < N_; ++i) {
            float angle = M_PI * i * i / N_;
            b_[i] = std::complex<float>(cosf(angle), sinf(angle));
            b_[M_ - i] = b_[i];
        }
        // 预计算 B = FFT(b)，避免每次重复
        fftM_->fft(b_.data());
    }

    ~BluesteinFFT() { delete fftM_; }

    void fft(std::complex<float>* data) {
        // a = x[n] * chirp[n], 零填充到 M
        std::vector<std::complex<float>> a(M_, {0, 0});
        for (int i = 0; i < N_; ++i) {
            a[i] = data[i] * chirp_[i];
        }

        fftM_->fft(a.data());
        for (int i = 0; i < M_; ++i) a[i] *= b_[i];
        fftM_->ifft(a.data());

        // 后乘 chirp 得到 X[k]
        for (int k = 0; k < N_; ++k) {
            data[k] = a[k] * chirp_[k];
        }
    }

    int size() const { return N_; }

private:
    int N_;
    int M_;
    R2FFT* fftM_;
    std::vector<std::complex<float>> chirp_;
    std::vector<std::complex<float>> b_;
};
