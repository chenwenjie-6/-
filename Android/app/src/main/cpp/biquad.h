#pragma once
#include <cstring>

#include "biquad_coeffs.h"

// 单个 Direct Form I Biquad 节
// y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
struct BiquadStage {
    float b0, b1, b2;
    float a1, a2;
    float x1, x2;  // x[n-1], x[n-2]
    float y1, y2;  // y[n-1], y[n-2]

    void init(const BiquadSection& sec) {
        b0 = sec.b0; b1 = sec.b1; b2 = sec.b2;
        a1 = sec.a1; a2 = sec.a2;
        x1 = 0; x2 = 0;
        y1 = 0; y2 = 0;
    }

    void reset() {
        x1 = 0; x2 = 0;
        y1 = 0; y2 = 0;
    }

    float process(float x) {
        float y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1; x1 = x;
        y2 = y1; y1 = y;
        return y;
    }
};

// 级联多个 Biquad 节构成高阶滤波器
template <int N>
struct BiquadCascade {
    BiquadStage stages[N];

    void init(const BiquadSection* sections) {
        for (int i = 0; i < N; ++i) {
            stages[i].init(sections[i]);
        }
    }

    void reset() {
        for (int i = 0; i < N; ++i) {
            stages[i].reset();
        }
    }

    float process(float x) {
        for (int i = 0; i < N; ++i) {
            x = stages[i].process(x);
        }
        return x;
    }

    void processBlock(const float* input, float* output, int numSamples) {
        for (int i = 0; i < numSamples; ++i) {
            output[i] = process(input[i]);
        }
    }

    // filtfilt: 双向滤波，零相位 (匹配 Python scipy.signal.filtfilt)
    // scipy edge_pad = 3 * n_sections, 用 odd extension 消除启动瞬态
    void processFiltFilt(const float* input, float* output, int numSamples) {
        constexpr int pad = N * 3;  // scipy edge_pad for SOS filtfilt
        int paddedLen = numSamples + 2 * pad;
        std::vector<float> padded(paddedLen);

        // Odd extension left: x[-k] = 2*x[0] - x[k]
        for (int i = 0; i < pad; ++i)
            padded[i] = 2.0f * input[0] - input[pad - i];
        // Middle
        std::memcpy(padded.data() + pad, input, numSamples * sizeof(float));
        // Odd extension right: x[N+k] = 2*x[N-1] - x[N-1-k]
        for (int i = 0; i < pad; ++i)
            padded[pad + numSamples + i] = 2.0f * input[numSamples - 1] - input[numSamples - 2 - i];

        // 1. 正向滤波
        processBlock(padded.data(), padded.data(), paddedLen);
        // 2. 反转
        for (int i = 0; i < paddedLen / 2; ++i) {
            float tmp = padded[i];
            padded[i] = padded[paddedLen - 1 - i];
            padded[paddedLen - 1 - i] = tmp;
        }
        // 3. 重置 + 正向滤波
        reset();
        processBlock(padded.data(), padded.data(), paddedLen);
        // 4. 反转回原来顺序
        for (int i = 0; i < paddedLen / 2; ++i) {
            float tmp = padded[i];
            padded[i] = padded[paddedLen - 1 - i];
            padded[paddedLen - 1 - i] = tmp;
        }

        std::memcpy(output, padded.data() + pad, numSamples * sizeof(float));
    }
};

// 预定义滤波器类型
using BandpassFilter = BiquadCascade<kSosBandpassCount>;
using LowpassFilter  = BiquadCascade<kSosLowpassCount>;
