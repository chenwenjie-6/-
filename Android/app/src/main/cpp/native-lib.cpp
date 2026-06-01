#include <jni.h>
#include <android/log.h>
#include <aaudio/AAudio.h>

#include <cmath>
#include <vector>
#include <atomic>
#include <fstream>
#include <string>
#include <sstream>

#include "dsp.h"

#include <onnxruntime_c_api.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "UltraGesture", __VA_ARGS__)

// ======================================================
// 全局
// ======================================================

static AAudioStream *playStream = nullptr;
static AAudioStream *recordStream = nullptr;

static std::atomic<bool> isRunning(false);
static std::atomic<bool> isRecording(false);

// 参数由 constants.h 提供

// ======================================================
// 发射 chirp
// ======================================================

float txChirp[kChirpSamples];

// ======================================================
// PCM 缓冲（录制模式）
// ======================================================

std::vector<float> pcmBuffer;

// ======================================================
// 识别模式
// ======================================================

static std::atomic<bool> isRecognition(false);
static std::vector<float> recogBuffer;
static ProcessResult lastRecogResult;
static ChirpProcessor gProcessor;

constexpr int kRecogDuration = 3;
constexpr int kRecogSamples = kRecogDuration * kSampleRate;

// ======================================================
// ONNX Runtime 模型
// ======================================================

static const OrtApi* gOrtApi = nullptr;
static OrtEnv* gOrtEnv = nullptr;
static OrtSession* gOrtSession = nullptr;
static OrtMemoryInfo* gOrtMemInfo = nullptr;
static bool gModelLoaded = false;
static std::string gModelPath;

static const char* kInputNames[] = {"spectrogram", "envelope"};
static const char* kOutputNames[] = {"logits"};

static float minMaxNorm(std::vector<float>& v) {
    float vmin = v[0], vmax = v[0];
    for (float x : v) {
        if (x < vmin) vmin = x;
        if (x > vmax) vmax = x;
    }
    float denom = vmax - vmin + 1e-8f;
    for (float& x : v) x = (x - vmin) / denom;
    return denom;
}

// 居中裁剪/填充，匹配 Python pad_to (不插值，保留原始值)
// 2D: rows × curCols → rows × targetCols，沿时间轴居中操作
inline std::vector<float> padTo2D(const std::vector<float>& src,
                                   int rows, int curCols, int targetCols) {
    if (curCols == targetCols) return src;
    std::vector<float> dst(rows * targetCols, 0.0f);
    if (curCols < targetCols) {
        int padLeft = (targetCols - curCols) / 2;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < curCols; ++c)
                dst[r * targetCols + padLeft + c] = src[r * curCols + c];
    } else {
        int start = (curCols - targetCols) / 2;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < targetCols; ++c)
                dst[r * targetCols + c] = src[r * curCols + start + c];
    }
    return dst;
}

inline std::vector<float> padTo1D(const std::vector<float>& src, int targetLen) {
    int n = (int)src.size();
    if (n == targetLen) return src;
    std::vector<float> dst(targetLen, 0.0f);
    if (n < targetLen) {
        int padLeft = (targetLen - n) / 2;
        for (int i = 0; i < n; ++i) dst[padLeft + i] = src[i];
    } else {
        int start = (n - targetLen) / 2;
        for (int i = 0; i < targetLen; ++i) dst[i] = src[start + i];
    }
    return dst;
}

// ======================================================
// 保存路径
// ======================================================

std::string currentSavePath;
std::string recogSavePath;

// ======================================================
// 汉宁窗
// ======================================================

float hanning(int n, int N) {
    return 0.5f * (1.0f - cosf(2.0f * M_PI * n / (N - 1)));
}

// ======================================================
// 初始化 FMCW Chirp
// ======================================================

void initFMCW() {
    double k = (kEndFreq - kStartFreq) / kChirpTime;
    for (int i = 0; i < kChirpSamples; ++i) {
        double t = (double)i / kSampleRate;
        double phase = 2.0 * M_PI * (kStartFreq * t + 0.5 * k * t * t);
        float win = hanning(i, kChirpSamples);
        txChirp[i] = 0.9f * sinf(phase) * win;
    }
    LOGI("FMCW chirp 初始化完成");
}

// ======================================================
// 保存 PCM
// ======================================================

void savePCM() {
    std::string path = currentSavePath;
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        LOGI("PCM 文件打开失败");
        return;
    }
    out.write(reinterpret_cast<const char*>(pcmBuffer.data()),
              pcmBuffer.size() * sizeof(float));
    out.close();
    LOGI("PCM 已保存: %s", path.c_str());
    LOGI("样本数: %zu", pcmBuffer.size());
}

// ======================================================
// 播放回调
// ======================================================

aaudio_data_callback_result_t playCallback(
        AAudioStream *stream, void *userData,
        void *audioData, int32_t numFrames) {
    float *out = static_cast<float*>(audioData);
    static int phase = 0;
    for (int i = 0; i < numFrames; ++i) {
        out[i] = txChirp[phase];
        phase++;
        if (phase >= kChirpSamples) phase = 0;
    }
    return AAUDIO_CALLBACK_RESULT_CONTINUE;
}

// ======================================================
// 录音回调
// ======================================================

aaudio_data_callback_result_t recordCallback(
        AAudioStream *stream, void *userData,
        void *audioData, int32_t numFrames) {
    float *in = static_cast<float*>(audioData);
    if (isRecording) {
        if (isRecognition) {
            recogBuffer.insert(recogBuffer.end(), in, in + numFrames);
        } else {
            pcmBuffer.insert(pcmBuffer.end(), in, in + numFrames);
        }
    }
    return AAUDIO_CALLBACK_RESULT_CONTINUE;
}

// ======================================================
// JNI：启动引擎
// ======================================================

extern "C"
JNIEXPORT void JNICALL
Java_com_cwj_ultragesture_MainActivity_startEngine(
        JNIEnv *env, jobject thiz) {
    if (isRunning) return;
    initFMCW();
    isRunning = true;

    AAudioStreamBuilder *playBuilder;
    AAudio_createStreamBuilder(&playBuilder);
    AAudioStreamBuilder_setDirection(playBuilder, AAUDIO_DIRECTION_OUTPUT);
    AAudioStreamBuilder_setSampleRate(playBuilder, kSampleRate);
    AAudioStreamBuilder_setChannelCount(playBuilder, 1);
    AAudioStreamBuilder_setFormat(playBuilder, AAUDIO_FORMAT_PCM_FLOAT);
    AAudioStreamBuilder_setPerformanceMode(playBuilder, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    AAudioStreamBuilder_setDataCallback(playBuilder, playCallback, nullptr);
    AAudioStreamBuilder_openStream(playBuilder, &playStream);
    AAudioStreamBuilder_delete(playBuilder);

    AAudioStreamBuilder *recordBuilder;
    AAudio_createStreamBuilder(&recordBuilder);
    AAudioStreamBuilder_setDirection(recordBuilder, AAUDIO_DIRECTION_INPUT);
    AAudioStreamBuilder_setSampleRate(recordBuilder, kSampleRate);
    AAudioStreamBuilder_setChannelCount(recordBuilder, 1);
    AAudioStreamBuilder_setFormat(recordBuilder, AAUDIO_FORMAT_PCM_FLOAT);
    AAudioStreamBuilder_setPerformanceMode(recordBuilder, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    AAudioStreamBuilder_setDataCallback(recordBuilder, recordCallback, nullptr);
    AAudioStreamBuilder_openStream(recordBuilder, &recordStream);
    AAudioStreamBuilder_delete(recordBuilder);

    AAudioStream_requestStart(playStream);
    AAudioStream_requestStart(recordStream);
    LOGI("引擎启动成功");
}

// ======================================================
// JNI：停止引擎
// ======================================================

extern "C"
JNIEXPORT void JNICALL
Java_com_cwj_ultragesture_MainActivity_stopEngine(
        JNIEnv *env, jobject thiz) {
    isRunning = false;
    if (playStream) { AAudioStream_close(playStream); playStream = nullptr; }
    if (recordStream) { AAudioStream_close(recordStream); recordStream = nullptr; }
    LOGI("引擎已停止");
}

// ======================================================
// JNI：开始录制
// ======================================================

extern "C"
JNIEXPORT void JNICALL
Java_com_cwj_ultragesture_MainActivity_startRecording(
        JNIEnv *env, jobject thiz, jstring path) {
    const char *cpath = env->GetStringUTFChars(path, nullptr);
    currentSavePath = std::string(cpath);
    env->ReleaseStringUTFChars(path, cpath);
    pcmBuffer.clear();
    isRecognition = false;
    isRecording = true;
    LOGI("开始录制");
}

// ======================================================
// JNI：停止录制
// ======================================================

extern "C"
JNIEXPORT void JNICALL
Java_com_cwj_ultragesture_MainActivity_stopRecording(
        JNIEnv *env, jobject thiz) {
    isRecording = false;
    savePCM();
    LOGI("录制结束");
}

// ======================================================
// JNI：开始识别
// ======================================================

extern "C"
JNIEXPORT void JNICALL
Java_com_cwj_ultragesture_MainActivity_startRecognition(
        JNIEnv *env, jobject thiz, jstring path) {
    const char *cpath = env->GetStringUTFChars(path, nullptr);
    recogSavePath = std::string(cpath);
    env->ReleaseStringUTFChars(path, cpath);
    recogBuffer.clear();
    recogBuffer.reserve(kRecogSamples);
    lastRecogResult = ProcessResult{};
    isRecognition = true;
    isRecording = true;
    LOGI("开始识别，PCM保存路径: %s", recogSavePath.c_str());
}

// ======================================================
// JNI：停止识别并处理（先存PCM→读回→处理→ONNX推理）
// ======================================================

// 对 lastRecogResult 执行 ONNX 推理，追加分类字段到 JSON stream
static void appendClassifyJson(std::ostringstream& json) {
    if (!gModelLoaded ||
        lastRecogResult.spectrogram.empty() || lastRecogResult.envelope.empty())
        return;

    std::vector<float> spec = lastRecogResult.spectrogram;
    std::vector<float> envVec = lastRecogResult.envelope;

    spec = padTo2D(spec, lastRecogResult.specRows, lastRecogResult.specCols, kSpecTime);
    envVec = padTo1D(envVec, kEnvelopeLen);

    minMaxNorm(spec);
    minMaxNorm(envVec);

    // 诊断
    {
        float smin = spec[0], smax = spec[0], ssum = 0;
        for (float x : spec) {
            if (x < smin) smin = x;
            if (x > smax) smax = x;
            ssum += x;
        }
        float emin = envVec[0], emax = envVec[0], esum = 0;
        for (float x : envVec) {
            if (x < emin) emin = x;
            if (x > emax) emax = x;
            esum += x;
        }
        LOGI("spec[%d,%d]: min=%.4f max=%.4f sum=%.4f  env[%d]: min=%.4f max=%.4f sum=%.4f",
             lastRecogResult.specRows, lastRecogResult.specCols,
             smin, smax, ssum,
             (int)envVec.size(), emin, emax, esum);
    }

    int64_t specShape[] = {1, 1, kSpecFreq, kSpecTime};
    OrtValue* specTensor = nullptr;
    gOrtApi->CreateTensorWithDataAsOrtValue(
        gOrtMemInfo, spec.data(), spec.size() * sizeof(float),
        specShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &specTensor);

    int64_t envShape[] = {1, 1, kEnvelopeLen};
    OrtValue* envTensor = nullptr;
    gOrtApi->CreateTensorWithDataAsOrtValue(
        gOrtMemInfo, envVec.data(), envVec.size() * sizeof(float),
        envShape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &envTensor);

    const OrtValue* inputs[] = {specTensor, envTensor};
    OrtValue* outputs[1] = {nullptr};

    OrtStatus* s = gOrtApi->Run(gOrtSession, nullptr,
                          kInputNames,  inputs, 2,
                          kOutputNames, 1, outputs);

    if (s) {
        LOGI("推理失败: %s", gOrtApi->GetErrorMessage(s));
        gOrtApi->ReleaseStatus(s);
        gOrtApi->ReleaseValue(specTensor);
        gOrtApi->ReleaseValue(envTensor);
        return;
    }

    float* logits = nullptr;
    gOrtApi->GetTensorMutableData(outputs[0], (void**)&logits);

    LOGI("kNumClasses=%d  logits=[%.8f, %.8f, %.8f, %.8f, %.8f]",
         kNumClasses, logits[0], logits[1], logits[2], logits[3], logits[4]);

    float lmax = logits[0];
    for (int i = 1; i < kNumClasses; ++i) {
        if (logits[i] > lmax) lmax = logits[i];
    }
    float probs[8];
    float esum = 0;
    for (int i = 0; i < kNumClasses; ++i) {
        probs[i] = expf(logits[i] - lmax);
        esum += probs[i];
    }
    for (int i = 0; i < kNumClasses; ++i) {
        probs[i] /= esum;
    }

    int pred = 0;
    for (int i = 1; i < kNumClasses; ++i) {
        if (probs[i] > probs[pred]) pred = i;
    }

    const char* names[] = {"前推", "后拉", "横扫", "滑动", "碰拳"};

    json << ",\"class\":" << pred;
    json << ",\"name\":\"" << names[pred] << "\"";
    json << ",\"logits\":[";
    for (int i = 0; i < kNumClasses; ++i) {
        if (i > 0) json << ",";
        json << logits[i];
    }
    json << "],\"probs\":[";
    for (int i = 0; i < kNumClasses; ++i) {
        if (i > 0) json << ",";
        json << probs[i];
    }
    json << "]";

    LOGI("推理: %s", names[pred]);

    gOrtApi->ReleaseValue(specTensor);
    gOrtApi->ReleaseValue(envTensor);
    gOrtApi->ReleaseValue(outputs[0]);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_cwj_ultragesture_MainActivity_stopRecognition(
        JNIEnv *env, jobject thiz) {
    isRecording = false;
    isRecognition = false;

    LOGI("识别停止，采集 %zu 个采样", recogBuffer.size());

    // 1. 保存 PCM 到文件
    if (!recogSavePath.empty() && !recogBuffer.empty()) {
        std::ofstream out(recogSavePath, std::ios::binary);
        out.write(reinterpret_cast<const char*>(recogBuffer.data()),
                  recogBuffer.size() * sizeof(float));
        out.close();
        LOGI("PCM 已保存: %s (%zu samples)", recogSavePath.c_str(), recogBuffer.size());
    }

    // 2. 从文件读回（对齐 predict.py：文件 → 处理）
    std::vector<float> pcmData;
    if (!recogSavePath.empty()) {
        std::ifstream in(recogSavePath, std::ios::binary);
        in.seekg(0, std::ios::end);
        size_t fileSize = in.tellg();
        in.seekg(0, std::ios::beg);
        pcmData.resize(fileSize / sizeof(float));
        in.read(reinterpret_cast<char*>(pcmData.data()), fileSize);
        in.close();
    } else {
        // fallback: 直接从 buffer 处理
        pcmData = std::move(recogBuffer);
    }

    // 3. 信号处理
    lastRecogResult = gProcessor.process(pcmData.data(), (int)pcmData.size());

    // 4. 构建 JSON（信号元数据 + 分类结果）
    std::ostringstream json;
    json << "{";
    json << "\"nChirps\":" << lastRecogResult.nChirps << ",";
    json << "\"bestRangeCm\":" << lastRecogResult.bestRangeCm << ",";
    json << "\"maxDopplerHz\":" << lastRecogResult.maxDopplerHz << ",";
    json << "\"specRows\":" << lastRecogResult.specRows << ",";
    json << "\"specCols\":" << lastRecogResult.specCols << ",";
    json << "\"envLen\":" << lastRecogResult.envLen;

    // ONNX 推理（追加 class, name, logits, probs）
    appendClassifyJson(json);

    json << "}";

    LOGI("处理完成: chirps=%d, range=%.1fcm, doppler=%.1fHz",
         lastRecogResult.nChirps,
         lastRecogResult.bestRangeCm,
         lastRecogResult.maxDopplerHz);

    return env->NewStringUTF(json.str().c_str());
}

// ======================================================
// JNI：从 PCM 文件处理+识别（用于离线识别已存文件）
// ======================================================

extern "C"
JNIEXPORT jstring JNICALL
Java_com_cwj_ultragesture_MainActivity_processPcmFile(
        JNIEnv *env, jobject thiz, jstring pcmPath) {
    const char *cpath = env->GetStringUTFChars(pcmPath, nullptr);

    // 读文件
    std::ifstream in(cpath, std::ios::binary);
    if (!in.is_open()) {
        env->ReleaseStringUTFChars(pcmPath, cpath);
        return env->NewStringUTF("{\"error\":\"无法打开文件\"}");
    }
    in.seekg(0, std::ios::end);
    size_t fileSize = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<float> pcmData(fileSize / sizeof(float));
    in.read(reinterpret_cast<char*>(pcmData.data()), fileSize);
    in.close();
    env->ReleaseStringUTFChars(pcmPath, cpath);

    LOGI("processPcmFile: %zu samples", pcmData.size());

    // 信号处理
    lastRecogResult = gProcessor.process(pcmData.data(), (int)pcmData.size());

    // 构建 JSON
    std::ostringstream json;
    json << "{";
    json << "\"nChirps\":" << lastRecogResult.nChirps << ",";
    json << "\"bestRangeCm\":" << lastRecogResult.bestRangeCm << ",";
    json << "\"maxDopplerHz\":" << lastRecogResult.maxDopplerHz << ",";
    json << "\"specRows\":" << lastRecogResult.specRows << ",";
    json << "\"specCols\":" << lastRecogResult.specCols << ",";
    json << "\"envLen\":" << lastRecogResult.envLen;

    appendClassifyJson(json);

    json << "}";
    return env->NewStringUTF(json.str().c_str());
}

// ======================================================
// JNI：获取谱图数据
// ======================================================

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_cwj_ultragesture_MainActivity_getSpectrogram(
        JNIEnv *env, jobject thiz) {
    int size = (int)lastRecogResult.spectrogram.size();
    jfloatArray arr = env->NewFloatArray(size);
    env->SetFloatArrayRegion(arr, 0, size, lastRecogResult.spectrogram.data());
    return arr;
}

// ======================================================
// JNI：获取包络数据
// ======================================================

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_cwj_ultragesture_MainActivity_getEnvelope(
        JNIEnv *env, jobject thiz) {
    int size = (int)lastRecogResult.envelope.size();
    jfloatArray arr = env->NewFloatArray(size);
    env->SetFloatArrayRegion(arr, 0, size, lastRecogResult.envelope.data());
    return arr;
}

// ======================================================
// JNI：加载模型
// ======================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_cwj_ultragesture_MainActivity_loadModel(
        JNIEnv *env, jobject thiz, jstring modelPath) {
    const char *cpath = env->GetStringUTFChars(modelPath, nullptr);

    // 获取 API 入口（仅一次）
    if (!gOrtApi) {
        gOrtApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    }

    // 创建 env（全局，仅一次）
    if (!gOrtEnv) {
        OrtStatus* s = gOrtApi->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "gesture", &gOrtEnv);
        if (s) {
            LOGI("OrtCreateEnv 失败: %s", gOrtApi->GetErrorMessage(s));
            gOrtApi->ReleaseStatus(s);
            env->ReleaseStringUTFChars(modelPath, cpath);
            return JNI_FALSE;
        }
        gOrtApi->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &gOrtMemInfo);
    }

    // 销毁旧 session
    if (gOrtSession) {
        gOrtApi->ReleaseSession(gOrtSession);
        gOrtSession = nullptr;
    }

    OrtSessionOptions* opts = nullptr;
    gOrtApi->CreateSessionOptions(&opts);
    gOrtApi->SetSessionGraphOptimizationLevel(opts, ORT_DISABLE_ALL);
    OrtStatus* s = gOrtApi->CreateSession(gOrtEnv, cpath, opts, &gOrtSession);
    gOrtApi->ReleaseSessionOptions(opts);

    if (s) {
        LOGI("模型加载失败: %s", gOrtApi->GetErrorMessage(s));
        gOrtApi->ReleaseStatus(s);
        env->ReleaseStringUTFChars(modelPath, cpath);
        return JNI_FALSE;
    }

    gModelPath = std::string(cpath);
    env->ReleaseStringUTFChars(modelPath, cpath);
    gModelLoaded = true;
    LOGI("模型加载成功");
    return JNI_TRUE;
}

