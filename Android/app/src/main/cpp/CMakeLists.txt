cmake_minimum_required(VERSION 3.18.1)
project("ultragesture")

# ======================================================
# ONNX Runtime 配置
# 下载: onnxruntime-android-1.26.0.aar (Maven Central)
# 解压 .aar 到 D:/AndroidProjects/onnxruntime-android/
# ======================================================
set(ORT_DIR "D:/AndroidProjects/onnxruntime-android"
    CACHE PATH "ONNX Runtime Android distribution root")

include_directories(${ORT_DIR}/headers)
link_directories(${ORT_DIR}/jni/arm64-v8a)

# ======================================================
# 源文件
# ======================================================
add_library(ultragesture SHARED native-lib.cpp)

find_library(log-lib log)
find_library(aaudio-lib aaudio)

target_link_libraries(ultragesture
    onnxruntime
    ${log-lib}
    ${aaudio-lib}
)
