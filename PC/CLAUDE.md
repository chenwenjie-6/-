# UltraGesture — 超声手势识别毕设

**身份**: 本科生毕设项目，我是项目唯一开发者。
**目标**: 利用 17-23kHz FMCW 超声实现空中手势识别（精度 → 特征 → 模型 → Android 部署）。
**沟通**: 直接执行，带简短解释。中文优先。不自作聪明加"自适应"逻辑，按用户明确要求改。

## 项目双仓库

| 仓库 | 路径 | 作用 |
|------|------|------|
| **Android 采集端** | `D:\AndroidProjects\UltraGesture` | 单 Activity + 单 native-lib.cpp，AAudio 低延迟音频管线 |
| **PC 处理端** | `C:\Users\86132\Documents\手势毕设` | Python 信号处理，当前工作目录 |

## PC 端核心文件

- `process_pcm.py` — V4 数据集模式，默认 adb pull + 处理 → dataset/
- `merge_spectrograms.py` — 扫描 dataset/ 按手势拼接对比大图
- `train_cnn.py` — 双流CNN训练，自动加载 dataset/ 下所有 npy
- `export_onnx.py` — 将 TorchScript 模型导出为 ONNX 格式供 Android 端使用
- `gen_constants.py` — 从 Python 参数生成 C++ constants.h 和 biquad_coeffs.h
- `RECORD.md` — 论文工作节点记录（里程碑、参数、放弃方案、验证结果）
- `dataset/` — 7种手势数据，每个含 `pcm/` `幅度包络图/` `微多普勒谱图/` `npy/`

## 数据集模式流程

```
python process_pcm.py                 # adb pull 手机数据 → 处理全部手势
python process_pcm.py --gesture Push  # 只处理 Push
python process_pcm.py --no-pull       # 跳过 adb pull
python process_pcm.py --legacy        # 旧模式 (output/ 子目录)
```

输出结构（平铺）：
- `dataset/Push/幅度包络图/Push_001.png`
- `dataset/Push/微多普勒谱图/Push_001.png`
- `dataset/Push/npy/Push_001_spectrogram.npy` + `Push_001_envelope.npy`

## merge 脚本

```bash
python merge_spectrograms.py          # 全部手势 → merged/
python merge_spectrograms.py Push     # 单个手势
```

每个手势出两张拼接图，输出到 `merged/` 目录。

## 输出图表

| 图表 | 内容 | 用途 |
|------|------|------|
| 幅度包络图 | SG平滑幅度曲线 | 区分Push/Pull等纵向手势 |
| 微多普勒谱图 | STFT幅度谱，Y轴[-0.2, -1.4]Hz | 区分不同运动模式 |
| 03_幅度相位多普勒.png | 3面板诊断图 | 仅旧模式(legacy)生成 |

## 关键参数

- FS=48000Hz, Chirp=17-23kHz 持续20ms(960采样), stride=960(无重叠)
- 慢时间=50Hz(Nyquist=25Hz), 距离分辨率≈2.86cm
- STFT: 32点Hann窗, 28重叠, gamma=0.5幂压, gaussian sigma=0.6
- 幅度SG平滑: 25帧窗口(0.5s), 3阶多项式
- 多普勒谱图Y轴: [-0.2, -1.4]Hz (下边界-0.2, 上边界-1.4)

## Android 端核心参数

- 手势集6种: Push(前推), Pull(后拉), Sweep(横扫), Slide(滑动), Fist_bump(碰拳), Grab(抓取)
- 采集: 按下按钮→3s自动停止，固定长度。起始编号可自定义。
- PCM存储: `pcm_data/<手势名>/<GestureName>_<NNN>.pcm`, float32 raw PCM
- 关键源文件: `MainActivity.java`(所有Java逻辑), `native-lib.cpp`(AAudio+FMCW+JNI)
- `kChirpTime=0.02`, `kSampleRate=48000`, 100%占空比

## Android C++ 文件结构

| 文件 | 作用 |
|------|------|
| `native-lib.cpp` | JNI入口，引擎+录制+识别三种模式，ONNX Runtime CNN推理 |
| `dsp.h` | ChirpProcessor 完整信号管线（header-only） |
| `biquad.h` | Direct Form I 级联滤波器，预定义 BandpassFilter/LowpassFilter |
| `fft.h` | 轻量 radix-2 FFT/IFFT |
| `biquad_coeffs.h` | gen_constants.py 生成，6阶带通+6阶低通系数 |
| `constants.h` | gen_constants.py 生成，参考chirp、窗函数、SG核、高斯核等 |
| `CMakeLists.txt` | 单源文件编译，链接 onnxruntime + log + aaudio |

识别模式流程: 按下"开始识别"→3秒录音→ChirpProcessor.process()→谱图(32×N)+包络(M)→padTo2D/padTo1D(居中裁剪)→minMaxNorm→ONNX Runtime推理→JNI返回分类结果+概率

## PC→Android 信号管线对照

| 步骤 | Python (process_pcm.py) | C++ (dsp.h) |
|------|------------------------|-------------|
| 带通滤波 | filtfilt(b_bp, a_bp) | BandpassFilter.process() |
| 互相关 | np.correlate(rx, ref_real) | findChirpOffset() |
| 静态背景 | np.median(chirps, axis=0) | 逐bin中值 |
| 去斜 | chirps * conj(ref_cplx) | 复数乘法 per sample |
| 低通 | sosfiltfilt(sos_lp) | LowpassFilter.process() |
| 距离FFT | np.fft.fft(960点) | R2FFT(1024点, 零填充) |
| 最强距离门 | np.argmax(energy) | 循环求最大值 |
| 多普勒 | np.angle * conj | atan2 相位差分 |
| SG平滑 | savgol_filter | sgSmooth() 用预计算核 |
| STFT | signal.stft | 手写32点STFT |
| 幂压+高斯 | power(0.5)+gaussian_filter | sqrt+gauss2D() |
| resize | pad_to | padTo2D/padTo1D (居中裁剪填充，不插值) |

## 手势动作定义

- Push(前推): 手掌靠近手机
- Pull(后拉): 手掌远离手机
- Sweep(横扫): 手掌平移从左到右再从右到左
- Slide(滑动): 手掌单向滑过
- Fist_bump(碰拳): 拳头靠近手机再远离
- Grab(抓取): 手掌从张开到握拳
- Tap(敲击): 拳头从上到下再从下到上

采集姿态: 手机平放桌边，屏幕朝上，手在手机底部(充电口)方向做动作，距离10-30cm。

## 当前阶段

**阶段5: 论文撰写** (2026-05-31)

论文仓库: `D:\桌面\毕业论文声学手势`，LaTeX (oucart.cls 模板)。

### 论文各章完成状态

| 章节 | 文件 | 状态 |
|------|------|------|
| 第1章 引言 | section_01.tex | ✅ 已完成 |
| 第2章 理论与技术 | section_02.tex | ✅ 已完成 |
| 第3章 系统设计与实现 | section_03.tex | ✅ 已完成（五层架构+三组图片） |
| 第4章 实验与结果分析 | section_04.tex | ✅ 已完成（3组实验+3张混淆矩阵图） |
| 第5章 结论与展望 | section_05.tex | ✅ 已完成 |

### 论文图片

| 位置 | 图片 | 说明 |
|------|------|------|
| 3.3.2 | 对比图.png | 数据增强效果 |
| 3.3.3 | Push/Pull/Sweep/Slide/Fist_bump_025_doppler.png | 五类手势微多普勒谱图 |
| 3.3.3 | Push/Pull/Sweep/Slide/Fist_bump_025_envelope.png | 五类手势幅度包络图 |
| 3.4 | 待画 | 双流CNN结构图 |
| 3.5.3 | 待画 | C++处理流程图 |
| 4.3 | cm_indoor/outdoor/outdoor_multi.png | 三组混淆矩阵 |

### 参考文献

20篇（5篇外文经CrossRef DOI验证 + 15篇中文知网真文献）。

### Android App 改动 (2026-05-31)

- app_name → "手势识别"
- 所有按钮放大（height 50dp, textSize 22sp/20sp）
- 识别结果仅显示手势名称（28sp），隐藏详细信息
- 谱图和包络图 `visibility="gone"`
- 手势下拉框改为中文
- C++ 返回中文手势名

## 实验数据

- 5类手势: 前推/后拉/横扫/滑动/碰拳，每类50样本，共250
- 5折CV: 99.60%（±0.89%）
- 室内安静: 98.4%（50样本）
- 室外嘈杂: 96.4%（50样本）
- 室外多人: 95.6%（150样本，3人）

手势集: Push×50, Pull×50, Sweep×50, Slide×50, Fist_bump×50, Grab×50(待处理)
模型: `models/model.onnx`(ONNX导出), `app/src/main/assets/model.onnx`(Android assets)

## CNN 训练脚本

- `train_cnn.py` — 双流CNN (Conv2d谱图 + Conv1d包络 → 拼接 → FC分类)
- 数据增强: 高斯噪声、时间平移、幅度缩放
- 5折分层交叉验证
- 用法: `python train_cnn.py`

## 历史决策记录（避免重复尝试）

- 试过方案A(60%占空比)→效果不好，已回退100%占空比
- 试过V4(50%重叠→100Hz慢时间)→效果不好，已回退50Hz
- 试过图6(STFT帧间相位差)→噪声太大，已删除
- 试过自适应多普勒Y轴→用户不满，改为固定值[-0.2, -1.4]Hz
- 画圈手势多普勒信号弱，已放弃
- 试过多进程/多线程加速→Windows spawn开销大/线程GIL争用，不如单线程
- DPI从150降到100，plt.style.use("fast")试过→效果不明显已回退

## 需要避免

- 不自作聪明加"自适应"或"自动"逻辑
- 不新建不必要的文档或 README
- Android端改了代码要提醒用户重新编译安装
- 旧PCM与新参数不兼容时要提醒清空重录
