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
- `export_onnx.py` — 将训练权重导出为 ONNX 格式供 Android 端使用
- `gen_constants.py` — 从 Python 参数生成 C++ constants.h 和 biquad_coeffs.h
- `RECORD.md` — 论文工作节点记录（里程碑、参数、放弃方案、验证结果）
- `dataset/` — 5种手势数据（Push/Pull/Sweep/Slide/Fist_bump），每个含 `pcm/` `幅度包络图/` `微多普勒谱图/` `npy/`

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

- 手势集7种: Push(前推), Pull(后拉), Sweep(横扫), Slide(滑动), Fist_bump(碰拳), Grab(抓取), Tap(敲击)
- 采集: 按下按钮→2s自动停止，固定长度。起始编号可自定义。
- 识别: 按下开始识别→3s录音→ChirpProcessor→ONNX推理→显示结果+概率
- PCM存储: `pcm_data/<手势名>/<GestureName>_<NNN>.pcm`, float32 raw PCM
- 关键源文件: `MainActivity.java`(所有Java逻辑), `native-lib.cpp`(AAudio+FMCW+JNI+ONNX)
- `kChirpTime=0.02`, `kSampleRate=48000`, 100%占空比, `kRecogDuration=3`

## Android C++ 文件结构

| 文件 | 作用 |
|------|------|
| `native-lib.cpp` | JNI入口，引擎+录制+识别三种模式，ONNX Runtime CNN推理 |
| `dsp.h` | ChirpProcessor 完整信号管线（header-only），12步处理 |
| `biquad.h` | Direct Form I 级联滤波器，含 filtfilt(odd extension 匹配 scipy) |
| `fft.h` | 轻量 radix-2 FFT/IFFT + Bluestein 任意长度 FFT |
| `biquad_coeffs.h` | gen_constants.py 生成，6阶带通+6阶低通系数 (SOS) |
| `constants.h` | gen_constants.py 生成，参考chirp、窗函数、SG核、高斯核等 |
| `CMakeLists.txt` | 单源文件编译，链接 onnxruntime + log + aaudio |

识别模式流程: 按下"开始识别"→3秒录音→ChirpProcessor.process()→谱图(32×26)+包络(100)→ONNX Runtime推理→JNI返回分类结果+概率

## PC→Android 信号管线对照

| 步骤 | Python (process_pcm.py) | C++ (dsp.h) |
|------|------------------------|-------------|
| 带通滤波 | filtfilt(b_bp, a_bp) | BandpassFilter.processFiltFilt() |
| 互相关 | np.correlate(rx, ref_real) | findChirpOffset() |
| 静态背景 | np.median(chirps, axis=0) | 逐bin中值 |
| 去斜 | chirps * conj(ref_cplx) | 复数乘法 per sample |
| 低通 | sosfiltfilt(sos_lp) | LowpassFilter.processFiltFilt() (odd extension, 匹配 scipy) |
| 距离FFT | np.fft.fft(960点) | Bluestein(960点) |
| 最强距离门 | np.argmax(energy), r<=1.0m | 循环求最大值, kValidRangeBins=35 |
| 多普勒 | np.angle * conj | atan2 相位差分 |
| SG平滑 | savgol_filter(window=25, polyorder=3) | sgSmooth() 预计算核 (边界模式有微弱差异) |
| STFT | signal.stft(32Hann, 28重叠, padded=False) | 手写32点STFT |
| 幂压+高斯 | power(0.5)+gaussian_filter(sigma=0.6) | sqrt+gauss2D() |
| pad_to | pad_to (居中填充/裁剪) | padTo2D/padTo1D (居中填充/裁剪) |

### 管线对齐状态 (2026-05-26)

filtfilt 边界处理已对齐 (biquad.h processFiltFilt 增加 odd extension, pad=3×n_sections)。
频谱图形状一致 (32×nTime), CNN 输入一致 (32×26 spectrogram + 100 envelope)。
剩余 float32/float64 精度差 ~1e-4, Bluestein vs np.fft 数值差 ~1e-5, SG 边界差 12/100 帧。
以上对 CNN 分类无实际影响，但无法做到逐像素完全一致。

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

**阶段4: Android实时推理移植** (2026-05-24)

C++信号处理管线已完成并多次对齐修正。双模式UI（录制/识别）已验证。
ONNX Runtime CNN推理已集成，Android端可实时识别手势并显示概率。

已采集: Push×50, Pull×50, Sweep×50, Slide×50, Fist_bump×50。Grab/Tap 待采集。
训练结果: 3分类 100%, 4分类 100%, 5分类 99.60%。

模型: `models/Pushvs_Pullvs_Sweepvs_Slide.pth`(4分类权重), `models/Pushvs_Pullvs_Sweepvs_Slidevs_Fist_bump.pth`(5分类权重), `models/model.onnx`(当前ONNX导出), `app/src/main/assets/model.onnx`(Android assets)

已知问题:
- ONNX Runtime NEON SIMD Gemm bug: 4/5分类模型在 Android 上最后1-2通道恒为0。3分类正常。已排除权重/bias/特征全零/数据泄漏等原因，确认为 ORT 1.21.0 ARM NEON 特定版本 bug。
- ORT 1.26.0 升级尝试崩溃，已回退。

## CNN 训练脚本

- `train_cnn.py` — 双流CNN (Conv2d谱图 + Conv1d包络 → 拼接 → FC分类)
- 数据增强: 高斯噪声、时间平移、幅度缩放
- 5折分层交叉验证 + 标签打乱对照实验
- `GESTURES` 列表决定分类数，模型名自动拼接
- 用法: 修改 `GESTURES` 列表后 `python train_cnn.py`
- 训练后: `python export_onnx.py` 导出 ONNX → 复制到 Android assets/

## 历史决策记录（避免重复尝试）

- 试过方案A(60%占空比)→效果不好，已回退100%占空比
- 试过V4(50%重叠→100Hz慢时间)→效果不好，已回退50Hz
- 试过图6(STFT帧间相位差)→噪声太大，已删除
- 试过自适应多普勒Y轴→用户不满，改为固定值[-0.2, -1.4]Hz
- 画圈手势多普勒信号弱，已放弃
- 试过多进程/多线程加速→Windows spawn开销大/线程GIL争用，不如单线程
- DPI从150降到100，plt.style.use("fast")试过→效果不明显已回退
- 试过LibTorch Android AAR→无法下载，改用 ONNX Runtime C API
- 试过 ORT 1.26.0 升级→崩溃，保留 1.21.0

## 需要避免

- 不自作聪明加"自适应"或"自动"逻辑
- 不新建不必要的文档或 README
- Android端改了 C++ 代码要提醒用户重新编译安装
- 旧PCM与新参数不兼容时要提醒清空重录
- 重大修改询问用户
- gen_constants.py 生成的 constants.h 参数有变时要同步更新 Android 端
