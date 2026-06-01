package com.cwj.ultragesture;

import android.Manifest;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private boolean engineActive = false;
    private boolean isCollecting = false;
    private boolean isRecognizing = false;
    private boolean recordMode = true;
    private boolean modelReady = false;
    private String currentGesture = "Push";
    private String modelPath = "";
    private Handler handler = new Handler(Looper.getMainLooper());
    private Runnable autoStopRunnable;

    private static final int RECORD_REQUEST_CODE = 100;
    private static final int AUTO_STOP_MS = 3000;
    private static final int SPEC_FREQ = 32;
    private static final double kSlowFS = 50.0;
    private static final int kRecogDuration = 3;
    private int specCols = 26;    // 从 JNI 动态读取
    private int envLen = 100;     // 从 JNI 动态读取

    private static final String[] GESTURES = {
            "前推", "后拉", "横扫", "滑动", "碰拳"
    };

    private static final Map<String, String> GESTURE_EN = new HashMap<String, String>() {{
        put("前推", "Push");
        put("后拉", "Pull");
        put("横扫", "Sweep");
        put("滑动", "Slide");
        put("碰拳", "Fist_bump");
    }};

    private static final Map<String, String> GESTURE_CN = new HashMap<String, String>() {{
        put("Push", "前推");
        put("Pull", "后拉");
        put("Sweep", "横扫");
        put("Slide", "滑动");
        put("Fist_bump", "碰拳");
    }};

    private LinearLayout recordArea, recogArea;
    private Button recordModeBtn, recogModeBtn, recogBtn, collectBtn;
    private TextView recogResultText, recogInfoText, menuButton;
    private ImageView gestureImageView;
    private ImageView spectrogramView, envelopeView;

    static { System.loadLibrary("ultragesture"); }

    public native void startEngine();
    public native void stopEngine();
    public native void startRecording(String path);
    public native void stopRecording();
    public native void startRecognition(String path);
    public native String stopRecognition();
    public native float[] getSpectrogram();
    public native float[] getEnvelope();
    public native boolean loadModel(String modelPath);
    public native String processPcmFile(String pcmPath);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        recordArea = findViewById(R.id.recordArea);
        recogArea = findViewById(R.id.recogArea);
        recordModeBtn = findViewById(R.id.recordModeBtn);
        recogModeBtn = findViewById(R.id.recogModeBtn);
        recogBtn = findViewById(R.id.recogButton);
        collectBtn = findViewById(R.id.collectButton);
        recogResultText = findViewById(R.id.recogResultText);
        recogInfoText = findViewById(R.id.recogInfoText);
        gestureImageView = findViewById(R.id.gestureImageView);

        // 菜单按钮
        menuButton = findViewById(R.id.menuButton);
        menuButton.setOnClickListener(v -> showInfoDialog());
        spectrogramView = findViewById(R.id.spectrogramView);
        envelopeView = findViewById(R.id.envelopeView);

        EditText startNumberInput = findViewById(R.id.startNumberInput);

        // ---- 准备模型文件 ----
        prepareModelFile();

        // ---- 手势下拉框 ----
        Spinner gestureSpinner = findViewById(R.id.gestureSpinner);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_item, GESTURES);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        gestureSpinner.setAdapter(adapter);
        gestureSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
                currentGesture = GESTURES[pos];
                updateCounterDisplay();
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });

        // ---- 模式切换 ----
        recordModeBtn.setOnClickListener(v -> switchMode(true));
        recogModeBtn.setOnClickListener(v -> switchMode(false));

        // ---- 采集按钮 ----
        collectBtn.setOnClickListener(v -> {
            if (!engineActive) {
                Toast.makeText(this, "请先启动扬声器", Toast.LENGTH_SHORT).show();
                return;
            }
            if (!isCollecting) {
                startCollecting();
                collectBtn.setText("正在采集");
            }
        });

        // ---- 识别按钮 ----
        recogBtn.setOnClickListener(v -> {
            if (!engineActive) {
                Toast.makeText(this, "请先启动扬声器", Toast.LENGTH_SHORT).show();
                return;
            }
            if (!modelReady) {
                Toast.makeText(this, "模型未就绪，请稍候", Toast.LENGTH_SHORT).show();
                return;
            }
            if (!isRecognizing) {
                startRecognizing();
                recogBtn.setText("正在识别");
            }
        });

        // ---- 扬声器按钮 ----
        Button engineBtn = findViewById(R.id.engineButton);
        engineBtn.setOnClickListener(v -> {
            if (!engineActive) {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                        != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(this,
                            new String[]{Manifest.permission.RECORD_AUDIO},
                            RECORD_REQUEST_CODE);
                } else {
                    doStartEngine(engineBtn);
                }
            } else {
                if (isCollecting) stopCollecting();
                if (isRecognizing) stopRecognizing();
                stopEngine();
                engineActive = false;
                engineBtn.setText("启动扬声器");
                Toast.makeText(this, "扬声器已关闭", Toast.LENGTH_SHORT).show();
            }
        });

        switchMode(true);
    }

    private void doStartEngine(Button engineBtn) {
        startEngine();
        engineActive = true;
        engineBtn.setText("关闭扬声器");
        Toast.makeText(this, "扬声器已启动", Toast.LENGTH_SHORT).show();

        // 后台加载模型
        if (!modelReady && !modelPath.isEmpty()) {
            Toast.makeText(this, "模型加载中...", Toast.LENGTH_SHORT).show();
            new Thread(() -> {
                boolean ok = loadModel(modelPath);
                modelReady = ok;
                handler.post(() -> {
                    if (ok) {
                        Toast.makeText(this, "模型就绪", Toast.LENGTH_SHORT).show();
                    } else {
                        Toast.makeText(this, "模型加载失败", Toast.LENGTH_SHORT).show();
                    }
                });
            }).start();
        }
    }

    // ==========================================
    // 模式切换
    // ==========================================
    private void switchMode(boolean record) {
        recordMode = record;
        if (record) {
            recordModeBtn.setBackgroundResource(R.drawable.btn_gray);
            recordModeBtn.setTextColor(getColor(R.color.recog_text));
            recogModeBtn.setBackgroundResource(R.drawable.btn_blue);
            recogModeBtn.setTextColor(getColor(android.R.color.white));
        } else {
            recordModeBtn.setBackgroundResource(R.drawable.btn_blue);
            recordModeBtn.setTextColor(getColor(android.R.color.white));
            recogModeBtn.setBackgroundResource(R.drawable.btn_gray);
            recogModeBtn.setTextColor(getColor(R.color.recog_text));
        }
        recordArea.setVisibility(record ? View.VISIBLE : View.GONE);
        recogArea.setVisibility(record ? View.GONE : View.VISIBLE);
    }

    private void showInfoDialog() {
        new androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("相关信息")
            .setMessage("基于声学感知的移动端手势识别系统\n版本: 5.0\nAndroid: 12+\n\n作者: 陈文杰\n学号: 2103021008")
            .setPositiveButton("关闭", null)
            .show();
    }

    // ==========================================
    // 模型文件：从 assets 复制到内部存储
    // ==========================================
    private void prepareModelFile() {
        File outFile = new File(getFilesDir(), "model.onnx");
        modelPath = outFile.getAbsolutePath();
        if (outFile.exists()) return;

        try (InputStream in = getAssets().open("model.onnx");
             FileOutputStream out = new FileOutputStream(outFile)) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) != -1) {
                out.write(buf, 0, n);
            }
        } catch (Exception e) {
            modelPath = "";
        }
    }

    // ==========================================
    // 采集
    // ==========================================
    private void startCollecting() {
        String gestureEn = GESTURE_EN.get(currentGesture);
        File dir = new File(getExternalFilesDir("pcm_data"), gestureEn);
        if (!dir.exists()) dir.mkdirs();
        int count = getStartNumber();
        String filename = gestureEn + "_" + String.format(Locale.US, "%03d", count) + ".pcm";
        File file = new File(dir, filename);
        startRecording(file.getAbsolutePath());
        isCollecting = true;
        collectBtn.setEnabled(false);
        Toast.makeText(this, "录制: " + filename + "  (3s自动停止)", Toast.LENGTH_SHORT).show();

        autoStopRunnable = () -> stopCollecting();
        handler.postDelayed(autoStopRunnable, AUTO_STOP_MS);
    }

    private void stopCollecting() {
        if (autoStopRunnable != null) {
            handler.removeCallbacks(autoStopRunnable);
            autoStopRunnable = null;
        }
        stopRecording();
        isCollecting = false;
        int count = getStartNumber();
        EditText input = findViewById(R.id.startNumberInput);
        input.setText(String.valueOf(count + 1));
        collectBtn.setEnabled(true);
        collectBtn.setText("开始采集");
        updateCounterDisplay();
        Toast.makeText(this, "已保存", Toast.LENGTH_SHORT).show();
    }

    private int getStartNumber() {
        try {
            return Integer.parseInt(
                    ((EditText) findViewById(R.id.startNumberInput)).getText().toString());
        } catch (NumberFormatException e) {
            return 1;
        }
    }

    private void updateCounterDisplay() {
        TextView tv = findViewById(R.id.counterText);
        if (tv != null) {
            tv.setText("已采集: " + (getStartNumber() - 1) + " 次");
        }
    }

    // ==========================================
    // 识别
    // ==========================================
    private void startRecognizing() {
        // 生成临时 PCM 保存路径
        File tempPcm = new File(getFilesDir(), "last_recog.pcm");
        startRecognition(tempPcm.getAbsolutePath());
        isRecognizing = true;
        recogBtn.setEnabled(false);
        recogResultText.setText("");
        recogInfoText.setText("");
        spectrogramView.setImageBitmap(null);
        envelopeView.setImageBitmap(null);
        Toast.makeText(this, "识别中...  (3s)", Toast.LENGTH_SHORT).show();

        autoStopRunnable = () -> stopRecognizing();
        handler.postDelayed(autoStopRunnable, AUTO_STOP_MS);
    }

    private void stopRecognizing() {
        if (autoStopRunnable != null) {
            handler.removeCallbacks(autoStopRunnable);
            autoStopRunnable = null;
        }
        String json = stopRecognition();
        isRecognizing = false;
        recogBtn.setEnabled(true);
        recogBtn.setText("开始识别");

        // 解析 JSON（现在包含信号元数据 + 分类结果）
        try {
            org.json.JSONObject obj = new org.json.JSONObject(json);
            int nChirps = obj.getInt("nChirps");
            double range = obj.getDouble("bestRangeCm");
            double doppler = obj.getDouble("maxDopplerHz");
            specCols = obj.optInt("specCols", 26);
            envLen = obj.optInt("envLen", 100);

            // 显示信号信息
            recogInfoText.setText(String.format(Locale.US,
                    "帧数: %d | 距离: %.1fcm | 多普勒: %.1fHz",
                    nChirps, range, doppler));

            // 显示分类结果
            String name = obj.optString("name", "");
            int classId = obj.optInt("class", -1);
            if (!name.isEmpty()) {
                recogResultText.setText("识别结果: " + name);
                recogResultText.setTextColor(Color.WHITE);
                // 显示对应手势图片
                int[] gestureDrawables = {
                    R.drawable.gesture_push, R.drawable.gesture_pull,
                    R.drawable.gesture_sweep, R.drawable.gesture_slide,
                    R.drawable.gesture_fistbump
                };
                if (classId >= 0 && classId < gestureDrawables.length) {
                    gestureImageView.setImageResource(gestureDrawables[classId]);
                    gestureImageView.setVisibility(View.VISIBLE);
                }
            } else {
                gestureImageView.setVisibility(View.GONE);
            }
        } catch (Exception e) {
            recogInfoText.setText("解析失败: " + e.getMessage());
        }

        // 渲染谱图
        float[] specData = getSpectrogram();
        if (specData != null && specData.length > 0) {
            int targetW = getResources().getDisplayMetrics().widthPixels - dpToPx(36);
            Bitmap bmp = renderSpectrogram(specData, specCols, SPEC_FREQ, targetW);
            spectrogramView.setImageBitmap(bmp);
        }

        // 渲染包络
        float[] envData = getEnvelope();
        if (envData != null && envData.length > 0) {
            int targetW = getResources().getDisplayMetrics().widthPixels - dpToPx(36);
            Bitmap bmp = renderEnvelope(envData, envLen, targetW);
            envelopeView.setImageBitmap(bmp);
        }

        Toast.makeText(this, "识别完成", Toast.LENGTH_SHORT).show();
    }

    // ==========================================
    // 谱图渲染 — 匹配 Python: Y轴 -0.2(下) ~ -1.4(上) Hz, 宽高比 2.5:1
    // 原理: raw 32 bins 自然FFT序, extent 线性映射 0→-1.56Hz, viewport 裁剪到 [-1.4, -0.2]
    // ==========================================
    private Bitmap renderSpectrogram(float[] data, int cols, int rows, int targetW) {
        int nSrcRows = rows;    // 32
        int nCols = cols;       // 动态, ~17

        // 自然FFT序: fd[0]=0Hz .. fd[15]=+23.4Hz, fd[16]=-25Hz .. fd[31]=-1.56Hz
        // 线性 extent: row r → dataY = -r * dYPerRow, 范围 [0, -1.5625]
        float fsByN = (float) kSlowFS / nSrcRows;     // 1.5625
        float dYPerRow = fsByN / (nSrcRows - 1);       // 0.0504 Hz/row
        float viewBottom = -0.2f;   // 对应 dataY 下边界
        float viewTop = -1.4f;      // 对应 dataY 上边界
        float viewRange = viewBottom - viewTop;        // 1.2

        // 可视行: dataY ∈ [-1.4, -0.2]
        int firstRow = Math.max(0, (int) Math.ceil(-viewBottom / dYPerRow));
        int lastRow = Math.min(nSrcRows - 1, (int) Math.floor(-viewTop / dYPerRow));
        if (lastRow < firstRow) { firstRow = 0; lastRow = nSrcRows - 1; }
        int nDisplayRows = lastRow - firstRow + 1;

        // 匹配 Python figsize(13,5) → 宽高比 2.5:1
        int cellW = Math.max(4, targetW / nCols);
        int plotW = nCols * cellW;
        int cellH = Math.max(1, (int)(cellW / 3.5f));   // cellW/cellH ≈ 3.5 → plotW/plotH ≈ 2.5
        int plotH = nDisplayRows * cellH;
        int left = 70, topMargin = 28, right = 18, bottom = 55;
        int canvasW = plotW + left + right;
        int canvasH = plotH + topMargin + bottom;

        Bitmap bmp = Bitmap.createBitmap(canvasW, canvasH, Bitmap.Config.ARGB_8888);
        android.graphics.Canvas canvas = new android.graphics.Canvas(bmp);
        canvas.drawColor(Color.rgb(34, 34, 34));
        android.graphics.Paint paint = new android.graphics.Paint();
        paint.setAntiAlias(true);

        // --- 色块 (仅可视行) ---
        float vmin = Float.MAX_VALUE, vmax = -Float.MAX_VALUE;
        for (int i = 0; i < nDisplayRows; i++) {
            int r = firstRow + i;
            for (int c = 0; c < nCols; c++) {
                float v = data[r * nCols + c];
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
        }
        if (vmax - vmin < 1e-8f) vmax = vmin + 1;

        for (int i = 0; i < nDisplayRows; i++) {
            int srcRow = firstRow + i;
            // i=0 (firstRow, dataY≈-0.2) 放底部, i=last (dataY≈-1.4) 放顶部
            int py = topMargin + (nDisplayRows - 1 - i) * cellH;
            for (int c = 0; c < nCols; c++) {
                float v = (data[srcRow * nCols + c] - vmin) / (vmax - vmin);
                paint.setColor(jetColor(v));
                int px = left + c * cellW;
                canvas.drawRect(px, py, px + cellW, py + cellH, paint);
            }
        }

        // --- 轴标签 ---
        paint.setColor(Color.WHITE);
        paint.setTextSize(20);

        // Y轴标题
        canvas.save();
        canvas.rotate(-90, left - 34, topMargin + plotH / 2);
        paint.setFakeBoldText(true);
        canvas.drawText("多普勒 (Hz)", left - 34 - 55, topMargin + plotH / 2 + 6, paint);
        canvas.restore();

        // X轴标题
        canvas.drawText("时间 (s)", left + plotW / 2 - 32, topMargin + plotH + 32, paint);
        paint.setFakeBoldText(false);
        paint.setTextSize(14);

        // Y轴刻度 (viewBottom=-0.2 在下, viewTop=-1.4 在上)
        float[] yTicks = {-0.2f, -0.5f, -0.8f, -1.1f, -1.4f};
        for (float tick : yTicks) {
            float normY = (tick - viewTop) / viewRange;  // -0.2→1.0, -1.4→0.0
            int y = topMargin + (int)(normY * plotH);
            String label = String.format(Locale.US, "%.1f", tick);
            canvas.drawText(label, 4, y + 4, paint);
            paint.setColor(Color.argb(50, 255, 255, 255));
            canvas.drawLine(left, y, left + plotW, y, paint);
            paint.setColor(Color.WHITE);
        }

        // X轴刻度
        float secPerCol = (float) kRecogDuration / nCols;
        for (int c = 0; c < nCols; c += 5) {
            float sec = c * secPerCol;
            String label = String.format(Locale.US, "%.1f", sec);
            int x = left + c * cellW + cellW / 2 - 10;
            canvas.drawText(label, x, topMargin + plotH + 16, paint);
        }

        // 边框
        paint.setStyle(android.graphics.Paint.Style.STROKE);
        paint.setColor(Color.WHITE);
        paint.setStrokeWidth(2);
        canvas.drawRect(left, topMargin, left + plotW, topMargin + plotH, paint);

        return bmp;
    }

    // ==========================================
    // 幅度包络图渲染 — 自适应Y轴，匹配 Python 显示
    // ==========================================
    private Bitmap renderEnvelope(float[] data, int dataLen, int targetW) {
        int nPts = dataLen;
        // 自适应Y轴：p5*0.6 ~ p98*1.25 (同Python)
        float[] sorted = Arrays.copyOf(data, nPts);
        java.util.Arrays.sort(sorted);
        float p5 = sorted[(int)(0.05f * nPts)];
        float p98 = sorted[(int)(0.98f * nPts)];
        float yLo = Math.max(0, p5 * 0.6f);
        float yHi = p98 * 1.25f;
        if (yHi - yLo < 1e-6f) yHi = yLo + 1;

        int left = 65, top = 18, right = 18, bottom = 45;
        int plotW = targetW - left - right;
        int plotH = (int)(plotW * 0.42f);  // 宽高比约 2.4:1
        int canvasW = plotW + left + right;
        int canvasH = plotH + top + bottom;

        Bitmap bmp = Bitmap.createBitmap(canvasW, canvasH, Bitmap.Config.ARGB_8888);
        android.graphics.Canvas canvas = new android.graphics.Canvas(bmp);
        canvas.drawColor(Color.rgb(34, 34, 34));
        android.graphics.Paint paint = new android.graphics.Paint();
        paint.setAntiAlias(true);

        // --- 曲线 ---
        android.graphics.Path path = new android.graphics.Path();
        for (int i = 0; i < nPts; i++) {
            float x = left + (float) i / (nPts - 1) * plotW;
            float normY = Math.max(0, Math.min(1, (data[i] - yLo) / (yHi - yLo)));
            float y = top + (1.0f - normY) * plotH;
            if (i == 0) path.moveTo(x, y);
            else path.lineTo(x, y);
        }
        paint.setStyle(android.graphics.Paint.Style.STROKE);
        paint.setColor(Color.parseColor("#FF6B35"));
        paint.setStrokeWidth(3);
        canvas.drawPath(path, paint);

        // 填充
        path.lineTo(left + plotW, top + plotH);
        path.lineTo(left, top + plotH);
        path.close();
        paint.setStyle(android.graphics.Paint.Style.FILL);
        paint.setColor(Color.argb(35, 255, 107, 53));
        canvas.drawPath(path, paint);

        // --- 标签 ---
        paint.setStyle(android.graphics.Paint.Style.FILL);
        paint.setColor(Color.WHITE);
        paint.setTextSize(20);
        paint.setFakeBoldText(true);
        canvas.drawText("幅度包络", left + plotW / 2 - 40, top + plotH + 26, paint);
        paint.setFakeBoldText(false);
        paint.setTextSize(14);

        // Y轴: 幅度刻度
        canvas.drawText(String.format(Locale.US, "%.2f", yHi), 2, top + 12, paint);
        canvas.drawText(String.format(Locale.US, "%.2f", yLo), 2, top + plotH - 4, paint);

        // X轴: 时间刻度
        float secPerPt = (float) kRecogDuration / nPts;
        paint.setTextAlign(android.graphics.Paint.Align.CENTER);
        for (int t = 0; t <= 4; t++) {
            int idx = t * nPts / 4;
            if (idx >= nPts) idx = nPts - 1;
            float sec = idx * secPerPt;
            float x = left + (float) idx / (nPts - 1) * plotW;
            canvas.drawLine(x, top + plotH, x, top + plotH + 4, paint);
            canvas.drawText(String.format(Locale.US, "%.1f", sec), x, top + plotH + 18, paint);
        }
        paint.setTextAlign(android.graphics.Paint.Align.LEFT);

        // 边框
        paint.setStyle(android.graphics.Paint.Style.STROKE);
        paint.setStrokeWidth(2);
        canvas.drawRect(left, top, left + plotW, top + plotH, paint);

        return bmp;
    }

    private int dpToPx(int dp) {
        float density = getResources().getDisplayMetrics().density;
        return (int)(dp * density + 0.5f);
    }

    private int jetColor(float t) {
        t = Math.max(0, Math.min(1, t));
        float r, g, b;
        if (t < 0.125f) {
            r = 0; g = 0; b = 0.5f + t * 4;
        } else if (t < 0.375f) {
            r = 0; g = (t - 0.125f) * 4; b = 1;
        } else if (t < 0.625f) {
            r = (t - 0.375f) * 4; g = 1; b = 1 - (t - 0.375f) * 4;
        } else if (t < 0.875f) {
            r = 1; g = 1 - (t - 0.625f) * 4; b = 0;
        } else {
            r = 1 - (t - 0.875f) * 4; g = 0; b = 0;
        }
        return Color.rgb(
                Math.max(0, Math.min(255, (int)(r * 255))),
                Math.max(0, Math.min(255, (int)(g * 255))),
                Math.max(0, Math.min(255, (int)(b * 255))));
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == RECORD_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Button btn = findViewById(R.id.engineButton);
                doStartEngine(btn);
            } else {
                Toast.makeText(this, "录音权限被拒绝，无法启动", Toast.LENGTH_SHORT).show();
            }
        }
    }
}
