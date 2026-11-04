# ObjectDetector-ONNX

基于 Java + OpenCV + ONNX Runtime 的目标检测 / 实例分割推理工具，包含 YOLO、YOLO26、RF-DETR，以及用于漫画版面分析的 Koharu RF-DETR Seg 模型。

## 环境

- JDK 11+
- Maven
- OpenCV 4.9 原生库：把 `opencv_java490.dll` 放在工作目录，或加入 `java.library.path`
- 依赖（见 `pom.xml`）：
  - `com.microsoft.onnxruntime:onnxruntime:1.20.0`
  - `org.openpnp:opencv:4.9.0-0`

## 构建与运行

```powershell
mvn -q -DskipTests test-compile
mvn -q dependency:build-classpath -Dmdep.outputFile=target/cp.txt

$cp = "target/classes;target/test-classes;" + (Get-Content target/cp.txt)
java "-Djava.library.path=." -cp $cp com.xulihang.ONNXRFDETRSeg
```

Linux / macOS 下 classpath 分隔符换成 `:`。

## 检测器

| 类 | 用途 | 输入 | 特点 |
|---|---|---|---|
| `ONNXYOLO` | Ultralytics YOLO 系列 | 640 | letterbox + NMS |
| `ONNXYOLO26` | YOLO26 (COCO 80 类) | 可变 | COCO 类别名与配色 |
| `ONNXRFDETR` | RF-DETR 检测（漫画文字等） | 512 | 拉伸缩放 + ImageNet 归一化；逐类 sigmoid |
| `ONNXRFDETRSeg` | Koharu 漫画版面检测/分割 | 1152 或 768 | 4 类：text / onomatopoeia / bubble / panel |

### ONNXRFDETRSeg

支持两种模型，输入尺寸由模型自动识别：

- **分割版** `koharu/koharu_layout_rfdetr_seg_2xl_1152.onnx`（或 `_int8`）：输出 dets / labels / masks
- **检测版** `koharu/rfdetr-seg-2xlarge.onnx`（768，或 `_int8`）：只有 dets / labels，无 mask

```java
System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

ONNXRFDETRSeg detector = new ONNXRFDETRSeg("koharu/koharu_layout_rfdetr_seg_2xl_1152_int8.onnx");
Mat image = Imgcodecs.imread("18_094.jpg");

List<ONNXRFDETRSeg.DetectionResult> detections = detector.detect(image);

Mat preview = detector.drawDetections(image, detections);
Imgcodecs.imwrite("result_koharu.jpg", preview);

for (ONNXRFDETRSeg.DetectionResult det : detections) {
    det.release();
}
detector.close();
```

默认类别阈值：text `0.25`、onomatopoeia `0.25`、bubble `0.50`、panel `0.50`，可通过
`setClassThreshold(classId, threshold)` 或 `setClassThresholds(...)` 调整。

#### 文字覆盖掩膜

一次 `detect()` 之后即可获取三类掩膜（0/255，原图尺寸）：

```java
Mat textMask = detector.getTextMask();                 // 仅 text 实例并集
Mat onomatopoeiaMask = detector.getOnomatopoeiaMask(); // 仅 onomatopoeia 实例并集

detector.setBubbleMaskShrink(3);                       // 内缩像素，默认 3
if (detector.hasBubbleTextMask()) {
    Mat bubbleMask = detector.getBubbleTextMask();     // 气泡实例并集
    Mat covered = image.clone();
    Scalar white = new Scalar(255, 255, 255);
    covered.setTo(white, bubbleMask);                  // 气泡内文字
    covered.setTo(white, textMask);                    // 气泡外文字
    covered.setTo(white, onomatopoeiaMask);            // 拟声词
    Imgcodecs.imwrite("covered.jpg", covered);
}
```

- `getBubbleTextMask()` 是**惰性生成**的：第一次调用才做去锯齿平滑 + 内缩，之后复用缓存；每次
  `detect()` 或修改 `setBubbleMaskShrink()` 后重建。707×1100 页面上首次生成约 18 ms。
- 内缩是为了保留气泡的黑色轮廓、避免白块溢出；`setBubbleMaskShrink(0)` 回到原始贴合效果。
- `hasBubbleTextMask()` 只检查本次检测结果，不会触发掩膜生成。
- 返回的 Mat 是内部缓存：**不要 `release()`，也不要修改**；只在本次 `detect()` 到下一次
  `detect()` 之间有效，需要跨帧保留请自行 `clone()`。
- `DetectionResult` 中的 mask 由调用方 `release()`。
- 下一次 `detect()` 会自动释放并重建所有缓存掩膜，内存不会跨次累积。

## 测试与工具

| 文件 | 说明 |
|---|---|
| `src/test/java/com/xulihang/BubbleTextMaskTest.java` | 调用 `getBubbleTextMask()` 把气泡内文字涂白并保存结果 |
| `src/test/java/com/xulihang/RFDETRSegBenchmark.java` | 对比 1152 分割版 / 768 检测版的耗时与检测结果 |
| `bubble_mask_preview.py` | 直接用 onnxruntime + OpenCV 预览气泡掩膜涂白效果（对照 Java 实现） |

```powershell
# 气泡文字覆盖测试：输出 18_094_bubble_text_covered.jpg 与 _mask.png
java "-Djava.library.path=." -cp $cp com.xulihang.BubbleTextMaskTest

# 模型对比
java "-Djava.library.path=." -cp $cp com.xulihang.RFDETRSegBenchmark
```

## 模型文件

`koharu/` 目录不会提交到 Git（见 `.gitignore`）。分割版与 INT8 量化版使用完全相同的推理代码，
直接替换模型路径即可。
