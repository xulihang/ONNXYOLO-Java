package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.nio.file.Paths;
import java.util.*;

/**
 * KoharuLayout RF-DETR Seg 2XL ONNX 检测/分割器 (漫画版面分析)
 *
 * 模型由 koharu/model.safetensors 导出 (见 koharu/export_onnx.py);
 * INT8 版本由 koharu/quantize_onnx.py 生成, 输出结构不变, 可直接替换模型路径.
 *   输入:  input  [1, 3, 1152, 1152]  拉伸缩放 (非 letterbox), RGB, ImageNet 归一化
 *   输出1: dets   [1, 300, 4]         归一化框 [cx, cy, w, h]
 *   输出2: labels [1, 300, 5]         类别 logits, 最后一列为 no-object, 前 4 列为真实类别
 *   输出3: masks  [1, 300, 288, 288]  实例 mask logits (输入分辨率的 1/4)
 *
 * 类别: 0=text, 1=onomatopoeia, 2=bubble, 3=panel
 *
 * 后处理 (与 rfdetr 官方 ONNX 推理一致):
 *   1. 丢弃 labels 最后一列, 逐类 sigmoid (focal loss, 不是 softmax)
 *   2. 在 query×class 上取 top-k (k = numSelect, 默认 160)
 *   3. 按类别阈值过滤, 归一化 cxcywh -> 原图像素 xyxy
 *   4. onomatopoeia 不使用 dets 的框, 改为用 mask 非零像素的外接矩形定位
 *   5. onomatopoeia 的 mask 高度重叠时合并为一个 detection (mask 取并集, 框取并集外接矩形)
 *   6. 可选的同类重叠抑制: 只保留较小框 (默认关闭; 开启后大框被删,
 *      而大框的 mask 往往才是完整拟声词, 可能导致 mask 不完整)
 *   7. mask 双线性放大到原图尺寸, logit > 0 视为前景
 */
public class ONNXRFDETRSeg {

    // 类别名 (RF-DETR Seg 2XL, 4 类)
    private static final String[] CLASS_NAMES = {
            "text", "onomatopoeia", "bubble", "panel"
    };

    // 推荐的类别阈值 (koharu 模型卡; 拟声词从 0.20 提到 0.25 以压掉低分误检)
    private static final float[] DEFAULT_CLASS_THRESHOLDS = {
            0.25f, 0.25f, 0.50f, 0.50f
    };

    private static final int ONOMATOPOEIA_CLASS_ID = 1;
    private static final int PANEL_CLASS_ID = 3;

    // ImageNet 归一化参数, 顺序 R,G,B
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    // 类别颜色 (BGR)
    private static final Scalar[] COLORS = {
            new Scalar(60, 60, 255),    // text: 红
            new Scalar(0, 165, 255),    // onomatopoeia: 橙
            new Scalar(255, 160, 60),   // bubble: 蓝
            new Scalar(60, 220, 60)     // panel: 绿
    };

    private static final double MASK_OVERLAY_ALPHA = 0.35;

    private OrtEnvironment env;
    private OrtSession session;

    private int inpHeight = 1152;
    private int inpWidth = 1152;

    // 每个类别独立的置信度阈值
    private float[] classThresholds = DEFAULT_CLASS_THRESHOLDS.clone();
    // 无类别阈值配置时的回退阈值
    private float fallbackThreshold = 0.3f;
    // 在 query×class 上保留的候选数量 (模型卡推荐 160)
    private int numSelect = 160;

    // 用 mask 外接矩形代替模型框的类别 (默认仅拟声词)
    private final Set<Integer> useMaskBboxClasses = new HashSet<>(Collections.singletonList(ONOMATOPOEIA_CLASS_ID));

    // 将 mask 高度重叠的重复检测合并为一类的类别 (默认仅拟声词)
    private final Set<Integer> mergeOverlappingMaskClasses = new HashSet<>(Collections.singletonList(ONOMATOPOEIA_CLASS_ID));
    // 两个 mask 的交集 / 较小 mask 面积 >= 该值视为同一个目标
    private float maskMergeThreshold = 0.5f;

    // 重叠时只保留较小框的类别 (默认关闭: 保留所有框, mask 更完整)
    private final Set<Integer> keepSmallerOnOverlapClasses = new HashSet<>();
    // 交集面积 / 较小框面积 >= 该值视为重叠
    private float overlapThreshold = 0.5f;

    private String inputName;
    private String detsOutputName;   // [1, Q, 4]
    private String labelsOutputName; // [1, Q, C+1]
    private String masksOutputName;  // [1, Q, h, w], 可能不存在

    public ONNXRFDETRSeg(String onnxPath) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        session = env.createSession(Paths.get(onnxPath).toAbsolutePath().toString(), sessionOptions);
        initModelInfo();
    }

    // 从模型读取输入尺寸并识别输出名称; 名称不可靠时按形状回退
    private void initModelInfo() throws OrtException {
        Map<String, NodeInfo> inputInfo = session.getInputInfo();
        Map<String, NodeInfo> outputInfo = session.getOutputInfo();

        if (!inputInfo.isEmpty()) {
            inputName = inputInfo.keySet().iterator().next();
            ValueInfo info = inputInfo.get(inputName).getInfo();
            if (info instanceof TensorInfo) {
                long[] shape = ((TensorInfo) info).getShape();
                // [1, 3, H, W]
                if (shape.length == 4 && shape[2] > 0 && shape[3] > 0) {
                    inpHeight = (int) shape[2];
                    inpWidth = (int) shape[3];
                }
            }
        }

        for (Map.Entry<String, NodeInfo> e : outputInfo.entrySet()) {
            String name = e.getKey();
            String lower = name.toLowerCase(Locale.ROOT);
            if (lower.contains("det")) {
                detsOutputName = name;
            } else if (lower.contains("label") || lower.contains("logit")) {
                labelsOutputName = name;
            } else if (lower.contains("mask")) {
                masksOutputName = name;
            }
        }

        // 按形状回退识别: rank4 -> masks, rank3 且最后一维=4 -> dets, rank3 -> labels
        for (Map.Entry<String, NodeInfo> e : outputInfo.entrySet()) {
            String name = e.getKey();
            if (name.equals(detsOutputName) || name.equals(labelsOutputName) || name.equals(masksOutputName)) {
                continue;
            }
            ValueInfo info = e.getValue().getInfo();
            if (!(info instanceof TensorInfo)) continue;
            long[] shape = ((TensorInfo) info).getShape();
            if (shape.length == 4) {
                masksOutputName = name;
            } else if (shape.length == 3) {
                if (shape[2] == 4) {
                    detsOutputName = name;
                } else {
                    labelsOutputName = name;
                }
            }
        }

        if (detsOutputName == null || labelsOutputName == null) {
            throw new IllegalStateException("无法识别 RF-DETR Seg 输出: " + outputInfo.keySet());
        }
    }

    public void setInputSize(int width, int height) {
        this.inpWidth = width;
        this.inpHeight = height;
    }

    // 设置全部类别的阈值, 按 classId 顺序
    public void setClassThresholds(float... thresholds) {
        if (thresholds == null || thresholds.length == 0) {
            throw new IllegalArgumentException("thresholds 不能为空");
        }
        this.classThresholds = thresholds.clone();
    }

    // 设置某个类别的阈值
    public void setClassThreshold(int classId, float threshold) {
        if (classId < 0 || classId >= classThresholds.length) {
            throw new IllegalArgumentException("classId 越界: " + classId);
        }
        classThresholds[classId] = threshold;
    }

    // 未配置类别阈值时的回退阈值
    public void setConfThreshold(float threshold) {
        this.fallbackThreshold = threshold;
    }

    // 在 query×class 上取 top-k
    public void setNumSelect(int numSelect) {
        this.numSelect = numSelect;
    }

    // 某类别是否改用 mask 外接矩形作为检测框 (默认仅拟声词启用)
    public void setUseMaskBbox(int classId, boolean enabled) {
        if (enabled) {
            useMaskBboxClasses.add(classId);
        } else {
            useMaskBboxClasses.remove(classId);
        }
    }

    // 某类别是否合并 mask 高度重叠的重复检测 (默认仅拟声词启用)
    public void setMergeOverlappingMasks(int classId, boolean enabled) {
        if (enabled) {
            mergeOverlappingMaskClasses.add(classId);
        } else {
            mergeOverlappingMaskClasses.remove(classId);
        }
    }

    // 交集 / 较小 mask 面积达到该比例才合并
    public void setMaskMergeThreshold(float threshold) {
        if (threshold <= 0 || threshold > 1) {
            throw new IllegalArgumentException("maskMergeThreshold 需在 (0, 1] 之间");
        }
        this.maskMergeThreshold = threshold;
    }

    // 某类别重叠时是否只保留较小框 (默认关闭)
    public void setKeepSmallerOnOverlap(int classId, boolean enabled) {
        if (enabled) {
            keepSmallerOnOverlapClasses.add(classId);
        } else {
            keepSmallerOnOverlapClasses.remove(classId);
        }
    }

    // 交集 / 较小框面积 达到该比例才算重叠
    public void setOverlapThreshold(float threshold) {
        if (threshold <= 0 || threshold > 1) {
            throw new IllegalArgumentException("overlapThreshold 需在 (0, 1] 之间");
        }
        this.overlapThreshold = threshold;
    }

    // 检测 + 分割
    public List<DetectionResult> detect(Mat image) {
        try {
            int orgW = image.cols();
            int orgH = image.rows();

            float[] inputData = preprocess(image);
            long[] inputShape = {1, 3, inpHeight, inpWidth};

            try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), inputShape);
                 OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor))) {

                float[][][] dets = (float[][][]) result.get(detsOutputName)
                        .orElseThrow(() -> new IllegalStateException("缺少 dets 输出"))
                        .getValue();
                float[][][] labels = (float[][][]) result.get(labelsOutputName)
                        .orElseThrow(() -> new IllegalStateException("缺少 labels 输出"))
                        .getValue();

                float[][][][] masks = null;
                if (masksOutputName != null) {
                    Optional<OnnxValue> maskValue = result.get(masksOutputName);
                    if (maskValue.isPresent()) {
                        masks = (float[][][][]) maskValue.get().getValue();
                    }
                }

                return postprocess(dets[0], labels[0], masks == null ? null : masks[0], orgW, orgH);
            }

        } catch (Exception e) {
            throw new RuntimeException("推理失败: " + e.getMessage(), e);
        }
    }

    // 预处理: 拉伸到输入尺寸, BGR->RGB, (x/255-mean)/std, 输出 CHW float 数组
    private float[] preprocess(Mat image) {
        Mat bgr = image;
        Mat converted = null;
        if (image.channels() == 1) {
            converted = new Mat();
            Imgproc.cvtColor(image, converted, Imgproc.COLOR_GRAY2BGR);
            bgr = converted;
        } else if (image.channels() == 4) {
            converted = new Mat();
            Imgproc.cvtColor(image, converted, Imgproc.COLOR_BGRA2BGR);
            bgr = converted;
        }

        Mat resized = new Mat();
        Imgproc.resize(bgr, resized, new Size(inpWidth, inpHeight), 0, 0, Imgproc.INTER_LINEAR);

        byte[] pixels = new byte[inpHeight * inpWidth * 3];
        resized.get(0, 0, pixels);

        float[] data = new float[3 * inpHeight * inpWidth];
        int plane = inpHeight * inpWidth;
        for (int i = 0; i < plane; i++) {
            int b = pixels[i * 3] & 0xFF;
            int g = pixels[i * 3 + 1] & 0xFF;
            int r = pixels[i * 3 + 2] & 0xFF;
            data[i] = (float) ((r / 255.0 - MEAN[0]) / STD[0]);
            data[plane + i] = (float) ((g / 255.0 - MEAN[1]) / STD[1]);
            data[2 * plane + i] = (float) ((b / 255.0 - MEAN[2]) / STD[2]);
        }

        resized.release();
        if (converted != null) converted.release();
        return data;
    }

    // 后处理: 丢弃 no-object 列 -> sigmoid -> top-k -> 类别阈值 -> 重叠抑制 -> 框/mask 映射
    private List<DetectionResult> postprocess(float[][] dets, float[][] labels, float[][][] masks,
                                              int orgW, int orgH) {
        int queries = labels.length;
        int logitCount = labels[0].length;
        // RF-DETR 的类别头多一列 no-object
        int numClasses = Math.max(1, logitCount - 1);

        List<ScoreEntry> entries = new ArrayList<>(queries * numClasses);
        for (int q = 0; q < queries; q++) {
            float[] row = labels[q];
            for (int c = 0; c < numClasses; c++) {
                entries.add(new ScoreEntry(q, c, sigmoid(row[c])));
            }
        }

        entries.sort((a, b) -> Float.compare(b.score, a.score));
        int limit = numSelect > 0 ? Math.min(numSelect, entries.size()) : entries.size();

        List<Candidate> candidates = new ArrayList<>();
        for (int i = 0; i < limit; i++) {
            ScoreEntry entry = entries.get(i);
            float threshold = entry.classId < classThresholds.length
                    ? classThresholds[entry.classId] : fallbackThreshold;
            if (entry.score < threshold) continue;

            float[] d = dets[entry.query];
            float cx = d[0], cy = d[1], w = d[2], h = d[3];
            float x1 = clamp((cx - w * 0.5f) * orgW, 0, orgW - 1);
            float y1 = clamp((cy - h * 0.5f) * orgH, 0, orgH - 1);
            float x2 = clamp((cx + w * 0.5f) * orgW, 0, orgW - 1);
            float y2 = clamp((cy + h * 0.5f) * orgH, 0, orgH - 1);
            if (x2 <= x1 || y2 <= y1) continue;

            candidates.add(new Candidate(entry.query, entry.classId, entry.score,
                    new float[]{x1, y1, x2, y2}));
        }

        candidates = suppressOverlappingKeepingSmaller(candidates);

        List<DetectionResult> results = new ArrayList<>(candidates.size());
        for (Candidate candidate : candidates) {
            Mat mask = null;
            float[] bbox = candidate.bbox;
            if (masks != null && candidate.query < masks.length) {
                mask = decodeMask(masks[candidate.query], orgW, orgH);
                if (useMaskBboxClasses.contains(candidate.classId)) {
                    Rect maskRect = maskBoundingRect(mask);
                    if (maskRect != null) {
                        bbox = new float[]{
                                maskRect.x,
                                maskRect.y,
                                maskRect.x + maskRect.width - 1,
                                maskRect.y + maskRect.height - 1
                        };
                    }
                }
            }
            results.add(new DetectionResult(candidate.classId, bbox, candidate.score, mask));
        }
        return mergeOverlappingMasks(results);
    }

    // 同一类别内, 若两个框重叠 (交集 / 较小框面积 >= 阈值), 只保留较小的那个
    private List<Candidate> suppressOverlappingKeepingSmaller(List<Candidate> candidates) {
        Map<Integer, List<Candidate>> byClass = new LinkedHashMap<>();
        for (Candidate candidate : candidates) {
            byClass.computeIfAbsent(candidate.classId, k -> new ArrayList<>()).add(candidate);
        }

        List<Candidate> kept = new ArrayList<>(candidates.size());
        for (Map.Entry<Integer, List<Candidate>> entry : byClass.entrySet()) {
            List<Candidate> list = entry.getValue();
            if (!keepSmallerOnOverlapClasses.contains(entry.getKey())) {
                kept.addAll(list);
                continue;
            }

            // 从小到大检查: 先保留小框, 被已保留小框覆盖的大框丢弃
            list.sort(Comparator.comparingDouble(c -> area(c.bbox)));
            List<Candidate> classKept = new ArrayList<>();
            for (Candidate candidate : list) {
                boolean overlapped = false;
                for (Candidate keptCandidate : classKept) {
                    if (overlapOverSmaller(candidate.bbox, keptCandidate.bbox) >= overlapThreshold) {
                        overlapped = true;
                        break;
                    }
                }
                if (!overlapped) {
                    classKept.add(candidate);
                }
            }
            kept.addAll(classKept);
        }
        return kept;
    }

    private static float area(float[] box) {
        return Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
    }

    // 交集面积 / 较小框面积; 小框完全落在大框内时为 1
    private static float overlapOverSmaller(float[] a, float[] b) {
        float interW = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
        float interH = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
        float inter = interW * interH;
        float minArea = Math.min(area(a), area(b));
        return minArea > 0 ? inter / minArea : 0;
    }

    // mask 非零像素的外接矩形; 空 mask 返回 null
    private static Rect maskBoundingRect(Mat mask) {
        if (mask == null || mask.empty()) return null;

        int w = mask.cols();
        int h = mask.rows();
        byte[] data = new byte[w * h];
        mask.get(0, 0, data);

        int minX = w, minY = h, maxX = -1, maxY = -1;
        for (int y = 0; y < h; y++) {
            int row = y * w;
            for (int x = 0; x < w; x++) {
                if (data[row + x] != 0) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        if (maxX < minX || maxY < minY) return null;
        return new Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
    }

    // 合并同一类别中 mask 高度重叠的重复检测:
    // mask 取并集, 框取并集 mask 的外接矩形, 分数取最高
    private List<DetectionResult> mergeOverlappingMasks(List<DetectionResult> results) {
        if (results.isEmpty() || mergeOverlappingMaskClasses.isEmpty()) return results;

        List<DetectionResult> output = new ArrayList<>(results.size());
        Map<Integer, List<DetectionResult>> byClass = new LinkedHashMap<>();
        for (DetectionResult det : results) {
            if (mergeOverlappingMaskClasses.contains(det.getClassId()) && det.getMask() != null) {
                byClass.computeIfAbsent(det.getClassId(), k -> new ArrayList<>()).add(det);
            } else {
                output.add(det);
            }
        }

        for (Map.Entry<Integer, List<DetectionResult>> entry : byClass.entrySet()) {
            output.addAll(mergeMaskGroup(entry.getKey(), entry.getValue()));
        }
        return output;
    }

    private List<DetectionResult> mergeMaskGroup(int classId, List<DetectionResult> detections) {
        int n = detections.size();
        if (n <= 1) return detections;

        int w = detections.get(0).getMask().cols();
        int h = detections.get(0).getMask().rows();
        byte[][] masks = new byte[n][];
        int[] areas = new int[n];
        for (int i = 0; i < n; i++) {
            byte[] data = new byte[w * h];
            detections.get(i).getMask().get(0, 0, data);
            masks[i] = data;
            areas[i] = countNonZero(data);
        }

        int[] parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                // 框不相交时 mask 不会相交
                float bboxOverlap = overlapOverSmaller(detections.get(i).getBbox(), detections.get(j).getBbox());
                if (bboxOverlap <= 0) continue;

                float maskOverlap = maskOverlapOverSmaller(masks[i], masks[j], areas[i], areas[j]);
                // mask 高度重叠, 或一个框基本包含另一个框, 都视为同一目标
                if (maskOverlap >= maskMergeThreshold || bboxOverlap >= maskMergeThreshold) {
                    union(parent, i, j);
                }
            }
        }

        Map<Integer, List<Integer>> groups = new LinkedHashMap<>();
        for (int i = 0; i < n; i++) {
            groups.computeIfAbsent(find(parent, i), k -> new ArrayList<>()).add(i);
        }

        List<DetectionResult> merged = new ArrayList<>(groups.size());
        for (List<Integer> group : groups.values()) {
            if (group.size() == 1) {
                merged.add(detections.get(group.get(0)));
                continue;
            }

            byte[] unionData = new byte[w * h];
            float bestScore = -1f;
            for (int index : group) {
                byte[] data = masks[index];
                for (int k = 0; k < unionData.length; k++) {
                    if (data[k] != 0) unionData[k] = (byte) 255;
                }
                bestScore = Math.max(bestScore, detections.get(index).getConfidence());
            }

            Mat unionMask = new Mat(h, w, CvType.CV_8UC1);
            unionMask.put(0, 0, unionData);

            float[] bbox = detections.get(group.get(0)).getBbox();
            Rect rect = maskBoundingRect(unionMask);
            if (rect != null) {
                bbox = new float[]{rect.x, rect.y, rect.x + rect.width - 1, rect.y + rect.height - 1};
            }

            for (int index : group) {
                detections.get(index).release();
            }
            merged.add(new DetectionResult(classId, bbox, bestScore, unionMask));
        }
        return merged;
    }

    private static int find(int[] parent, int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    private static void union(int[] parent, int a, int b) {
        int ra = find(parent, a);
        int rb = find(parent, b);
        if (ra != rb) parent[rb] = ra;
    }

    private static int countNonZero(byte[] data) {
        int count = 0;
        for (byte b : data) {
            if (b != 0) count++;
        }
        return count;
    }

    // 交集 / 较小 mask 面积
    private static float maskOverlapOverSmaller(byte[] a, byte[] b, int areaA, int areaB) {
        int minArea = Math.min(areaA, areaB);
        if (minArea <= 0) return 0;

        int inter = 0;
        int len = Math.min(a.length, b.length);
        for (int i = 0; i < len; i++) {
            if (a[i] != 0 && b[i] != 0) inter++;
        }
        return (float) inter / minArea;
    }

    // mask logits [h, w] -> 原图尺寸二值 mask (0/255)
    private Mat decodeMask(float[][] maskLogits, int orgW, int orgH) {
        int maskH = maskLogits.length;
        int maskW = maskLogits[0].length;

        float[] flat = new float[maskH * maskW];
        for (int y = 0; y < maskH; y++) {
            System.arraycopy(maskLogits[y], 0, flat, y * maskW, maskW);
        }

        Mat logits = new Mat(maskH, maskW, CvType.CV_32FC1);
        logits.put(0, 0, flat);

        Mat resized = new Mat();
        Imgproc.resize(logits, resized, new Size(orgW, orgH), 0, 0, Imgproc.INTER_LINEAR);

        Mat binary = new Mat();
        // logit > 0 视为前景
        Imgproc.threshold(resized, binary, 0.0, 255.0, Imgproc.THRESH_BINARY);

        // threshold 会保留 CV_32F, 转成 CV_8UC1 便于绘制和保存
        Mat mask = new Mat();
        binary.convertTo(mask, CvType.CV_8UC1);

        logits.release();
        resized.release();
        binary.release();
        return mask;
    }

    // 绘制 mask 和检测框; panel 只画框, 避免整块面板遮挡画面
    public Mat drawDetections(Mat image, List<DetectionResult> detections) {
        Mat result = image.clone();

        // 半透明叠加 mask
        Mat overlay = image.clone();
        for (DetectionResult det : detections) {
            Mat mask = det.getMask();
            if (mask == null || det.getClassId() == PANEL_CLASS_ID) continue;
            Mat colorLayer = new Mat(image.size(), CvType.CV_8UC3, colorOf(det.getClassId()));
            colorLayer.copyTo(overlay, mask);
            colorLayer.release();
        }
        Core.addWeighted(overlay, MASK_OVERLAY_ALPHA, result, 1.0 - MASK_OVERLAY_ALPHA, 0, result);
        overlay.release();

        for (DetectionResult det : detections) {
            float[] box = det.getBbox();
            Scalar color = colorOf(det.getClassId());
            String label = String.format("%s: %.2f", det.getClassName(), det.getConfidence());

            Imgproc.rectangle(result, new Point(box[0], box[1]), new Point(box[2], box[3]), color, 2);

            Size textSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, 1, null);
            int textW = (int) Math.ceil(textSize.width);
            int textH = (int) Math.ceil(textSize.height);
            int labelX = Math.max(0, Math.min((int) box[0], image.cols() - textW - 4));
            int labelY = (int) box[1] - 4;
            if (labelY - textH < 0) {
                labelY = (int) box[1] + textH + 8;
            }
            labelY = Math.min(labelY, image.rows() - 2);

            Rect textBg = new Rect(labelX, labelY - textH - 2, textW + 4, textH + 4);
            Imgproc.rectangle(result, textBg.tl(), textBg.br(), color, -1);
            Imgproc.putText(result, label, new Point(labelX + 2, labelY),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255), 1);
        }

        return result;
    }

    private static Scalar colorOf(int classId) {
        return COLORS[Math.floorMod(classId, COLORS.length)];
    }

    private static float sigmoid(float x) {
        double v = Math.max(-88.0, Math.min(88.0, x));
        return (float) (1.0 / (1.0 + Math.exp(-v)));
    }

    private static float clamp(float v, float min, float max) {
        return Math.max(min, Math.min(max, v));
    }

    public void close() throws OrtException {
        if (session != null) session.close();
        if (env != null) env.close();
    }

    // 单个 query×class 的候选
    private static class ScoreEntry {
        final int query;
        final int classId;
        final float score;

        ScoreEntry(int query, int classId, float score) {
            this.query = query;
            this.classId = classId;
            this.score = score;
        }
    }

    // 通过阈值、尚未解码 mask 的候选框
    private static class Candidate {
        final int query;
        final int classId;
        final float score;
        final float[] bbox;

        Candidate(int query, int classId, float score, float[] bbox) {
            this.query = query;
            this.classId = classId;
            this.score = score;
            this.bbox = bbox;
        }
    }

    // 检测/分割结果
    public static class DetectionResult {
        private final int classId;
        private final float[] bbox; // [x1, y1, x2, y2] 原图像素
        private final float confidence;
        private Mat mask;           // CV_8UC1, 0/255, 原图尺寸; 可能为 null

        public DetectionResult(int classId, float[] bbox, float confidence, Mat mask) {
            this.classId = classId;
            this.bbox = bbox;
            this.confidence = confidence;
            this.mask = mask;
        }

        public int getClassId() { return classId; }

        public float[] getBbox() { return bbox; }

        public float getConfidence() { return confidence; }

        // 实例分割 mask, 调用方负责 release()
        public Mat getMask() { return mask; }

        public String getClassName() {
            return classId < CLASS_NAMES.length ? CLASS_NAMES[classId] : "class_" + classId;
        }

        public Rect2d getRect() {
            return new Rect2d(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
        }

        public void release() {
            if (mask != null) {
                mask.release();
                mask = null;
            }
        }

        @Override
        public String toString() {
            return String.format("%s: %.3f @ [%.0f, %.0f, %.0f, %.0f]",
                    getClassName(), confidence, bbox[0], bbox[1], bbox[2], bbox[3]);
        }
    }

    // 使用示例: 模型由 koharu/export_onnx.py 导出, INT8 版本由 koharu/quantize_onnx.py 生成
    // 参数: [模型路径] [图片路径] [输出路径]
    public static void main(String[] args) {
        String modelPath = args.length > 0 ? args[0] : "koharu/koharu_layout_rfdetr_seg_2xl_1152_int8.onnx";
        String imagePath = args.length > 1 ? args[1] : "18_094.jpg";
        String outputPath = args.length > 2 ? args[2] : "result_koharu.jpg";

        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

            System.out.println("加载模型: " + modelPath);
            ONNXRFDETRSeg detector = new ONNXRFDETRSeg(modelPath);

            Mat image = Imgcodecs.imread(imagePath);
            if (image.empty()) {
                System.err.println("无法读取图片: " + imagePath);
                return;
            }
            System.out.printf("输入图片: %dx%d%n", image.width(), image.height());

            List<DetectionResult> detections = detector.detect(image);
            System.out.println("检测数量: " + detections.size());
            for (DetectionResult det : detections) {
                String maskInfo = det.getMask() == null ? "无 mask"
                        : det.getMask().cols() + "x" + det.getMask().rows();
                System.out.println("  " + det + "  mask=" + maskInfo);
            }

            Mat result = detector.drawDetections(image, detections);
            Imgcodecs.imwrite(outputPath, result);
            System.out.println("\n结果保存: " + outputPath);

            for (DetectionResult det : detections) {
                det.release();
            }
            image.release();
            result.release();
            detector.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
