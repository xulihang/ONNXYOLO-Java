package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.nio.file.Paths;
import java.util.*;

/**
 * RF-DETR (PaddleDetection 导出) ONNX 检测器
 *
 * 适用于文字/漫画文字检测等 RF-DETR 模型, 例如 inference_model.onnx:
 *   输入:  [1, 3, 512, 512]  (拉伸到 512x512, 非 letterbox)
 *   输出1: dets   [1, 300, 4]  归一化框 [cx, cy, w, h]
 *   输出2: labels [1, 300, C]  类别原始 logits (C = 类别数)
 *
 * 与 YOLO 的关键区别:
 *   1. 预处理用拉伸 (keep_ratio=false) + ImageNet 归一化 ((x/255-mean)/std), 不做 letterbox
 *   2. RF-DETR 用 focal loss 训练, 分类是逐类 sigmoid (不是 softmax),
 *      必须对 logits 逐类做 sigmoid, 再取 argmax; 否则背景 query 会被误判为高置信度
 */
public class ONNXRFDETR {

    // 类别名 (漫画/图片文字检测 5 类; 按模型实际输出类别数使用)
    private static final String[] CLASS_NAMES = {
            "vertical", "balloon", "rect", "horizontal", "outside"
    };

    private OrtEnvironment env;
    private OrtSession session;
    private int inpHeight = 512;
    private int inpWidth = 512;
    private float confThreshold = 0.3f;
    private float nmsThreshold = 0.5f;

    private String inputName;
    private String detsOutputName;   // 框输出 [1, 300, 4] = [cx, cy, w, h]
    private String labelsOutputName; // 类别输出 [1, 300, C] = 原始 logits

    // ImageNet 归一化参数 (PaddleDetection RF-DETR), 顺序 R,G,B
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    public ONNXRFDETR(String onnxPath) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        session = env.createSession(Paths.get(onnxPath).toAbsolutePath().toString(), sessionOptions);
        initModelInfo();
    }

    // 识别输入输出名称: 输出中最后一维为 4 的是框, 否则是类别
    private void initModelInfo() throws OrtException {
        Map<String, NodeInfo> inputInfo = session.getInputInfo();
        Map<String, NodeInfo> outputInfo = session.getOutputInfo();

        if (!inputInfo.isEmpty()) {
            inputName = inputInfo.keySet().iterator().next();
        }

        for (Map.Entry<String, NodeInfo> e : outputInfo.entrySet()) {
            ValueInfo info = e.getValue().getInfo();
            if (info instanceof TensorInfo) {
                long[] shape = ((TensorInfo) info).getShape();
                if (shape.length > 0 && shape[shape.length - 1] == 4) {
                    detsOutputName = e.getKey();
                } else {
                    labelsOutputName = e.getKey();
                }
            }
        }
        if (detsOutputName == null || labelsOutputName == null) {
            throw new IllegalStateException("无法识别 RF-DETR 输出: " + outputInfo.keySet());
        }
    }

    public void setInputSize(int width, int height) {
        this.inpWidth = width;
        this.inpHeight = height;
    }

    public void setConfThreshold(float threshold) {
        this.confThreshold = threshold;
    }

    public void setNMSThreshold(float threshold) {
        this.nmsThreshold = threshold;
    }

    // 检测图像
    public List<DetectionResult> detect(Mat image) {
        try {
            int orgW = image.cols();
            int orgH = image.rows();

            // 1. 预处理 (拉伸到 512x512 + ImageNet 归一化)
            float[] inputData = preprocess(image);

            // 2. 创建输入张量
            long[] inputShape = {1, 3, inpHeight, inpWidth};
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), inputShape);

            // 3. 推理
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(inputName, inputTensor);
            OrtSession.Result result = session.run(inputs);

            // 4. 获取两个输出
            float[][][] dets = (float[][][]) result.get(detsOutputName)
                    .orElseThrow(() -> new RuntimeException("缺少框输出")).getValue();
            float[][][] labels = (float[][][]) result.get(labelsOutputName)
                    .orElseThrow(() -> new RuntimeException("缺少类别输出")).getValue();

            // 5. 后处理
            List<DetectionResult> detections = postprocess(dets[0], labels[0], orgW, orgH);

            // 6. 清理
            inputTensor.close();
            result.close();

            return detections;

        } catch (Exception e) {
            throw new RuntimeException("推理失败: " + e.getMessage(), e);
        }
    }

    // 预处理: 拉伸到 (512,512), BGR->RGB, (x/255-mean)/std, 输出 CHW float 数组
    private float[] preprocess(Mat image) {
        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(inpWidth, inpHeight));

        float[] data = new float[3 * inpHeight * inpWidth];
        int plane = inpHeight * inpWidth;

        for (int i = 0; i < inpHeight; i++) {
            for (int j = 0; j < inpWidth; j++) {
                double[] p = resized.get(i, j); // BGR
                // R 通道
                data[0 * plane + i * inpWidth + j] = (float) ((p[2] / 255.0 - MEAN[0]) / STD[0]);
                // G 通道
                data[1 * plane + i * inpWidth + j] = (float) ((p[1] / 255.0 - MEAN[1]) / STD[1]);
                // B 通道
                data[2 * plane + i * inpWidth + j] = (float) ((p[0] / 255.0 - MEAN[2]) / STD[2]);
            }
        }

        resized.release();
        return data;
    }

    // 后处理: sigmoid -> argmax 类别 + 置信度; cxcywh -> xyxy 映射回原图; 按类别 NMS
    private List<DetectionResult> postprocess(float[][] dets, float[][] labels, int orgW, int orgH) {
        // 候选框: [x1, y1, x2, y2, confidence, classId]
        List<float[]> candidates = new ArrayList<>();
        int numClasses = labels[0].length;

        for (int q = 0; q < dets.length; q++) {
            // 逐类 sigmoid (RF-DETR 用 focal loss, 不是 softmax)
            float bestProb = -1f;
            int bestClass = -1;
            for (int c = 0; c < numClasses; c++) {
                float prob = (float) (1.0 / (1.0 + Math.exp(-labels[q][c])));
                if (prob > bestProb) {
                    bestProb = prob;
                    bestClass = c;
                }
            }

            if (bestProb < confThreshold) continue;

            // 解码归一化 [cx, cy, w, h] -> 原图像素 [x1, y1, x2, y2]
            float cx = dets[q][0], cy = dets[q][1], w = dets[q][2], h = dets[q][3];
            float x1 = (cx - w * 0.5f) * orgW;
            float y1 = (cy - h * 0.5f) * orgH;
            float x2 = (cx + w * 0.5f) * orgW;
            float y2 = (cy + h * 0.5f) * orgH;

            if (x2 <= x1 || y2 <= y1) continue;

            candidates.add(new float[]{x1, y1, x2, y2, bestProb, bestClass});
        }

        // 按类别分组做 NMS
        Map<Integer, List<float[]>> class2Bbox = new HashMap<>();
        for (float[] bbox : candidates) {
            int classId = (int) bbox[5];
            class2Bbox.putIfAbsent(classId, new ArrayList<>());
            class2Bbox.get(classId).add(bbox);
        }

        List<DetectionResult> detections = new ArrayList<>();
        for (Map.Entry<Integer, List<float[]>> entry : class2Bbox.entrySet()) {
            int classId = entry.getKey();
            List<float[]> kept = nonMaxSuppression(entry.getValue(), nmsThreshold);
            for (float[] bbox : kept) {
                float[] box = new float[4];
                box[0] = clamp(bbox[0], 0, orgW - 1);
                box[1] = clamp(bbox[1], 0, orgH - 1);
                box[2] = clamp(bbox[2], 0, orgW - 1);
                box[3] = clamp(bbox[3], 0, orgH - 1);
                detections.add(new DetectionResult(classId, box, bbox[4]));
            }
        }
        return detections;
    }

    private float clamp(float v, float min, float max) {
        return Math.max(min, Math.min(max, v));
    }

    // 非极大值抑制
    private List<float[]> nonMaxSuppression(List<float[]> bboxes, float iouThreshold) {
        List<float[]> bestBboxes = new ArrayList<>();
        bboxes.sort(Comparator.comparing(a -> a[4]));

        while (!bboxes.isEmpty()) {
            float[] best = bboxes.remove(bboxes.size() - 1);
            bestBboxes.add(best);

            List<float[]> filtered = new ArrayList<>();
            for (float[] bbox : bboxes) {
                if (computeIOU(bbox, best) < iouThreshold) {
                    filtered.add(bbox);
                }
            }
            bboxes = filtered;
        }
        return bestBboxes;
    }

    // 计算 IoU
    private float computeIOU(float[] box1, float[] box2) {
        float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);

        float left = Math.max(box1[0], box2[0]);
        float top = Math.max(box1[1], box2[1]);
        float right = Math.min(box1[2], box2[2]);
        float bottom = Math.min(box1[3], box2[3]);

        float interArea = Math.max(right - left, 0) * Math.max(bottom - top, 0);
        float unionArea = area1 + area2 - interArea;

        return Math.max(interArea / unionArea, 1e-8f);
    }

    // 绘制检测结果
    public Mat drawDetections(Mat image, List<DetectionResult> detections) {
        Mat result = image.clone();
        for (DetectionResult det : detections) {
            float[] box = det.bbox;
            String className = det.classId < CLASS_NAMES.length ? CLASS_NAMES[det.classId] : "cls_" + det.classId;
            String label = String.format("%s: %.2f", className, det.confidence);

            Scalar color = new Scalar(0, 0, 255);
            Imgproc.rectangle(result, new Point(box[0], box[1]), new Point(box[2], box[3]), color, 2);

            Size textSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, 1, null);
            Point textOrg = new Point(box[0], box[1] - 4);
            Rect textBg = new Rect((int) box[0], (int) (box[1] - textSize.height - 6),
                    (int) textSize.width + 6, (int) textSize.height + 6);
            Imgproc.rectangle(result, textBg.tl(), textBg.br(), color, -1);
            Imgproc.putText(result, label, textOrg, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5,
                    new Scalar(255, 255, 255), 1);
        }
        return result;
    }

    public void close() throws OrtException {
        if (session != null) session.close();
        if (env != null) env.close();
    }

    // 检测结果类
    public static class DetectionResult {
        private int classId;
        private float[] bbox; // [x1, y1, x2, y2]
        private float confidence;

        public DetectionResult(int classId, float[] bbox, float confidence) {
            this.classId = classId;
            this.bbox = bbox;
            this.confidence = confidence;
        }

        public int getClassId() { return classId; }

        public float[] getBbox() { return bbox; }

        public float getConfidence() { return confidence; }

        public String getClassName() {
            return classId < CLASS_NAMES.length ? CLASS_NAMES[classId] : "class_" + classId;
        }

        public Rect2d getRect() {
            return new Rect2d(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
        }

        @Override
        public String toString() {
            return String.format("%s: %.3f @ [%.0f, %.0f, %.0f, %.0f]",
                    getClassName(), confidence, bbox[0], bbox[1], bbox[2], bbox[3]);
        }
    }

    // 使用示例
    public static void main(String[] args) {
        // 模型放项目根目录, 与 model.onnx / yolo26.onnx 一致
        String modelPath = "inference_model.onnx";
        String imagePath = "003_0101.jpg";

        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

            System.out.println("加载模型: " + modelPath);
            ONNXRFDETR detector = new ONNXRFDETR(modelPath);

            Mat image = Imgcodecs.imread(imagePath);
            if (image.empty()) {
                System.err.println("无法读取图片: " + imagePath);
                return;
            }
            System.out.printf("输入图片: %dx%d%n", image.width(), image.height());

            List<DetectionResult> detections = detector.detect(image);
            System.out.println("检测数量: " + detections.size());

            for (DetectionResult det : detections) {
                System.out.println("  " + det);
            }

            Mat result = detector.drawDetections(image, detections);
            Imgcodecs.imwrite("result_rfdetr.jpg", result);
            System.out.println("\n结果保存: result_rfdetr.jpg");

            image.release();
            result.release();
            detector.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
