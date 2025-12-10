package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.nio.file.Paths;
import java.util.*;

public class ONNXYOLO {
    private OrtEnvironment env;
    private OrtSession session;
    private int inpHeight = 640;
    private int inpWidth = 640;
    private float confThreshold = 0.25f;
    private float nmsThreshold = 0.45f;

    // 预处理参数
    private float[] scaleFactors = new float[]{1.0f, 1.0f}; // width, height 缩放因子
    private int[] pad = new int[]{0, 0}; // width, height 填充

    private String inputName;
    private String outputName;

    public ONNXYOLO(String onnxPath) throws OrtException {
        // 初始化ONNX Runtime环境
        env = OrtEnvironment.getEnvironment();

        // 创建会话选项
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();

        // 创建会话
        session = env.createSession(Paths.get(onnxPath).toAbsolutePath().toString(), sessionOptions);

        // 初始化模型信息
        initModelInfo();
    }

    private void initModelInfo() throws OrtException {
        Map<String, NodeInfo> inputInfo = session.getInputInfo();
        Map<String, NodeInfo> outputInfo = session.getOutputInfo();

        // 获取输入名称
        if (!inputInfo.isEmpty()) {
            inputName = inputInfo.keySet().iterator().next();
        }

        // 获取输出名称
        if (!outputInfo.isEmpty()) {
            outputName = outputInfo.keySet().iterator().next();
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
            // 1. 预处理
            PreprocessResult preprocessResult = preprocess(image);
            float[] inputData = preprocessResult.inputData;

            // 2. 创建输入张量
            long[] inputShape = {1, 3, inpHeight, inpWidth}; // NCHW格式
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), inputShape);

            // 3. 执行推理
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(inputName, inputTensor);

            OrtSession.Result results = session.run(inputs);

            // 4. 获取输出
            Optional<OnnxValue> outputOptional = results.get(outputName);
            if (!outputOptional.isPresent()) {
                throw new RuntimeException("无法获取模型输出");
            }

            OnnxValue outputValue = outputOptional.get();
            Object outputObject = outputValue.getValue();

            // 5. 后处理
            List<DetectionResult> detections;

            if (outputObject instanceof float[][][]) {
                // 输出形状: [1, dimensions, num_boxes]
                float[][][] outputData = (float[][][]) outputObject;
                detections = postprocess3D(outputData, preprocessResult);
            } else if (outputObject instanceof float[][]) {
                // 输出形状: [dimensions, num_boxes]
                float[][] outputData = (float[][]) outputObject;
                detections = postprocess2D(outputData, preprocessResult);
            } else {
                throw new RuntimeException("不支持的输出类型: " + outputObject.getClass().getName());
            }

            // 6. 清理资源
            inputTensor.close();
            results.close();

            return detections;

        } catch (Exception e) {
            throw new RuntimeException("推理失败: " + e.getMessage(), e);
        }
    }

    // 预处理结果封装
    private static class PreprocessResult {
        float[] inputData;
        float ratio;
        float dw;
        float dh;
        int originalWidth;
        int originalHeight;

        PreprocessResult(float[] inputData, float ratio, float dw, float dh, int originalWidth, int originalHeight) {
            this.inputData = inputData;
            this.ratio = ratio;
            this.dw = dw;
            this.dh = dh;
            this.originalWidth = originalWidth;
            this.originalHeight = originalHeight;
        }
    }

    // 预处理（保持宽高比的letterbox）
    private PreprocessResult preprocess(Mat image) {
        int originalHeight = image.rows();
        int originalWidth = image.cols();

        // 计算缩放比例（保持宽高比）
        float ratio = Math.min((float) inpWidth / originalWidth, (float) inpHeight / originalHeight);
        int newWidth = Math.round(originalWidth * ratio);
        int newHeight = Math.round(originalHeight * ratio);

        // 调整大小
        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(newWidth, newHeight));

        // 创建画布并填充
        Mat canvas = new Mat(inpHeight, inpWidth, CvType.CV_8UC3, new Scalar(114, 114, 114));

        // 计算填充
        int dw = (inpWidth - newWidth) / 2;
        int dh = (inpHeight - newHeight) / 2;

        // 将图像复制到画布中心
        Mat roi = canvas.submat(dh, dh + newHeight, dw, dw + newWidth);
        resized.copyTo(roi);

        // 转换为CHW格式的float数组并归一化
        float[] floatArray = new float[3 * inpHeight * inpWidth];

        // 按CHW格式填充数据（参考原代码的像素填充方式）
        for (int i = 0; i < inpHeight; i++) {
            for (int j = 0; j < inpWidth; j++) {
                double[] pixel = canvas.get(i, j);
                // BGR转RGB并归一化
                floatArray[inpHeight * inpWidth * 0 + i * inpWidth + j] = (float) (pixel[2] / 255.0); // R
                floatArray[inpHeight * inpWidth * 1 + i * inpWidth + j] = (float) (pixel[1] / 255.0); // G
                floatArray[inpHeight * inpWidth * 2 + i * inpWidth + j] = (float) (pixel[0] / 255.0); // B
            }
        }

        // 保存预处理参数用于后处理
        scaleFactors[0] = (float) newWidth / originalWidth;
        scaleFactors[1] = (float) newHeight / originalHeight;
        pad[0] = dw;
        pad[1] = dh;

        // 清理
        resized.release();
        canvas.release();
        roi.release();

        return new PreprocessResult(floatArray, ratio, dw, dh, originalWidth, originalHeight);
    }

    // 处理3D输出 [1, dimensions, num_boxes]
    private List<DetectionResult> postprocess3D(float[][][] outputData, PreprocessResult preprocessResult) {
        // 转置矩阵，使形状为 [num_boxes, dimensions]
        float[][] transposed = transposeMatrix(outputData[0]);
        return postprocessTransposed(transposed, preprocessResult);
    }

    // 处理2D输出 [dimensions, num_boxes]
    private List<DetectionResult> postprocess2D(float[][] outputData, PreprocessResult preprocessResult) {
        // 转置矩阵，使形状为 [num_boxes, dimensions]
        float[][] transposed = transposeMatrix(outputData);
        return postprocessTransposed(transposed, preprocessResult);
    }

    // 处理转置后的数据 [num_boxes, dimensions]
    private List<DetectionResult> postprocessTransposed(float[][] outputData, PreprocessResult preprocessResult) {
        List<DetectionResult> detections = new ArrayList<>();

        // 按类别分组
        Map<Integer, List<float[]>> class2Bbox = new HashMap<>();

        for (float[] bbox : outputData) {
            // bbox格式: [x, y, w, h, class_score_1, class_score_2, ...]

            // 提取类别概率
            int numClasses = bbox.length - 4;
            float[] classScores = Arrays.copyOfRange(bbox, 4, bbox.length);

            // 找到最大类别分数
            int classId = argmax(classScores);
            float confidence = classScores[classId];

            // 应用置信度阈值
            if (confidence < confThreshold) continue;

            // 保存置信度
            bbox[4] = confidence;

            // 转换xywh为xyxy
            xywh2xyxy(bbox);

            // 跳过无效预测
            if (bbox[0] >= bbox[2] || bbox[1] >= bbox[3]) continue;

            // 按类别分组
            class2Bbox.putIfAbsent(classId, new ArrayList<>());
            class2Bbox.get(classId).add(bbox);
        }

        // 对每个类别应用NMS
        for (Map.Entry<Integer, List<float[]>> entry : class2Bbox.entrySet()) {
            int classId = entry.getKey();
            List<float[]> bboxes = entry.getValue();

            // 应用非极大值抑制
            bboxes = nonMaxSuppression(bboxes, nmsThreshold);

            // 创建检测结果
            for (float[] bbox : bboxes) {
                // 将坐标映射回原始图像
                float[] scaledBbox = scaleCoords(
                        Arrays.copyOfRange(bbox, 0, 4),
                        preprocessResult.originalWidth,
                        preprocessResult.originalHeight,
                        preprocessResult.dw,
                        preprocessResult.dh,
                        preprocessResult.ratio
                );

                DetectionResult detection = new DetectionResult(
                        classId,
                        scaledBbox,
                        bbox[4]
                );
                detections.add(detection);
            }
        }

        return detections;
    }

    // 转置矩阵（参考原代码）
    private float[][] transposeMatrix(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[][] transposed = new float[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }

        return transposed;
    }

    // 找到最大值的索引（参考原代码）
    private int argmax(float[] array) {
        float max = -Float.MAX_VALUE;
        int arg = -1;

        for (int i = 0; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                arg = i;
            }
        }

        return arg;
    }

    // 转换xywh为xyxy（参考原代码）
    private void xywh2xyxy(float[] bbox) {
        float x = bbox[0];
        float y = bbox[1];
        float w = bbox[2];
        float h = bbox[3];

        bbox[0] = x - w * 0.5f; // x1
        bbox[1] = y - h * 0.5f; // y1
        bbox[2] = x + w * 0.5f; // x2
        bbox[3] = y + h * 0.5f; // y2
    }

    // 将坐标缩放回原始图像（参考原代码）
    private float[] scaleCoords(float[] bbox, float orgW, float orgH, float padW, float padH, float gain) {
        float[] scaled = new float[4];

        // xmin, ymin, xmax, ymax -> (xmin_org, ymin_org, xmax_org, ymax_org)
        scaled[0] = Math.max(0, Math.min(orgW - 1, (bbox[0] - padW) / gain));
        scaled[1] = Math.max(0, Math.min(orgH - 1, (bbox[1] - padH) / gain));
        scaled[2] = Math.max(0, Math.min(orgW - 1, (bbox[2] - padW) / gain));
        scaled[3] = Math.max(0, Math.min(orgH - 1, (bbox[3] - padH) / gain));

        return scaled;
    }

    // 非极大值抑制（参考原代码）
    private List<float[]> nonMaxSuppression(List<float[]> bboxes, float iouThreshold) {
        List<float[]> bestBboxes = new ArrayList<>();

        // 按置信度排序
        bboxes.sort(Comparator.comparing(a -> a[4]));

        while (!bboxes.isEmpty()) {
            float[] bestBbox = bboxes.remove(bboxes.size() - 1); // 取置信度最高的
            bestBboxes.add(bestBbox);

            // 过滤掉与当前框IoU大于阈值的框
            List<float[]> filtered = new ArrayList<>();
            for (float[] bbox : bboxes) {
                if (computeIOU(bbox, bestBbox) < iouThreshold) {
                    filtered.add(bbox);
                }
            }
            bboxes = filtered;
        }

        return bestBboxes;
    }

    // 计算IoU（参考原代码）
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

    // 关闭资源
    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
        if (env != null) {
            env.close();
        }
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

        public int getClassId() {
            return classId;
        }

        public float[] getBbox() {
            return bbox;
        }

        public float getConfidence() {
            return confidence;
        }

        public Rect2d getRect() {
            return new Rect2d(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
        }

        @Override
        public String toString() {
            return String.format("Detection{classId=%d, confidence=%.3f, bbox=[%.1f, %.1f, %.1f, %.1f]}",
                    classId, confidence, bbox[0], bbox[1], bbox[2], bbox[3]);
        }
    }
}