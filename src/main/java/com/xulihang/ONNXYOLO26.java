package com.xulihang;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import ai.onnxruntime.*;

import java.nio.FloatBuffer;
import java.util.*;

public class ONNXYOLO26 {

    // COCOクラス名
    private static final String[] COCO_CLASSES = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    // クラスごとの色を生成
    private static final Scalar[] COLORS;

    static {
        // OpenCVのライブラリをロード
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // 色の初期化
        Random rand = new Random(42);
        COLORS = new Scalar[COCO_CLASSES.length];
        for (int i = 0; i < COLORS.length; i++) {
            COLORS[i] = new Scalar(
                    rand.nextInt(255),
                    rand.nextInt(255),
                    rand.nextInt(255)
            );
        }
    }

    private OrtSession session;
    private String inputName;
    private int inputWidth = 640;
    private int inputHeight = 640;

    /**
     * コンストラクタ
     * @param modelPath ONNXモデルファイルのパス
     */
    public ONNXYOLO26(String modelPath) {
        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            session = env.createSession(modelPath, sessionOptions);
            inputName = session.getInputInfo().keySet().iterator().next();
        } catch (Exception e) {
            throw new RuntimeException("モデルの読み込みに失敗しました", e);
        }
    }

    /**
     * 検出結果を表すクラス
     */
    public static class DetectionResult {
        public Rect bbox;        // バウンディングボックス
        public float score;      // 信頼度スコア
        public int classId;      // クラスID
        public String className; // クラス名

        public DetectionResult(Rect bbox, float score, int classId) {
            this.bbox = bbox;
            this.score = score;
            this.classId = classId;
            this.className = classId < COCO_CLASSES.length ?
                    COCO_CLASSES[classId] : "class_" + classId;
        }

        @Override
        public String toString() {
            return String.format("%s: %.2f @ [%d,%d,%d,%d]",
                    className, score, bbox.x, bbox.y, bbox.width, bbox.height);
        }
    }

    /**
     * 画像の前処理
     * @param image 入力画像 (BGR形式)
     * @return 前処理結果を含むPreprocessResultオブジェクト
     */
    private PreprocessResult preprocess(Mat image) {
        int h = image.height();
        int w = image.width();

        // アスペクト比を維持してリサイズ
        float scale = Math.min((float)inputHeight / h, (float)inputWidth / w);
        int newH = (int)(h * scale);
        int newW = (int)(w * scale);

        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(newW, newH));

        // パディング (グレー: 114)
        int padH = (inputHeight - newH) / 2;
        int padW = (inputWidth - newW) / 2;

        Mat padded = new Mat(inputHeight, inputWidth, CvType.CV_8UC3, new Scalar(114, 114, 114));
        Mat roi = padded.submat(padH, padH + newH, padW, padW + newW);
        resized.copyTo(roi);

        // 正規化とチャンネル順変更
        Mat floatMat = new Mat();
        padded.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0);

        // HWC -> CHW 変換
        List<Mat> channels = new ArrayList<>();
        Core.split(floatMat, channels);

        // 4次元テンソル [1, 3, H, W] を作成
        float[][][][] blob = new float[1][3][inputHeight][inputWidth];
        for (int c = 0; c < 3; c++) {
            float[] channelData = new float[inputHeight * inputWidth];
            channels.get(c).get(0, 0, channelData);

            for (int y = 0; y < inputHeight; y++) {
                for (int x = 0; x < inputWidth; x++) {
                    blob[0][c][y][x] = channelData[y * inputWidth + x];
                }
            }
        }

        // リソース解放
        resized.release();
        padded.release();
        floatMat.release();
        for (Mat channel : channels) {
            channel.release();
        }

        return new PreprocessResult(blob, scale, padW, padH);
    }

    /**
     * ONNX Runtime用の入力を準備
     */
    private OnnxTensor prepareInput(float[][][][] blob) throws OrtException {
        long[] shape = {1, 3, inputHeight, inputWidth};
        FloatBuffer buffer = FloatBuffer.wrap(
                new float[1 * 3 * inputHeight * inputWidth]);

        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < inputHeight; y++) {
                for (int x = 0; x < inputWidth; x++) {
                    buffer.put(blob[0][c][y][x]);
                }
            }
        }
        buffer.rewind();

        return OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, shape);
    }

    /**
     * 推論を実行
     * @param image 入力画像
     * @param confThreshold 信頼度閾値 (デフォルト: 0.25)
     * @return 検出結果のリスト
     */
    public List<DetectionResult> infer(Mat image, float confThreshold) {
        List<DetectionResult> results = new ArrayList<>();

        try {
            // 前処理
            PreprocessResult preprocessed = preprocess(image);

            // テンソル準備
            OnnxTensor inputTensor = prepareInput(preprocessed.blob);

            // 推論実行
            OrtSession.Result output = session.run(Collections.singletonMap(inputName, inputTensor));

            // 後処理
            results = postprocess(output, preprocessed.scale, preprocessed.padW,
                    preprocessed.padH, confThreshold);

            // リソース解放
            inputTensor.close();

        } catch (Exception e) {
            throw new RuntimeException("推論中にエラーが発生しました", e);
        }

        return results;
    }

    /**
     * 推論を実行 (デフォルト閾値使用)
     */
    public List<DetectionResult> infer(Mat image) {
        return infer(image, 0.25f);
    }

    /**
     * 出力の後処理
     */
    private List<DetectionResult> postprocess(OrtSession.Result output, float scale,
                                              int padW, int padH, float confThreshold)
            throws OrtException {
        List<DetectionResult> results = new ArrayList<>();

        // 出力テンソル取得 [300, 6]
        float[][][] detections = (float[][][]) output.get(0).getValue();
        float[][] detectionData = detections[0];
        for (float[] det : detectionData) {
            float x1 = det[0];
            float y1 = det[1];
            float x2 = det[2];
            float y2 = det[3];
            float score = det[4];
            int classId = (int)det[5];

            if (score < confThreshold) {
                continue;
            }

            // パディングとスケールを元に戻す
            x1 = (x1 - padW) / scale;
            y1 = (y1 - padH) / scale;
            x2 = (x2 - padW) / scale;
            y2 = (y2 - padH) / scale;

            // バウンディングボックス作成
            int x = Math.max(0, (int)x1);
            int y = Math.max(0, (int)y1);
            int width = Math.max(0, (int)(x2 - x1));
            int height = Math.max(0, (int)(y2 - y1));

            Rect bbox = new Rect(x, y, width, height);
            results.add(new DetectionResult(bbox, score, classId));
        }

        return results;
    }

    /**
     * 検出結果を画像に描画
     * @param image 元画像
     * @param detections 検出結果
     * @return 描画済み画像
     */
    public Mat drawDetections(Mat image, List<DetectionResult> detections) {
        Mat result = image.clone();

        for (DetectionResult det : detections) {
            Rect bbox = det.bbox;
            Scalar color = det.classId < COLORS.length ?
                    COLORS[det.classId] : new Scalar(0, 255, 0);

            // バウンディングボックス描画
            Imgproc.rectangle(result, bbox.tl(), bbox.br(), color, 2);

            // ラベル作成
            String label = String.format("%s: %.2f", det.className, det.score);

            // ラベル背景
            Size textSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.6, 1, null);
            Point textOrg = new Point(bbox.x, bbox.y - 10);
            Rect textBg = new Rect(
                    bbox.x, (int) (bbox.y - textSize.height - 10),
                    (int) textSize.width, (int) (textSize.height + 10)
            );

            Imgproc.rectangle(result, textBg.tl(), textBg.br(), color, -1);
            Imgproc.putText(result, label, textOrg,
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.6,
                    new Scalar(255, 255, 255), 1);
        }

        return result;
    }

    /**
     * 前処理結果を保持する内部クラス
     */
    private static class PreprocessResult {
        float[][][][] blob;
        float scale;
        int padW;
        int padH;

        PreprocessResult(float[][][][] blob, float scale, int padW, int padH) {
            this.blob = blob;
            this.scale = scale;
            this.padW = padW;
            this.padH = padH;
        }
    }

    /**
     * 使用例
     */
    public static void main(String[] args) {


        String modelPath = "yolo26.onnx";
        String imagePath = "18_094.jpg";

        try {
            // モデル読み込み
            System.out.println("モデル読み込み中: " + modelPath);
            ONNXYOLO26 detector = new ONNXYOLO26(modelPath);

            // 画像読み込み
            System.out.println("画像読み込み中: " + imagePath);
            Mat image = Imgcodecs.imread(imagePath);
            if (image.empty()) {
                System.err.println("画像が読み込めません: " + imagePath);
                return;
            }
            System.out.printf("入力画像サイズ: %dx%d%n", image.width(), image.height());

            // 推論実行
            System.out.println("推論中...");
            List<DetectionResult> detections = detector.infer(image, 0.25f);
            System.out.println("検出数: " + detections.size());

            // 結果表示
            for (DetectionResult det : detections) {
                System.out.println("  " + det);
            }

            // 結果描画
            Mat result = detector.drawDetections(image, detections);

            // 保存
            String outputPath = "result.jpg";
            Imgcodecs.imwrite(outputPath, result);
            System.out.println("\n結果を保存しました: " + outputPath);

            // リソース解放
            image.release();
            result.release();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}