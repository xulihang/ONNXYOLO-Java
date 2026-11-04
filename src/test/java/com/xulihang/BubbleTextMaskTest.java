package com.xulihang;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.List;

/**
 * 气泡文字覆盖测试: 调用 ONNXRFDETRSeg.getBubbleTextMask() 把气泡内部涂白, 检查抹字效果.
 * 用法: BubbleTextMaskTest [模型路径] [图片路径] [输出路径]
 */
public class BubbleTextMaskTest {

    public static void main(String[] args) throws Exception {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String modelPath = args.length > 0 ? args[0] : "koharu/koharu_layout_rfdetr_seg_2xl_1152_int8.onnx";
        String imagePath = args.length > 1 ? args[1] : "18_094.jpg";
        String outputPath = args.length > 2 ? args[2] : "18_094_bubble_text_covered.jpg";

        Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            System.err.println("无法读取图片: " + imagePath);
            return;
        }

        ONNXRFDETRSeg detector = new ONNXRFDETRSeg(modelPath);
        try {
            List<ONNXRFDETRSeg.DetectionResult> detections = detector.detect(image);
            System.out.printf("输入图片: %dx%d, 检测数量: %d%n", image.cols(), image.rows(), detections.size());

            // 第一次调用才生成掩膜 (平滑 + 内缩)
            long start = System.nanoTime();
            Mat bubbleMask = detector.getBubbleTextMask();
            double elapsedMs = (System.nanoTime() - start) / 1e6;
            if (bubbleMask == null || bubbleMask.empty()) {
                System.err.println("没有得到气泡掩膜 (模型无 mask 输出, 或未检测到气泡)");
                return;
            }
            System.out.printf("气泡文字覆盖掩膜: %dx%d, 前景 %d px, 生成耗时 %.1f ms%n",
                    bubbleMask.cols(), bubbleMask.rows(), Core.countNonZero(bubbleMask), elapsedMs);

            Mat covered = image.clone();
            covered.setTo(new Scalar(255, 255, 255), bubbleMask);
            if (!Imgcodecs.imwrite(outputPath, covered)) {
                System.err.println("结果保存失败: " + outputPath);
            } else {
                System.out.println("结果保存: " + outputPath);
            }
            covered.release();

            String maskPath = outputPath.replaceAll("\\.[^.]+$", "") + "_mask.png";
            Imgcodecs.imwrite(maskPath, bubbleMask);
            System.out.println("掩膜保存: " + maskPath);

            for (ONNXRFDETRSeg.DetectionResult det : detections) {
                det.release();
            }
        } finally {
            detector.close();
            image.release();
        }
    }
}
