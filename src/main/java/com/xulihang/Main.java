package com.xulihang;

import ai.onnxruntime.OrtException;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.List;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) throws OrtException {
        // 加载OpenCV库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        // 1. 创建检测器

        ONNXYOLO detector = new ONNXYOLO("model.onnx");
        Mat image = Imgcodecs.imread("image.jpg");
// 2. 检测图像
        List<ONNXYOLO.DetectionResult> results = detector.detect(image);
        System.out.println("数量: " + results.size());
// 3. 处理结果
        for (ONNXYOLO.DetectionResult result : results) {
            System.out.println(result);
        }
// 5. 关闭检测器
        detector.close();
    }
}