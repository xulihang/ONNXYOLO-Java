package com.xulihang;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.List;

/**
 * 长时间循环 detect() 检查原生内存是否泄漏.
 * 用 -Xmx 限制 Java 堆, 跑完 warmup 后记录 RSS 增量即可判断堆外泄漏.
 * 用法: RFDETRSegLeakCheck [model] [image] [iterations=10] [warmup=1]
 */
public class RFDETRSegLeakCheck {

    public static void main(String[] args) throws Exception {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String modelPath = args.length > 0 ? args[0] : "koharu/koharu_layout_rfdetr_seg_2xl_1152_int8.onnx";
        String imagePath = args.length > 1 ? args[1] : "003_0101.jpg";
        int iterations = args.length > 2 ? Integer.parseInt(args[2]) : 10;
        int warmup = args.length > 3 ? Integer.parseInt(args[3]) : 1;

        Mat image = Imgcodecs.imread(imagePath);
        System.out.printf("模型: %s%n图片: %s %dx%d%n迭代: %d (warmup %d)%n",
                modelPath, imagePath, image.width(), image.height(), iterations, warmup);

        ONNXRFDETRSeg detector = new ONNXRFDETRSeg(modelPath);
        long rssAfterWarmup = 0;
        long[] checkpoints = new long[iterations];
        for (int i = 0; i < iterations; i++) {
            List<ONNXRFDETRSeg.DetectionResult> dets = detector.detect(image);
            // 同时走一遍气泡掩膜路径, 它有自己的缓存
            detector.getBubbleTextMask();
            for (ONNXRFDETRSeg.DetectionResult d : dets) d.release();
            if (i == warmup) rssAfterWarmup = rssKb();
            checkpoints[i] = rssKb();
            if (i % Math.max(1, iterations / 10) == 0) {
                System.out.printf("  iter %4d  RSS %6d MB  heap %5d MB%n",
                        i, rssKb() / 1024, usedHeapMb());
            }
        }
        long rssEnd = rssKb();
        System.out.printf("RSS: warmup 后 %d MB, 结束 %d MB, 增长 %+d MB (%.2f MB/次)%n",
                rssAfterWarmup / 1024, rssEnd / 1024, (rssEnd - rssAfterWarmup) / 1024,
                (rssEnd - rssAfterWarmup) / 1024.0 / Math.max(1, iterations - warmup));
        System.out.printf("峰值 RSS %d MB%n", java.util.Arrays.stream(checkpoints).max().orElse(0) / 1024);

        image.release();
        detector.close();
    }

    private static long usedHeapMb() {
        return (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024);
    }

    // 通过 tasklist 读本进程 Working Set (KB); JDK API 不暴露 RSS
    private static long rssKb() {
        try {
            long pid = ProcessHandle.current().pid();
            Process p = new ProcessBuilder("tasklist", "/FI", "PID eq " + pid, "/FO", "CSV", "/NH")
                    .redirectErrorStream(true).start();
            String line = new String(p.getInputStream().readAllBytes()).trim();
            p.waitFor();
            // "javaw.exe","1234","Console","1","1,234,567 K"
            String[] fields = line.split("\",\"");
            String mem = fields[fields.length - 1].replaceAll("[^0-9]", "");
            return Long.parseLong(mem);
        } catch (Exception e) {
            return -1;
        }
    }
}
