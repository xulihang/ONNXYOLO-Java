package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.Core;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.Mat;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 对比不同 RF-DETR Seg 模型 (1152 原版 / 768 速度优化版) 的推理耗时与检测结果.
 * 用法: RFDETRSegBenchmark [warmup=3] [runs=10] model1=path1.jpg model2=path2 ...
 */
public class RFDETRSegBenchmark {

    public static void main(String[] args) throws Exception {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        int warmup = Integer.parseInt(args.length > 0 ? args[0] : "3");
        int runs = Integer.parseInt(args.length > 1 ? args[1] : "10");
        List<String[]> configs = new ArrayList<>();
        for (int i = 2; i < args.length; i++) {
            int eq = args[i].indexOf('=');
            configs.add(new String[]{args[i].substring(0, eq), args[i].substring(eq + 1)});
        }
        if (configs.isEmpty()) {
            configs.add(new String[]{"koharu/koharu_layout_rfdetr_seg_2xl_1152_int8.onnx", "003_0101.jpg"});
            configs.add(new String[]{"koharu/rfdetr-seg-2xlarge.onnx", "003_0101.jpg"});
        }

        Map<String, List<String>> results = new LinkedHashMap<>();
        for (String[] config : configs) {
            String modelPath = config[0];
            Mat image = Imgcodecs.imread(config[1]);
            System.out.printf("%n=== %s  (image %dx%d)%n", modelPath, image.width(), image.height());

            ONNXRFDETRSeg detector = new ONNXRFDETRSeg(modelPath);
            printModelInfo(modelPath);

            double[] times = new double[warmup + runs];
            List<ONNXRFDETRSeg.DetectionResult> detections = null;
            for (int i = 0; i < times.length; i++) {
                long start = System.nanoTime();
                List<ONNXRFDETRSeg.DetectionResult> dets = detector.detect(image);
                times[i] = (System.nanoTime() - start) / 1e6;
                if (detections != null) detections.forEach(ONNXRFDETRSeg.DetectionResult::release);
                detections = dets;
            }

            double[] measured = Arrays.copyOfRange(times, warmup, times.length);
            Arrays.sort(measured);
            System.out.printf("耗时: 首次 %.1f ms | 中位数 %.1f ms | 平均 %.1f ms | 最小 %.1f ms | 最大 %.1f ms  (%d 次, warmup %d)%n",
                    times[0], measured[measured.length / 2], Arrays.stream(measured).average().orElse(0),
                    measured[0], measured[measured.length - 1], runs, warmup);
            System.out.printf("检测数量: %d%n", detections.size());

            List<String> lines = new ArrayList<>();
            Map<Integer, Integer> counts = new TreeMap<>();
            for (ONNXRFDETRSeg.DetectionResult d : detections) {
                counts.merge(d.getClassId(), 1, Integer::sum);
                float[] b = d.getBbox();
                lines.add(String.format("%s %.3f [%.0f, %.0f, %.0f, %.0f]",
                        d.getClassName(), d.getConfidence(), b[0], b[1], b[2], b[3]));
            }
            System.out.println("分类计数: " + counts.entrySet().stream()
                    .map(e -> e.getKey() + "=" + e.getValue()).collect(Collectors.joining(", ")));
            Collections.sort(lines);
            results.put(modelPath, lines);

            detections.forEach(ONNXRFDETRSeg.DetectionResult::release);
            image.release();
            detector.close();
        }

        if (results.size() == 2) {
            List<String> a = results.values().toArray(new List[0])[0];
            List<String> b = results.values().toArray(new List[0])[1];
            System.out.println("\n=== 检测结果差异 (按 类别+坐标浮点文本) ===");
            Set<String> setA = new LinkedHashSet<>(a);
            Set<String> setB = new LinkedHashSet<>(b);
            Set<String> common = new HashSet<>(setA);
            common.retainAll(setB);
            System.out.printf("相同: %d / A:%d B:%d%n", common.size(), a.size(), b.size());
            for (String s : setA) if (!setB.contains(s)) System.out.println("  A only: " + s);
            for (String s : setB) if (!setA.contains(s)) System.out.println("  B only: " + s);
        }
    }

    private static void printModelInfo(String path) throws OrtException {
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             OrtSession session = env.createSession(path)) {
            for (Map.Entry<String, NodeInfo> e : session.getInputInfo().entrySet()) {
                System.out.println("input  " + e.getKey() + " " + Arrays.toString(shape(e.getValue())));
            }
            for (Map.Entry<String, NodeInfo> e : session.getOutputInfo().entrySet()) {
                System.out.println("output " + e.getKey() + " " + Arrays.toString(shape(e.getValue())));
            }
        }
    }

    private static long[] shape(NodeInfo info) {
        return ((TensorInfo) info.getInfo()).getShape();
    }
}
