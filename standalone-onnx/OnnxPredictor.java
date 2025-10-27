import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.RunOptions;

/**
 * Standalone ONNX predictor relying only on local JARs.
 */
public final class OnnxPredictor {

    private static final String MODEL_OPTION = "--model";
    private static final String BATCH_OPTION = "--batch";
    private static final Path DEFAULT_MODEL_PATH = Path.of("model", "model.onnx");

    private static final String[] FEATURE_ORDER = {
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    };

    private static final Map<String, String> CLASS_LABELS = Map.of(
        "0", "setosa",
        "1", "versicolor",
        "2", "virginica"
    );

    private OnnxPredictor() {
    }

    public static void main(String[] args) throws Exception {
        ParsedInput parsedInput = parseInput(args);
        Path modelPath = resolveModelPath(parsedInput.modelPath(), parsedInput.modelExplicit());
        List<float[]> vectors = parsedInput.featureVectors();

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             OrtSession session = env.createSession(modelPath.toString(), new OrtSession.SessionOptions())) {

            String inputName = session.getInputNames().iterator().next();

            for (int i = 0; i < vectors.size(); i++) {
                float[] features = vectors.get(i);
                float[][] input = new float[][] { features };

                try (OnnxTensor tensor = OnnxTensor.createTensor(env, input);
                     RunOptions opts = new RunOptions();
                     OrtSession.Result result = session.run(Map.of(inputName, tensor), opts)) {

                    PredictionOutputs outputs = extractOutputs(result);
                    if (vectors.size() > 1) {
                        System.out.printf("=== Sample %d ===%n", i + 1);
                    }
                    System.out.println("Input features (order: sepal_length, sepal_width, petal_length, petal_width):");
                    for (int j = 0; j < FEATURE_ORDER.length; j++) {
                        System.out.printf(Locale.US, "  %-22s = %.2f%n", FEATURE_ORDER[j], features[j]);
                    }
                    System.out.println();
                    System.out.printf("Predicted class id: %d%n", outputs.predictedClass());
                    System.out.printf("Predicted class label: %s%n",
                        CLASS_LABELS.getOrDefault(String.valueOf(outputs.predictedClass()), "unknown"));
                    System.out.println();
                    System.out.println("Model loaded from: " + modelPath.toAbsolutePath());
                    System.out.println();
                    System.out.println("Class probabilities:");
                    for (int j = 0; j < outputs.probabilities().length; j++) {
                        System.out.printf(Locale.US, "  probability(%d)   : %.4f%n", j, outputs.probabilities()[j]);
                    }
                    if (i < vectors.size() - 1) {
                        System.out.println();
                    }
                }
            }
        }
    }

    private static ParsedInput parseInput(String[] args) {
        Path modelPath = DEFAULT_MODEL_PATH;
        boolean explicitModel = false;
        Path batchPath = null;
        List<String> featureArgs = new ArrayList<>();

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (MODEL_OPTION.equals(arg)) {
                if ((i + 1) >= args.length) {
                    throw new IllegalArgumentException("--model requires a file path.");
                }
                modelPath = Path.of(args[++i]);
                explicitModel = true;
            } else if (arg.startsWith(MODEL_OPTION + "=")) {
                modelPath = Path.of(arg.substring((MODEL_OPTION + "=").length()));
                explicitModel = true;
            } else if (BATCH_OPTION.equals(arg)) {
                if ((i + 1) >= args.length) {
                    throw new IllegalArgumentException("--batch requires a file path.");
                }
                batchPath = Path.of(args[++i]);
            } else if (arg.startsWith(BATCH_OPTION + "=")) {
                batchPath = Path.of(arg.substring((BATCH_OPTION + "=").length()));
            } else {
                featureArgs.add(arg);
            }
        }

        List<float[]> vectors = new ArrayList<>();
        if (!featureArgs.isEmpty()) {
            vectors.add(parseFeatures(featureArgs.toArray(String[]::new)));
        }
        if (batchPath != null) {
            vectors.addAll(readBatchFile(batchPath));
        }
        if (vectors.isEmpty()) {
            vectors.add(parseFeatures(new String[] {"5.1", "3.5", "1.4", "0.2"}));
        }

        return new ParsedInput(modelPath, vectors, explicitModel);
    }

    private static float[] parseFeatures(String[] args) {
        if (args.length != FEATURE_ORDER.length) {
            throw new IllegalArgumentException("Expected four values per sample.");
        }
        float[] values = new float[FEATURE_ORDER.length];
        for (int i = 0; i < FEATURE_ORDER.length; i++) {
            values[i] = Float.parseFloat(args[i]);
        }
        return values;
    }

    private static Path resolveModelPath(Path modelPath, boolean explicit) throws IOException {
        if (Files.exists(modelPath)) {
            return modelPath;
        }
        if (explicit || modelPath.isAbsolute()) {
            throw new IOException("Could not find ONNX model at " + modelPath.toAbsolutePath());
        }
        Path parentCandidate = Path.of("..").resolve(modelPath);
        if (Files.exists(parentCandidate)) {
            return parentCandidate;
        }
        throw new IOException("Could not find ONNX model. Checked " + modelPath.toAbsolutePath()
            + " and " + parentCandidate.toAbsolutePath());
    }

    private static List<float[]> readBatchFile(Path batchPath) {
        Path resolved = batchPath;
        if (!Files.exists(resolved)) {
            Path parentCandidate = Path.of("..").resolve(batchPath);
            if (Files.exists(parentCandidate)) {
                resolved = parentCandidate;
            }
        }
        if (!Files.exists(resolved)) {
            throw new IllegalArgumentException("Batch file not found: " + batchPath);
        }

        List<float[]> vectors = new ArrayList<>();
        try {
            for (String line : Files.readAllLines(resolved)) {
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                    continue;
                }
                String[] parts = trimmed.split("[,\\s]+");
                vectors.add(parseFeatures(parts));
            }
        } catch (IOException ex) {
            throw new IllegalStateException("Failed to read batch file: " + batchPath, ex);
        }
        return vectors;
    }

    private static PredictionOutputs extractOutputs(OrtSession.Result result) throws OrtException {
        Long predictedClass = null;
        double[] probabilities = null;

        for (Entry<String, OnnxValue> entry : result) {
            Object value = entry.getValue().getValue();
            String name = entry.getKey().toLowerCase(Locale.ROOT);

            if (predictedClass == null && name.contains("label")) {
                predictedClass = extractLabel(value);
            } else if (probabilities == null && (name.contains("prob") || name.contains("score"))) {
                probabilities = extractProbabilities(value);
            }
        }

        if (predictedClass == null) {
            throw new IllegalStateException("ONNX output did not contain a label tensor.");
        }
        if (probabilities == null) {
            probabilities = new double[0];
        }
        return new PredictionOutputs(predictedClass, probabilities);
    }

    private static long extractLabel(Object rawLabel) {
        if (rawLabel instanceof long[] array && array.length > 0) {
            return array[0];
        }
        if (rawLabel instanceof int[] array && array.length > 0) {
            return array[0];
        }
        if (rawLabel instanceof float[] array && array.length > 0) {
            return Math.round(array[0]);
        }
        if (rawLabel instanceof double[] array && array.length > 0) {
            return Math.round(array[0]);
        }
        if (rawLabel instanceof String[] array && array.length > 0) {
            String label = array[0];
            for (Entry<String, String> entry : CLASS_LABELS.entrySet()) {
                if (entry.getValue().equalsIgnoreCase(label)) {
                    return Long.parseLong(entry.getKey());
                }
            }
            return Long.parseLong(label);
        }
        throw new IllegalStateException("Unsupported label tensor type: " + rawLabel.getClass());
    }

    private static double[] extractProbabilities(Object rawProbabilities) {
        if (rawProbabilities instanceof float[][] array && array.length > 0) {
            return convertToDouble(array[0]);
        }
        if (rawProbabilities instanceof double[][] array && array.length > 0) {
            return array[0];
        }
        if (rawProbabilities instanceof float[] array) {
            return convertToDouble(array);
        }
        if (rawProbabilities instanceof double[] array) {
            return array;
        }
        if (rawProbabilities instanceof Map<?, ?> map) {
            double[] probs = new double[map.size()];
            int index = 0;
            for (Entry<?, ?> entry : map.entrySet()) {
                Object value = entry.getValue();
                if (value instanceof Number number) {
                    probs[index++] = number.doubleValue();
                }
            }
            return probs;
        }
        throw new IllegalStateException("Unsupported probability tensor type: " + rawProbabilities.getClass());
    }

    private static double[] convertToDouble(float[] array) {
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i];
        }
        return result;
    }

    private record ParsedInput(Path modelPath, List<float[]> featureVectors, boolean modelExplicit) { }

    private record PredictionOutputs(long predictedClass, double[] probabilities) { }
}
