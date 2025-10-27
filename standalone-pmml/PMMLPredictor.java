import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.Computable;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.InputField;
import org.jpmml.evaluator.ModelEvaluatorBuilder;
import org.jpmml.evaluator.OutputField;
import org.jpmml.evaluator.TargetField;
import org.jpmml.model.PMMLUtil;

/**
 * Standalone version of the PMML predictor that relies solely on local JAR files.
 *
 * Compilation example:
 *   javac -cp "libs/*" PMMLPredictor.java
 * Execution example:
 *   java -cp ".:libs/*" PMMLPredictor
 *   java -cp ".:libs/*" PMMLPredictor 6.1 2.8 4.7 1.2
 */
public final class PMMLPredictor {

    private static final String MODEL_OPTION = "--model";
    private static final Path DEFAULT_MODEL_PATH = Path.of("model", "model.pmml");

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

    private PMMLPredictor() {
        // static utility class
    }

    public static void main(String[] args) throws Exception {
        ParsedInput parsedInput = parseInput(args);
        Path resolvedModelPath = resolveModelPath(parsedInput.modelPath(), parsedInput.modelExplicit());

        Evaluator evaluator = loadEvaluator(resolvedModelPath);
        Map<String, Object> featureValues = parseArguments(parsedInput.featureArgs());

        Map<FieldName, FieldValue> arguments = prepareArguments(evaluator, featureValues);
        Map<FieldName, ?> results = evaluator.evaluate(arguments);

        TargetField targetField = evaluator.getTargetFields().get(0);
        FieldName targetFieldName = targetField.getName();
        Object predictedValue = unwrapComputable(results.get(targetFieldName));

        System.out.println("Input features (order: sepal_length, sepal_width, petal_length, petal_width):");
        Arrays.stream(FEATURE_ORDER).forEach(name ->
            System.out.printf(Locale.US, "  %-22s = %.2f%n", name, ((Number) featureValues.get(name)).doubleValue())
        );
        System.out.println();

        System.out.printf("Predicted class id: %s%n", predictedValue);
        System.out.printf("Predicted class label: %s%n", CLASS_LABELS.getOrDefault(String.valueOf(predictedValue), "unknown"));

        System.out.println();
        System.out.println("Model loaded from: " + resolvedModelPath.toAbsolutePath());
        System.out.println();
        System.out.println("Class probabilities:");
        printProbabilities(results, evaluator.getOutputFields());
    }

    private static ParsedInput parseInput(String[] args) {
        Path modelPath = DEFAULT_MODEL_PATH;
        boolean explicitModelPath = false;
        List<String> featureArgs = new ArrayList<>();

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (MODEL_OPTION.equals(arg)) {
                if ((i + 1) >= args.length) {
                    throw new IllegalArgumentException("--model requires a file path argument.");
                }
                modelPath = Path.of(args[++i]);
                explicitModelPath = true;
            } else if (arg.startsWith(MODEL_OPTION + "=")) {
                modelPath = Path.of(arg.substring((MODEL_OPTION + "=").length()));
                explicitModelPath = true;
            } else {
                featureArgs.add(arg);
            }
        }

        return new ParsedInput(modelPath, featureArgs.toArray(String[]::new), explicitModelPath);
    }

    private static Path resolveModelPath(Path modelPath, boolean explicitModelPath) throws IOException {
        if (Files.exists(modelPath)) {
            return modelPath;
        }

        if (explicitModelPath || modelPath.isAbsolute()) {
            throw new IOException("PMML model file not found: " + modelPath.toAbsolutePath());
        }

        Path parentCandidate = Path.of("..").resolve(modelPath);
        if (Files.exists(parentCandidate)) {
            return parentCandidate;
        }

        throw new IOException(
            "PMML model file not found. Checked: " + modelPath.toAbsolutePath() + " and " + parentCandidate.toAbsolutePath());
    }

    private static Evaluator loadEvaluator(Path resolvedModelPath) throws Exception {

        PMML pmml;
        try (InputStream is = new FileInputStream(resolvedModelPath.toFile())) {
            pmml = PMMLUtil.unmarshal(is);
        }

        Evaluator evaluator = new ModelEvaluatorBuilder(pmml)
            .build();
        evaluator.verify();
        return evaluator;
    }

    private static Map<String, Object> parseArguments(String[] args) {
        Map<String, Object> values = new LinkedHashMap<>();

        if (args.length != 0 && args.length != FEATURE_ORDER.length) {
            throw new IllegalArgumentException(
                "Provide exactly four values (sepal_length sepal_width petal_length petal_width) or no values to use defaults.");
        }

        if (args.length == 0) {
            values.put("sepal length (cm)", 5.1d);
            values.put("sepal width (cm)", 3.5d);
            values.put("petal length (cm)", 1.4d);
            values.put("petal width (cm)", 0.2d);
            return values;
        }

        for (int i = 0; i < FEATURE_ORDER.length; i++) {
            double parsedValue = Double.parseDouble(args[i]);
            values.put(FEATURE_ORDER[i], parsedValue);
        }
        return values;
    }

    private static Map<FieldName, FieldValue> prepareArguments(Evaluator evaluator, Map<String, Object> featureValues) {
        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();
        for (InputField inputField : evaluator.getInputFields()) {
            FieldName fieldName = inputField.getName();
            Object rawValue = featureValues.get(fieldName.getValue());
            if (rawValue == null) {
                throw new IllegalArgumentException("Missing value for required field: " + fieldName.getValue());
            }
            arguments.put(fieldName, inputField.prepare(rawValue));
        }
        return arguments;
    }

    private static void printProbabilities(Map<FieldName, ?> results, List<OutputField> outputFields) {
        DecimalFormat formatter = new DecimalFormat("0.0000", DecimalFormatSymbols.getInstance(Locale.US));

        for (OutputField outputField : outputFields) {
            FieldName name = outputField.getName();
            Object computedValue = unwrapComputable(results.get(name));

            if (computedValue instanceof Number number) {
                String formatted = formatter.format(number.doubleValue());
                System.out.printf(Locale.US, "  %-16s : %s%n", name.getValue(), formatted);
            } else if (computedValue instanceof Map<?, ?> map) {
                for (Map.Entry<?, ?> entry : map.entrySet()) {
                    Object entryValue = entry.getValue();
                    if (entryValue instanceof Number numberValue) {
                        String formatted = formatter.format(numberValue.doubleValue());
                        System.out.printf(Locale.US, "  %-16s : %s%n", entry.getKey(), formatted);
                    }
                }
            } else if (computedValue != null) {
                System.out.printf(Locale.US, "  %-16s : %s%n", name.getValue(), computedValue);
            }
        }
    }

    private static Object unwrapComputable(Object value) {
        if (value instanceof Computable computable) {
            return computable.getResult();
        }
        return value;
    }

    private record ParsedInput(Path modelPath, String[] featureArgs, boolean modelExplicit) {
    }
}
