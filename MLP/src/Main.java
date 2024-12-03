
// |------------------------ Implementing the MLP------------------------------|

import java.io.*;
import java.util.*;
import java.util.stream.*;

public class Main {
    public static void main(String[] args) {
        try {
            // Load datasets
            String dataSet1Path = "dataSet1.csv"; // Replace with your actual path
            String dataSet2Path = "dataSet2.csv"; // Replace with your actual path

            double[][] dataSet1 = DataLoader.loadDataset(dataSet1Path);
            double[][] dataSet2 = DataLoader.loadDataset(dataSet2Path);

            // Combine datasets
            double[][] combinedData = Stream.concat(Arrays.stream(dataSet1), Arrays.stream(dataSet2))
                    .toArray(double[][]::new);

            // Extract features and labels
            int[] labels = extractLabels(combinedData);
            double[][] features = normalize(extractFeatures(combinedData));

            // Debugging: Verify data
            System.out.println("Loaded dataset with " + features.length + " samples and " + features[0].length + " features.");
            System.out.println("Positive Labels: " + Arrays.stream(labels).filter(l -> l == 1).count());
            System.out.println("Negative Labels: " + Arrays.stream(labels).filter(l -> l == -1).count());

            // Create and Train MLP
            MLP mlp = new MLP(features[0].length, 64, 1, 0.01); // 64 hidden neurons, 0.01 learning rate
            TwoFoldValidation.crossValidate(features, labels, mlp);

        } catch (IOException e) {
            System.err.println("Error loading dataset: " + e.getMessage());
        }
    }

    // Helper method to extract labels (last column of dataset)
    private static int[] extractLabels(double[][] data) {
        int[] labels = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            // Binary classification: Label 1 for digit '0', -1 for all others
            labels[i] = (data[i][data[i].length - 1] == 0) ? 1 : -1;
        }
        return labels;
    }

    // Helper method to extract features (all but last column of dataset)
    private static double[][] extractFeatures(double[][] data) {
        double[][] features = new double[data.length][data[0].length - 1];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, features[i], 0, data[0].length - 1);
        }
        return features;
    }

    // Helper method to normalize features
    private static double[][] normalize(double[][] data) {
        int numFeatures = data[0].length;
        double[] min = new double[numFeatures];
        double[] max = new double[numFeatures];
        Arrays.fill(min, Double.MAX_VALUE);
        Arrays.fill(max, Double.MIN_VALUE);

        // Find min and max for each feature
        for (double[] row : data) {
            for (int i = 0; i < numFeatures; i++) {
                if (row[i] < min[i]) min[i] = row[i];
                if (row[i] > max[i]) max[i] = row[i];
            }
        }

        // Normalize each feature
        for (double[] row : data) {
            for (int i = 0; i < numFeatures; i++) {
                row[i] = (row[i] - min[i]) / (max[i] - min[i]);
            }
        }
        return data;
    }
}

class DataLoader {
    public static double[][] loadDataset(String filePath) throws IOException {
        List<double[]> data = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line;
        while ((line = br.readLine()) != null) {
            String[] tokens = line.split(",");
            double[] row = Arrays.stream(tokens).mapToDouble(Double::parseDouble).toArray();
            data.add(row);
        }
        br.close();
        return data.toArray(new double[0][0]);
    }
}

class MLP {
    private double[][] weightsInputHidden;
    private double[][] weightsHiddenOutput;
    private double[] biasHidden;
    private double[] biasOutput;
    private double learningRate;

    public MLP(int inputSize, int hiddenSize, int outputSize, double learningRate) {
        this.learningRate = learningRate;
        this.weightsInputHidden = initializeWeights(inputSize, hiddenSize);
        this.weightsHiddenOutput = initializeWeights(hiddenSize, outputSize);
        this.biasHidden = new double[hiddenSize];
        this.biasOutput = new double[outputSize];
    }

    public void train(double[][] data, int[] labels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            for (int i = 0; i < data.length; i++) {
                // Forward pass
                double[] hiddenLayer = activate(data[i], weightsInputHidden, biasHidden);
                double[] outputLayer = activate(hiddenLayer, weightsHiddenOutput, biasOutput);

                // Compute error
                double target = labels[i] == 1 ? 1.0 : 0.0; // Binary target
                double error = outputLayer[0] - target;
                totalLoss += 0.5 * error * error;

                // Backward pass
                double[] outputGradient = {error * sigmoidDerivative(outputLayer[0])};
                double[] hiddenGradient = new double[hiddenLayer.length];
                for (int j = 0; j < hiddenLayer.length; j++) {
                    hiddenGradient[j] = sigmoidDerivative(hiddenLayer[j]) * outputGradient[0] * weightsHiddenOutput[j][0];
                }

                // Update weights and biases
                for (int j = 0; j < weightsHiddenOutput.length; j++) {
                    weightsHiddenOutput[j][0] -= learningRate * outputGradient[0] * hiddenLayer[j];
                }
                for (int j = 0; j < weightsInputHidden.length; j++) {
                    for (int k = 0; k < weightsInputHidden[0].length; k++) {
                        weightsInputHidden[j][k] -= learningRate * hiddenGradient[k] * data[i][j];
                    }
                }
                for (int j = 0; j < biasHidden.length; j++) {
                    biasHidden[j] -= learningRate * hiddenGradient[j];
                }
                biasOutput[0] -= learningRate * outputGradient[0];
            }

            if (epoch % 100 == 0) {
                System.out.println("Epoch: " + epoch + ", Loss: " + totalLoss);
            }
        }
    }

    public int predict(double[] instance) {
        double[] hiddenLayer = activate(instance, weightsInputHidden, biasHidden);
        double[] outputLayer = activate(hiddenLayer, weightsHiddenOutput, biasOutput);
        return outputLayer[0] > 0.5 ? 1 : -1; // Binary classification threshold
    }

    private double[][] initializeWeights(int inputSize, int outputSize) {
        Random rand = new Random();
        double[][] weights = new double[inputSize][outputSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = rand.nextGaussian() * 0.1; // Small random initialization
            }
        }
        return weights;
    }

    private double[] activate(double[] inputs, double[][] weights, double[] bias) {
        double[] outputs = new double[weights[0].length];
        for (int j = 0; j < weights[0].length; j++) {
            outputs[j] = bias[j];
            for (int i = 0; i < inputs.length; i++) {
                outputs[j] += inputs[i] * weights[i][j];
            }
            outputs[j] = sigmoid(outputs[j]);
        }
        return outputs;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }
}

class TwoFoldValidation {
    public static void crossValidate(double[][] data, int[] labels, MLP mlp) {
        int mid = data.length / 2;

        // Split data into two halves
        double[][] train1 = Arrays.copyOfRange(data, 0, mid);
        int[] trainLabels1 = Arrays.copyOfRange(labels, 0, mid);
        double[][] test1 = Arrays.copyOfRange(data, mid, data.length);
        int[] testLabels1 = Arrays.copyOfRange(labels, mid, labels.length);

        double[][] train2 = test1;
        int[] trainLabels2 = testLabels1;
        double[][] test2 = train1;
        int[] testLabels2 = trainLabels1;

        System.out.println("Starting Two-Fold Validation...");

        // First Fold
        mlp.train(train1, trainLabels1, 1000); // Train on first split
        double accuracy1 = testModel(mlp, test1, testLabels1); // Test on second split
        System.out.println("Accuracy on First Fold: " + (accuracy1 * 100) + "%");

        // Second Fold
        mlp.train(train2, trainLabels2, 1000); // Train on second split
        double accuracy2 = testModel(mlp, test2, testLabels2); // Test on first split
        System.out.println("Accuracy on Second Fold: " + (accuracy2 * 100) + "%");

        // Final Average Accuracy
        System.out.println("Final Two-Fold Cross-Validation Accuracy: " + ((accuracy1 + accuracy2) / 2.0) * 100 + "%");
    }

    private static double testModel(MLP mlp, double[][] test, int[] labels) {
        int correct = 0;
        for (int i = 0; i < test.length; i++) {
            int predicted = mlp.predict(test[i]);
            if (predicted == labels[i]) correct++;
        }
        return (double) correct / test.length;
    }
}
