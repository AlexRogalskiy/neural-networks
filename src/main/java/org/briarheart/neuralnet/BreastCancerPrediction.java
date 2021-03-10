package org.briarheart.neuralnet;

import lombok.extern.slf4j.Slf4j;
import org.briarheart.neuralnet.activation.ActivationFunction;
import org.briarheart.neuralnet.util.*;
import org.briarheart.neuralnet.util.chart.Chart;
import org.briarheart.neuralnet.util.math.ConfusionMatrix;
import org.briarheart.neuralnet.util.resource.ClassPathResource;

import java.io.IOException;
import java.util.Map;

/**
 * @author Roman Chigvintsev
 */
@Slf4j
public class BreastCancerPrediction {
    public static void main(String[] args) throws IOException {
        DataLoader dataLoader = new CsvDataLoader();

        double[][] trainingSet = dataLoader.load(new ClassPathResource("data/breast_cancer_inputs_training.csv"));
        double[][] expectedOutput = dataLoader.load(new ClassPathResource("data/breast_cancer_output_training.csv"));

        DataNormalizer dataNormalizer = new MinMaxDataNormalizer();
        double[][] normalizedTrainingSet = dataNormalizer.normalize(trainingSet);

        NeuralNetwork neuralNetwork = NeuralNetwork.backpropagationBuilder()
                .numberOfInputs(9)
                .numberOfOutputs(1)
                .numberOfLayers(2)
                .hiddenLayerSize(5)
                .maxEpochs(1000)
                .learningRate(0.9)
                .targetError(0.00001)
                .outputLayerActivationFunction(ActivationFunction.SIGMOID)
                .build();
        neuralNetwork.train(normalizedTrainingSet, expectedOutput);

        Map<Integer, Double> msePerEpoch = neuralNetwork.getMsePerEpoch();
        double[] mse = msePerEpoch.values().stream().mapToDouble(v -> v).toArray();
        Chart mseChart = Chart.xyLineChartBuilder()
                .title("Mean squared error per epoch")
                .dataSeries().key("MSE").values(mse).done()
                .xAxisLabel("Epoch")
                .yAxisLabel("Mean squared error")
                .build();
        mseChart.show();

        double[][] testTrainingSet = dataLoader.load(new ClassPathResource("data/breast_cancer_inputs_test.csv"));
        double[][] testExpectedOutput = dataLoader.load(new ClassPathResource("data/breast_cancer_output_test.csv"));

        double[][] normalizedTestTrainingSet = dataNormalizer.normalize(testTrainingSet);
        double[][] testEstimatedOutput = new double[normalizedTestTrainingSet.length][];
        for (int i = 0; i < normalizedTestTrainingSet.length; i++) {
            testEstimatedOutput[i] = neuralNetwork.feed(normalizedTestTrainingSet[i]);
        }
        Chart networkOutputChart = Chart.layeredBarChartBuilder()
                .title("Neural network test output")
                .dataSeries().key("Expected").values(Arrays.flatten(testExpectedOutput)).barWidth(1.5).done()
                .dataSeries().key("Estimated").values(Arrays.flatten(testEstimatedOutput)).barWidth(0.5).done()
                .xAxisLabel("# of sample")
                .yAxisLabel("Diagnosis (0 - negative, 1 - positive)")
                .build();
        networkOutputChart.show();

        double[][] confusionMatrixData = new double[testExpectedOutput.length][2];
        for (int i = 0; i < testExpectedOutput.length; i++) {
            confusionMatrixData[i] = new double[2];
            confusionMatrixData[i][0] = testExpectedOutput[i][0];
            confusionMatrixData[i][1] = testEstimatedOutput[i][0];
        }
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(confusionMatrixData, 0.6);
        log.debug("Confusion matrix:\n{}", confusionMatrix);
        log.debug("Sensitivity: {}", confusionMatrix.getSensitivity());
        log.debug("Specificity: {}", confusionMatrix.getSpecificity());
        log.debug("Accuracy: {}", confusionMatrix.getAccuracy());
    }
}
