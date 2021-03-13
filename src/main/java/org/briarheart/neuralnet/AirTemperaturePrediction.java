package org.briarheart.neuralnet;

import org.briarheart.neuralnet.util.*;
import org.briarheart.neuralnet.util.chart.Chart;
import org.briarheart.neuralnet.util.resource.ClassPathResource;

import java.awt.*;
import java.io.IOException;
import java.util.Map;

/**
 * @author Roman Chigvintsev
 */
public class AirTemperaturePrediction {
    private static final double ERROR_MARGIN = 1.0;

    public static void main(String[] args) throws IOException {
        DataLoader dataLoader = new CsvDataLoader();
        double[][] trainingSet = dataLoader.load(new ClassPathResource("data/inmet_13_14_input.csv"));
        double[][] expectedOutput = dataLoader.load(new ClassPathResource("data/inmet_13_14_output.csv"));

        DataNormalizer dataNormalizer = new MinMaxEqualizedDataNormalizer();
        double[][] normalizedTrainingSet = dataNormalizer.normalize(trainingSet);
        double[][] normalizedExpectedOutput = dataNormalizer.normalize(expectedOutput);

        NeuralNetwork neuralNetwork = NeuralNetwork.backpropagationBuilder()
                .numberOfInputs(4)
                .numberOfOutputs(1)
                .numberOfLayers(2)
                .hiddenLayerSize(4)
                .maxEpochs(1000)
                .learningRate(0.5)
                .targetError(0.00001)
                .build();
        neuralNetwork.train(normalizedTrainingSet, normalizedExpectedOutput);

        Map<Integer, Double> msePerEpoch = neuralNetwork.getMsePerEpoch();
        double[] mse = msePerEpoch.values().stream().mapToDouble(v -> v).toArray();
        Chart mseChart = Chart.xyLineChartBuilder()
                .title("Mean squared error per epoch")
                .dataSeries().key("MSE").values(mse).done()
                .xAxisLabel("Epoch")
                .yAxisLabel("Mean squared error")
                .build();
        mseChart.show();

        double[][] estimatedOutput = new double[normalizedTrainingSet.length][];
        for (int i = 0; i < normalizedTrainingSet.length; i++) {
            estimatedOutput[i] = neuralNetwork.feed(normalizedTrainingSet[i]);
        }
        double[][] denormalizedEstimatedOutput = dataNormalizer.denormalize(expectedOutput, estimatedOutput);
        double[] lowerErrorMargins = new double[denormalizedEstimatedOutput.length];
        double[] upperErrorMargins = new double[denormalizedEstimatedOutput.length];
        for (int i = 0; i < denormalizedEstimatedOutput.length; i++) {
            lowerErrorMargins[i] = denormalizedEstimatedOutput[i][0] - ERROR_MARGIN;
            upperErrorMargins[i] = denormalizedEstimatedOutput[i][0] + ERROR_MARGIN;
        }
        Chart networkOutputChart = Chart.xyLineChartBuilder()
                .title("Neural network output")
                .dataSeries().key("Expected").values(Arrays.flatten(expectedOutput)).done()
                .dataSeries().key("Estimated").values(Arrays.flatten(denormalizedEstimatedOutput)).done()
                .dataSeries()
                    .key("Lower error margin")
                    .values(lowerErrorMargins)
                    .color(Color.BLACK)
                    .strokeWidth(2.0f)
                    .dashedStroke(true)
                    .done()
                .dataSeries()
                    .key("Upper error margin")
                    .values(upperErrorMargins)
                    .color(Color.BLACK)
                    .strokeWidth(2.0f)
                    .dashedStroke(true)
                    .done()
                .xAxisLabel("# of sample")
                .yAxisLabel("Temperature (Celsius)")
                .axisRange(22, 30)
                .build();
        networkOutputChart.show();

        double[][] testTrainingSet = dataLoader.load(new ClassPathResource("data/inmet_13_14_input_test.csv"));
        double[][] testExpectedOutput = dataLoader.load(new ClassPathResource("data/inmet_13_14_output_test.csv"));

        double[][] normalizedTestTrainingSet = dataNormalizer.normalize(testTrainingSet);
        estimatedOutput = new double[normalizedTestTrainingSet.length][];
        for (int i = 0; i < normalizedTestTrainingSet.length; i++) {
            estimatedOutput[i] = neuralNetwork.feed(normalizedTestTrainingSet[i]);
        }
        denormalizedEstimatedOutput = dataNormalizer.denormalize(testExpectedOutput, estimatedOutput);
        lowerErrorMargins = new double[denormalizedEstimatedOutput.length];
        upperErrorMargins = new double[denormalizedEstimatedOutput.length];
        for (int i = 0; i < denormalizedEstimatedOutput.length; i++) {
            lowerErrorMargins[i] = denormalizedEstimatedOutput[i][0] - ERROR_MARGIN;
            upperErrorMargins[i] = denormalizedEstimatedOutput[i][0] + ERROR_MARGIN;
        }
        networkOutputChart = Chart.xyLineChartBuilder()
                .title("Neural network test output")
                .dataSeries().key("Expected").values(Arrays.flatten(testExpectedOutput)).done()
                .dataSeries().key("Estimated").values(Arrays.flatten(denormalizedEstimatedOutput)).done()
                .dataSeries()
                    .key("Lower error margin")
                    .values(lowerErrorMargins)
                    .color(Color.BLACK)
                    .strokeWidth(2.0f)
                    .dashedStroke(true)
                    .done()
                .dataSeries()
                    .key("Upper error margin")
                    .values(upperErrorMargins)
                    .color(Color.BLACK)
                    .strokeWidth(2.0f)
                    .dashedStroke(true)
                    .done()
                .xAxisLabel("# of sample")
                .yAxisLabel("Temperature (Celsius)")
                .axisRange(22, 30)
                .build();
        networkOutputChart.show();
    }
}
