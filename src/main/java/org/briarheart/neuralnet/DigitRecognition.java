package org.briarheart.neuralnet;

import lombok.extern.slf4j.Slf4j;
import org.briarheart.neuralnet.activation.ActivationFunction;
import org.briarheart.neuralnet.util.CsvDataLoader;
import org.briarheart.neuralnet.util.DataLoader;
import org.briarheart.neuralnet.util.chart.Chart;
import org.briarheart.neuralnet.util.resource.ClassPathResource;

import java.io.IOException;
import java.util.Map;

/**
 * @author Roman Chigvintsev
 */
@Slf4j
public class DigitRecognition {
    public static void main(String[] args) throws IOException {
        DataLoader dataLoader = new CsvDataLoader();

        double[][] trainingSet = dataLoader.load(new ClassPathResource("data/ocr_traning_inputs.csv"));
        double[][] expectedOutput = dataLoader.load(new ClassPathResource("data/ocr_traning_outputs.csv"));

        NeuralNetwork neuralNetwork = NeuralNetwork.onlineBackpropagationBuilder()
                .numberOfInputs(25)
                .numberOfOutputs(10)
                .numberOfLayers(2)
                .hiddenLayerSize(18)
                .maxEpochs(6000)
                .learningRate(0.7)
                .learningRateReductionPercentage(0.01)
                .targetError(0.00001)
                .outputLayerActivationFunction(ActivationFunction.SIGMOID)
                .build();
        neuralNetwork.train(trainingSet, expectedOutput);

        Map<Integer, Double> msePerEpoch = neuralNetwork.getMsePerEpoch();
        double[] mse = msePerEpoch.values().stream().mapToDouble(v -> v).toArray();
        Chart mseChart = Chart.xyLineChartBuilder()
                .title("Mean squared error per epoch")
                .dataSeries().key("MSE").values(mse).done()
                .xAxisLabel("Epoch")
                .yAxisLabel("Mean squared error")
                .build();
        mseChart.show();
    }
}
