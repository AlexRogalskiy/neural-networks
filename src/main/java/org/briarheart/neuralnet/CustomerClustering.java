package org.briarheart.neuralnet;

import lombok.extern.slf4j.Slf4j;
import org.briarheart.neuralnet.util.CsvDataLoader;
import org.briarheart.neuralnet.util.DataLoader;
import org.briarheart.neuralnet.util.DataNormalizer;
import org.briarheart.neuralnet.util.MinMaxEqualizedDataNormalizer;
import org.briarheart.neuralnet.util.math.ConfusionMatrix;
import org.briarheart.neuralnet.util.resource.ClassPathResource;

import java.io.IOException;

/**
 * @author Roman Chigvintsev
 */
@Slf4j
public class CustomerClustering {
    public static void main(String[] args) throws IOException {
        DataLoader dataLoader = new CsvDataLoader();
        double[][] trainingSet = dataLoader.load(new ClassPathResource("data/card_inputs_training.csv"));

        DataNormalizer dataNormalizer = new MinMaxEqualizedDataNormalizer();
        trainingSet = dataNormalizer.normalize(trainingSet);

        NeuralNetwork neuralNetwork = NeuralNetwork.kohonenBuilder()
                .numberOfInputs(10)
                .numberOfOutputs(2)
                .maxEpochs(100)
                .learningRate(0.1)
                .build();
        neuralNetwork.train(trainingSet);

        double[][] testTrainingSet = dataLoader.load(new ClassPathResource("data/card_inputs_test.csv"));
        testTrainingSet = dataNormalizer.normalize(testTrainingSet);

        double[][] testExpectedOutput = dataLoader.load(new ClassPathResource("data/card_output_test.csv"));
        double[][] testEstimatedOutput = new double[testTrainingSet.length][];
        for (int i = 0; i < testTrainingSet.length; i++) {
            testEstimatedOutput[i] = neuralNetwork.feed(testTrainingSet[i]);
        }

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
