package org.briarheart.neuralnet;

import org.briarheart.neuralnet.util.CsvDataLoader;
import org.briarheart.neuralnet.util.DataLoader;
import org.briarheart.neuralnet.util.DataNormalizer;
import org.briarheart.neuralnet.util.MinMaxEqualizedDataNormalizer;
import org.briarheart.neuralnet.util.resource.ClassPathResource;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * Kohonen based tests.
 *
 * @author Roman Chigvintsev
 */
public class KohonenTest {
    /**
     * In this test we have some animals and three of their characteristics are: has pelage? (yes/no), is terrestrial?
     * (yes/no), and has mammary glands? (yes/no). Neural network's goal is to cluster the animals in two different
     * groups: mammals and not mammals.
     */
    @Test
    void shouldClusterAnimals() {
        /*
         * First column indicates whether an animal has pelage, second column indicates whether an animal is
         * terrestrial or not, finally third column indicates whether an animal has mammary glands. Every
         * characteristic is encoded by two numbers: 1.0 means "yes", and -1.0 means "no".
         */
        double[][] trainingSet = {
                {1.0, -1.0, 1.0},
                {-1.0, -1.0, -1.0},
                {-1.0, -1.0, 1.0},
                {1.0, 1.0, -1.0},
                {-1.0, 1.0, 1.0},
                {1.0, -1.0, -1.0}
        };
        NeuralNetwork neuralNetwork = NeuralNetwork.kohonenBuilder()
                .numberOfInputs(3)
                .numberOfOutputs(2)
                .maxEpochs(10)
                .learningRate(0.1)
                .build();
        neuralNetwork.train(trainingSet);

        double[][] validationSet = {{-1.0, 1.0, -1.0}, {1.0, 1.0, 1.0}};
        double[][] expectedOutput = {{-1.0, 1.0}, {1.0, -1.0}};
        for (int i = 0; i < validationSet.length; i++) {
            assertArrayEquals(expectedOutput[i], neuralNetwork.feed(validationSet[i]));
        }
    }

    /**
     * Goal of this test is to find two groups of customers that either share similar characteristics or buy the same
     * products.
     */
    @Test
    void shouldClusterCustomers() throws IOException {
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
        int misses = 0;
        for (int i = 0; i < testTrainingSet.length; i++) {
            if (!Arrays.equals(testExpectedOutput[i], neuralNetwork.feed(testTrainingSet[i]))) {
                misses++;
            }
        }

        double expectedErrorPercentage = 8.6;
        if (misses > 0) {
            double errorPercentage = misses / (testTrainingSet.length / 100.0);
            String message = "Estimated error percentage <" + errorPercentage + "> is greater than " +
                    "expected error percentage <" + expectedErrorPercentage + ">";
            assertFalse(Double.compare(errorPercentage, expectedErrorPercentage) > 0, message);
        }
    }
}
