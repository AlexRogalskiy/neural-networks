package org.briarheart.neuralnet;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * In this test we have some animals and three of their characteristics are: has pelage? (yes/no), is terrestrial?
 * (yes/no), and has mammary glands? (yes/no). Neural network's goal is to cluster the animals in two different groups:
 * mammals and not mammals.
 *
 * @author Roman Chigvintsev
 */
public class KohonenTest {
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
}
