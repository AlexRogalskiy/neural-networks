package org.briarheart.neuralnet;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Test in which ADALINE based neural network should predict traffic on the main avenue depending on traffic on three
 * streets leading to this avenue.
 *
 * @author Roman Chigvintsev
 */
class AdalineTest {
    @Test
    void shouldPredictTraffic() {
        /*
         * Each column represents traffic load factor (scaled by 100) in the corresponding input roads. Depending on
         * traffic load in the input roads the neural network should predict final traffic in the main output road.
         */
        double[][] trainingSet = {
                {0.98, 0.94, 0.95},
                {0.60, 0.60, 0.85},
                {0.35, 0.15, 0.15},
                {0.25, 0.30, 0.98},
                {0.75, 0.85, 0.91},
                {0.43, 0.57, 0.87},
                {0.05, 0.06, 0.01}
        };
        double[] expectedOutput = {0.80, 0.59, 0.23, 0.45, 0.74, 0.63, 0.10};

        NeuralNetwork neuralNetwork = NeuralNetwork.adalineBuilder()
                .numberOfInputs(3)
                .learningRate(0.5)
                .build();
        neuralNetwork.train(trainingSet, expectedOutput);

        double expectedError = 0.2;
        for (int i = 0; i < trainingSet.length; i++) {
            double estimatedError = Math.abs(expectedOutput[i] - neuralNetwork.feed(trainingSet[i])[0]);
            assertTrue(estimatedError <= expectedError,
                    "Estimated error <" + estimatedError + "> is greater than expected error <" + expectedError + ">");
        }
    }
}
