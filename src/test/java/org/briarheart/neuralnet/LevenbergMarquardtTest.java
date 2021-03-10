package org.briarheart.neuralnet;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Levenberg-Marquardt based test of prediction if some person will be able to enter university depending on his/her
 * gender and grade.
 *
 * @author Roman Chigvintsev
 */
public class LevenbergMarquardtTest {
    @Test
    void shouldPredictEnrollmentStatus() {
        /*
         * First column represents person's gender (one means female, and zero means male), and second column
         * represents person's grade scaled by 100. Enrollment status is encoded by two numbers ([1, 0] means
         * performed enrollment, [0, 1] means waiver enrollment).
         */
        double[][] trainingSet = {
                {1.0, 0.73},
                {1.0, 0.81},
                {1.0, 0.86},
                {1.0, 0.95},
                {0.0, 0.45},
                {1.0, 0.70},
                {0.0, 0.51},
                {1.0, 0.89},
                {1.0, 0.79},
                {0.0, 0.54}
        };
        double[][] expectedOutput = {
                {1.0, 0.0},
                {1.0, 0.0},
                {1.0, 0.0},
                {1.0, 0.0},
                {1.0, 0.0},
                {0.0, 1.0},
                {0.0, 1.0},
                {0.0, 1.0},
                {0.0, 1.0},
                {0.0, 1.0}
        };
        NeuralNetwork neuralNetwork = NeuralNetwork.levenbergMarquardtBuilder()
                .numberOfInputs(2)
                .numberOfOutputs(2)
                .numberOfLayers(2)
                .hiddenLayerSize(3)
                .maxEpochs(1000)
                .learningRate(0.1)
                .targetError(0.002)
                .build();
        neuralNetwork.train(trainingSet, expectedOutput);

        double expectedMeanError = 1.4;
        for (int i = 0; i < trainingSet.length; i++) {
            double[] estimatedOutput = neuralNetwork.feed(trainingSet[i]);
            double meanError = calculateMeanError(expectedOutput[i], estimatedOutput);
            assertTrue(meanError <= expectedMeanError,
                    "Estimated mean error <" + meanError + "> is greater than expected mean error <"
                            + expectedMeanError + ">");
        }
    }

    private double calculateMeanError(double[] estimatedOutput, double[] expectedOutput) {
        double errorSum = 0.0;
        for (int i = 0; i < estimatedOutput.length; i++) {
            double error = expectedOutput[i] - estimatedOutput[i];
            errorSum += Math.pow(error, 2.0);
        }
        return errorSum / estimatedOutput.length;
    }
}
