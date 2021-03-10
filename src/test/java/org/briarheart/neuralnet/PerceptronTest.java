package org.briarheart.neuralnet;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Test of modelling of a basic alarm system that uses perceptron to represent simple "AND" logic. There are two
 * detectors, and the rules of triggering an alarm are as follows:
 * <p>
 *     <li>if both or one of the detectors is disabled, the alarm is not triggered;</li>
 *     <li>if both of the detectors are enabled, the alarm is triggered.</li>
 * </p>
 * To encode the problem, inputs are represented as follows: "0" means detector is disabled, and "1" means detector is
 * enabled. Output is represented as follows: "0" means alarm is disabled, and "1" means alarm is enabled.
 *
 * @author Roman Chigvintsev
 */
class PerceptronTest {
    @Test
    void shouldPredictAlarmSignal() {
        /*
         * First column represents signal from first detector and second column represents signal from second detector.
         * Value "0.0" means no signal. Output detector should send signal only when there are signals from two other
         * detectors.
         */
        double[][] trainingSet = {
                {0.0, 0.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 1.0}
        };
        double[] expectedOutput = {0.0, 0.0, 0.0, 1.0};

        NeuralNetwork neuralNetwork = NeuralNetwork.perceptronBuilder()
                .numberOfInputs(2)
                .build();
        neuralNetwork.train(trainingSet, expectedOutput);

        for (int i = 0; i < trainingSet.length; i++) {
            assertEquals(expectedOutput[i], neuralNetwork.feed(trainingSet[i])[0]);
        }
    }
}
