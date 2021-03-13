package org.briarheart.neuralnet;

import org.briarheart.neuralnet.activation.ActivationFunction;
import org.briarheart.neuralnet.util.*;
import org.briarheart.neuralnet.util.resource.ClassPathResource;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Backpropagation based tests.
 *
 * @author Roman Chigvintsev
 */
class BackpropagationTest {
    /**
     * Test of prediction if some person will be able to enter university depending on his/her gender and grade.
     */
    @Test
    void shouldPredictUniversityEnrollmentStatus() {
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
        NeuralNetwork neuralNetwork = NeuralNetwork.backpropagationBuilder()
                .numberOfInputs(2)
                .numberOfOutputs(2)
                .numberOfLayers(2)
                .hiddenLayerSize(3)
                .maxEpochs(1000)
                .learningRate(0.1)
                .targetError(0.002)
                .build();
        neuralNetwork.train(trainingSet, expectedOutput);

        double expectedMeanError = 0.67;
        for (int i = 0; i < trainingSet.length; i++) {
            double[] estimatedOutput = neuralNetwork.feed(trainingSet[i]);
            double meanError = calculateMeanError(expectedOutput[i], estimatedOutput);
            assertTrue(meanError <= expectedMeanError,
                    "Estimated mean error <" + meanError + "> is greater than expected mean error <"
                            + expectedMeanError + ">");
        }
    }

    /**
     * Test of air temperature prediction based on precipitation (accumulation of daily rain), insolation (count of
     * hours receiving sun radiation), mean humidity (average of hourly measurement), and mean wind speed (average of
     * hourly measurement).
     */
    @Test
    void shouldPredictAirTemperature() throws IOException {
        DataLoader dataLoader = new CsvDataLoader();
        /*
         * First column represents precipitation (0 - 161.2 mm), second column represents insolation (0 - 10.4 h),
         * third column represents mean humidity (65.5 - 96.0 %), and finally fourth column represents mean wind speed
         * (0 - 3.27 km/h).
         */
        double[][] trainingSet = dataLoader.load(new ClassPathResource("data/inmet_13_14_input.csv"));
        double[][] expectedOutput = dataLoader.load(new ClassPathResource("data/inmet_13_14_output.csv"));

        DataNormalizer dataNormalizer = new MinMaxEqualizedDataNormalizer();
        trainingSet = dataNormalizer.normalize(trainingSet);
        expectedOutput = dataNormalizer.normalize(expectedOutput);

        NeuralNetwork neuralNetwork = NeuralNetwork.backpropagationBuilder()
                .numberOfInputs(4)
                .numberOfOutputs(1)
                .numberOfLayers(2)
                .hiddenLayerSize(4)
                .maxEpochs(1000)
                .learningRate(0.5)
                .targetError(0.00001)
                .build();
        neuralNetwork.train(trainingSet, expectedOutput);

        double[][] testTrainingSet = dataLoader.load(new ClassPathResource("data/inmet_13_14_input_test.csv"));
        double[][] testExpectedOutput = dataLoader.load(new ClassPathResource("data/inmet_13_14_output_test.csv"));
        testTrainingSet = dataNormalizer.normalize(testTrainingSet);
        double[][] testEstimatedOutput = new double[testTrainingSet.length][];
        for (int i = 0; i < testTrainingSet.length; i++) {
            testEstimatedOutput[i] = neuralNetwork.feed(testTrainingSet[i]);
        }
        testEstimatedOutput = dataNormalizer.denormalize(testExpectedOutput, testEstimatedOutput);

        double expectedMeanError = 2.12;
        for (int i = 0; i < testEstimatedOutput.length; i++) {
            double meanError = calculateMeanError(testExpectedOutput[i], testEstimatedOutput[i]);
            assertTrue(meanError <= expectedMeanError,
                    "Estimated mean error <" + meanError + "> is greater than expected mean error <"
                            + expectedMeanError + ">");
        }
    }

    /**
     * Test of breast cancer prediction based on the following variables:
     * <ol>
     *     <li>clump thickness (1 - 10);</li>
     *     <li>uniformity of cell size (1 - 10);</li>
     *     <li>uniformity of cell shape (1 - 10);</li>
     *     <li>marginal adhesion (1 - 10);</li>
     *     <li>singe epithelial cell size (1 - 10);</li>
     *     <li>bare nuclei (1 - 10);</li>
     *     <li>bland chromatin (1 - 10);</li>
     *     <li>normal nucleoli (1 - 10);</li>
     *     <li>mitoses (1 - 10).</li>
     * </ol>
     */
    @Test
    void shouldPredictBreastCancer() throws IOException {
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

        double[][] testTrainingSet = dataLoader.load(new ClassPathResource("data/breast_cancer_inputs_test.csv"));
        double[][] testExpectedOutput = dataLoader.load(new ClassPathResource("data/breast_cancer_output_test.csv"));

        double[][] normalizedTestTrainingSet = dataNormalizer.normalize(testTrainingSet);

        double[][] testEstimatedOutput = new double[normalizedTestTrainingSet.length][];
        for (int i = 0; i < normalizedTestTrainingSet.length; i++) {
            testEstimatedOutput[i] = neuralNetwork.feed(normalizedTestTrainingSet[i]);
        }

        double expectedMeanError = 1.0;
        for (int i = 0; i < testEstimatedOutput.length; i++) {
            double meanError = calculateMeanError(testExpectedOutput[i], testEstimatedOutput[i]);
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
