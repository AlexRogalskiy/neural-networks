package org.briarheart.neuralnet.training;

import lombok.Setter;
import org.briarheart.neuralnet.NeuralNetwork;

/**
 * Stochastic online learning algorithm. Its two main features are random choice of samples for training and variation
 * of learning rate in runtime (online). This training method is used when noise is found in the objective function.
 * It helps to escape the local minimum (one of the best solutions) and to reach the global minimum (the best solution).
 *
 * @author Roman Chigvintsev
 */
public class OnlineBackpropagation extends Backpropagation {
    @Setter
    private double learningRateReductionPercentage = 0.01;

    public OnlineBackpropagation(double learningRate) {
        super(learningRate, true);
    }

    @Override
    protected double train(NeuralNetwork neuralNetwork,
                           double[][] trainingSet,
                           double[][] expectedOutput,
                           int sampleIndex) {
        double meanError = super.train(neuralNetwork, trainingSet, expectedOutput, sampleIndex);
        setLearningRate(reduceLearningRate(getLearningRate(), learningRateReductionPercentage));
        return meanError;
    }

    private double reduceLearningRate(double learningRate, double reductionPercentage) {
        double newLearningRate = learningRate * ((100.0 - reductionPercentage) / 100.0);
        if (newLearningRate < 0.1) {
            newLearningRate = 1.0;
        }
        return newLearningRate;
    }
}
