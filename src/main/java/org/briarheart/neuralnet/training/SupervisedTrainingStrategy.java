package org.briarheart.neuralnet.training;

import org.briarheart.neuralnet.NeuralNetwork;

/**
 * @author Roman Chigvintsev
 */
public interface SupervisedTrainingStrategy extends TrainingStrategy {
    void train(NeuralNetwork neuralNetwork, double[][] trainingSet, double[][] expectedOutput);
}
