package org.briarheart.neuralnet.training;

import org.briarheart.neuralnet.NeuralNetwork;

/**
 * @author Roman Chigvintsev
 */
public interface UnsupervisedTrainingStrategy extends TrainingStrategy {
    void train(NeuralNetwork neuralNetwork, double[][] trainingSet);
}
