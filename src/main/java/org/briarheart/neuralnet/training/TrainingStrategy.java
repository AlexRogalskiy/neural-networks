package org.briarheart.neuralnet.training;

import org.briarheart.neuralnet.NeuralNetwork;
import org.briarheart.neuralnet.activation.ActivationFunction;
import org.briarheart.neuralnet.neuron.Neuron;

/**
 * Neural network training strategy.
 *
 * @author Roman Chigvintsev
 */
public interface TrainingStrategy {
    default void adjustWeights(Neuron neuron) {
        adjustWeights(neuron, null);
    }

    void adjustWeights(Neuron neuron, ActivationFunction activationFunction);

    default double[] feed(NeuralNetwork neuralNetwork, double[] input) {
        return neuralNetwork.getInputLayer().feed(input);
    }
}
