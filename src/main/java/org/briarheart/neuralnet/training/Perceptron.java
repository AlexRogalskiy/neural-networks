package org.briarheart.neuralnet.training;

import org.briarheart.neuralnet.activation.ActivationFunction;
import org.briarheart.neuralnet.neuron.Neuron;

/**
 * As Wikipedia states:
 * <blockquote cite="https://en.wikipedia.org/wiki/Perceptron">
 *     Perceptron is an algorithm for supervised learning of binary classifiers. A binary classifier is a function
 *     which can decide whether or not an input, represented by a vector of numbers, belongs to some specific class.
 *     It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear
 *     predictor function combining a set of weights with the feature vector.
 * </blockquote>
 *
 * @author Roman Chigvintsev
 */
public class Perceptron extends PerceptronBasedTrainingStrategy {
    public Perceptron(double learningRate) {
        super(learningRate);
    }

    @Override
    public void adjustWeights(Neuron neuron, ActivationFunction activationFunction) {
        neuron.getInputs().forEach(input -> {
            double newWeight = input.getWeight() + getLearningRate() * neuron.getError()
                    * input.getFrom().getOutputValue();
            input.setWeight(newWeight);
        });
    }
}
