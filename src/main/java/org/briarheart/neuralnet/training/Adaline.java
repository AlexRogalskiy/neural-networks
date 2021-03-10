package org.briarheart.neuralnet.training;

import org.briarheart.neuralnet.activation.ActivationFunction;
import org.briarheart.neuralnet.neuron.Neuron;

/**
 * As Wikipedia states:
 * <blockquote cite="https://en.wikipedia.org/wiki/ADALINE">
 *     ADALINE (Adaptive Linear Neuron or later Adaptive Linear Element) is an early single-layer artificial neural
 *     network and the name of the physical device that implemented this network. The network uses memistors.
 *     <sup>1</sup>. It was developed by Professor Bernard Widrow and his graduate student Ted Hoff at Stanford
 *     University in 1960. It is based on the McCulloch–Pitts neuron. It consists of a weight, a bias and a summation
 *     function.
 * </blockquote>
 * <sup>1</sup> memistors - resistors with memory.
 *
 * @author Roman Chigvintsev
 */
public class Adaline extends PerceptronBasedTrainingStrategy {
    public Adaline(double learningRate) {
        super(learningRate);
    }

    @Override
    public void adjustWeights(Neuron neuron, ActivationFunction activationFunction) {
        double weightedSumFactor = activationFunction.getDerivative().apply(neuron.getWeightedSum());
        neuron.getInputs().forEach(input -> {
            double newWeight = input.getWeight() + getLearningRate() * neuron.getError()
                    * input.getFrom().getOutputValue() * weightedSumFactor;
            input.setWeight(newWeight);
        });
    }
}
