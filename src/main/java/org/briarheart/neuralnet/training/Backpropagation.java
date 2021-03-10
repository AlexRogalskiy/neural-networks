package org.briarheart.neuralnet.training;

import com.google.common.base.Preconditions;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.briarheart.neuralnet.NeuralNetwork;
import org.briarheart.neuralnet.activation.ActivationFunction;
import org.briarheart.neuralnet.layer.NeuralLayer;
import org.briarheart.neuralnet.neuron.Neuron;

import java.util.HashMap;
import java.util.Map;

/**
 * As Wikipedia states:
 * <blockquote cite="https://en.wikipedia.org/wiki/ADALINE">
 *     ... backpropagation is a widely used algorithm for training feedforward neural networks. ... backpropagation
 *     computes the gradient of the loss function with respect to the weights of the network for a single input–output
 *     example, and does so efficiently, unlike a naive direct computation of the gradient with respect to each weight
 *     individually. This efficiency makes it feasible to use gradient methods for training multilayer networks,
 *     updating weights to minimize loss; gradient descent, or variants such as stochastic gradient descent, are
 *     commonly used. The backpropagation algorithm works by computing the gradient of the loss function with respect
 *     to each weight by the chain rule, computing the gradient one layer at a time, iterating backward from the last
 *     layer to avoid redundant calculations of intermediate terms in the chain rule ...
 * </blockquote>
 *
 * @author Roman Chigvintsev
 */
@Slf4j
@RequiredArgsConstructor
public class Backpropagation implements SupervisedTrainingStrategy {
    @Getter
    private final double learningRate;

    @Override
    public void train(NeuralNetwork neuralNetwork, double[][] trainingSet, double[][] expectedOutput) {
        Preconditions.checkNotNull(neuralNetwork, "Neural network must not be null");
        Preconditions.checkNotNull(trainingSet, "Training set must not be null");
        Preconditions.checkNotNull(expectedOutput, "Expected output must not be null");

        Map<Integer, Double> msePerEpoch = new HashMap<>();
        int epoch = 0;
        double mse = 1.0;
        while (mse > neuralNetwork.getTargetError() && epoch < neuralNetwork.getMaxEpochs()) {
            double errorSum = 0.0;
            for (int i = 0; i < trainingSet.length; i++) {
                errorSum += train(neuralNetwork, trainingSet, expectedOutput, i);
            }

            mse = errorSum / trainingSet.length;
            if (log.isDebugEnabled()) {
                log.debug("Epoch #{}: [mse={}]", epoch + 1, mse);
            }
            msePerEpoch.put(epoch, mse);
            onEpochEnd(neuralNetwork, epoch);
            epoch++;
        }
        neuralNetwork.setMsePerEpoch(msePerEpoch);
    }

    @Override
    public void adjustWeights(Neuron neuron, ActivationFunction activationFunction) {
        neuron.getInputs().forEach(input -> {
            double newWeight = input.getWeight() + learningRate * neuron.getSensibility()
                    * input.getFrom().getOutputValue();
            input.setWeight(newWeight);
        });
    }

    protected double train(
            NeuralNetwork neuralNetwork,
            double[][] trainingSet,
            double[][] expectedOutput,
            int sampleIndex
    ) {
        NeuralLayer inputLayer = neuralNetwork.getInputLayer();
        double[] estimatedOutput = inputLayer.feed(trainingSet[sampleIndex], expectedOutput[sampleIndex]);
        double meanError = calculateMeanError(estimatedOutput, expectedOutput[sampleIndex]);
        propagateErrorBack(neuralNetwork);
        inputLayer.adjustWeights(this);
        return meanError;
    }

    protected double calculateMeanError(double[] estimatedOutput, double[] expectedOutput) {
        double errorSum = 0.0;
        for (int i = 0; i < estimatedOutput.length; i++) {
            double error = expectedOutput[i] - estimatedOutput[i];
            errorSum += Math.pow(error, 2.0);
        }
        return errorSum / estimatedOutput.length;
    }

    protected void propagateErrorBack(NeuralNetwork neuralNetwork) {
        NeuralLayer inputLayer = neuralNetwork.getInputLayer();
        NeuralLayer currentLayer = neuralNetwork.getOutputLayer();
        while (currentLayer != inputLayer) {
            for (Neuron neuron : currentLayer.getNeurons()) {
                ActivationFunction activationFunction = currentLayer.getActivationFunction();
                double sensibility;
                if (!neuron.getOutputs().isEmpty()) {
                    sensibility = neuron.getOutputs().stream()
                            .mapToDouble(output -> output.getWeight() * output.getTo().getSensibility())
                            .sum();
                    sensibility *= activationFunction.getDerivative().apply(neuron.getOutputValue());
                } else {
                    sensibility = activationFunction.getDerivative().apply(neuron.getOutputValue()) * neuron.getError();
                }
                neuron.setSensibility(sensibility);
            }
            currentLayer = currentLayer.getPreviousLayer();
        }
    }

    protected void onEpochEnd(NeuralNetwork neuralNetwork, int epoch) {
        // Override in subclasses
    }
}
