package org.briarheart.neuralnet.layer;

import com.google.common.base.Preconditions;
import org.briarheart.neuralnet.activation.ActivationFunction;
import org.briarheart.neuralnet.neuron.BasicNeuron;
import org.briarheart.neuralnet.neuron.Neuron;

import java.util.List;

/**
 * @author Roman Chigvintsev
 */
public class OutputLayer extends AbstractNeuralLayer {
    public OutputLayer(int size, ActivationFunction activationFunction) {
        super(size, createNeurons(size), activationFunction);
    }

    @Override
    public void setNextLayer(NeuralLayer nextLayer) {
        throw new UnsupportedOperationException("Output layer cannot have layers behind it");
    }

    @Override
    public double[] feed(double[] inputValues, double[] expectedOutput) {
        double[] estimatedOutput = super.feed(inputValues, expectedOutput);
        if (expectedOutput != null) {
            List<Neuron> neurons = getNeurons();
            for (int i = 0; i < neurons.size(); i++) {
                neurons.get(i).setError(expectedOutput[i] - estimatedOutput[i]);
            }
        }
        return estimatedOutput;
    }

    private static List<Neuron> createNeurons(int size) {
        Preconditions.checkArgument(size > 0, "Output layer size must be greater than zero");
        Neuron[] neurons = new Neuron[size];
        for (int i = 0; i < size; i++) {
            neurons[i] = new BasicNeuron("OutputNeuron " + (i + 1));
        }
        return List.of(neurons);
    }
}
