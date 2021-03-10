package org.briarheart.neuralnet.layer;

import com.google.common.base.Preconditions;
import org.briarheart.neuralnet.neuron.BasicNeuron;
import org.briarheart.neuralnet.neuron.Bias;
import org.briarheart.neuralnet.neuron.Neuron;
import org.briarheart.neuralnet.training.TrainingStrategy;

import java.util.List;

/**
 * @author Roman Chigvintsev
 */
public class InputLayer extends AbstractNeuralLayer {
    public InputLayer(int size) {
        this(size, true);
    }

    public InputLayer(int size, boolean withBias) {
        super(size, createNeurons(size, withBias), null);
    }

    @Override
    public void setPreviousLayer(NeuralLayer previous) {
        throw new UnsupportedOperationException("Input layer cannot have layers in front of it");
    }

    @Override
    public double[] feed(double[] inputValues, double[] expectedOutput) {
        Preconditions.checkNotNull(inputValues, "Input values must not be null");
        Preconditions.checkArgument(inputValues.length == getSize(),
                "Number of input values must match number of neurons in layer");

        int i = 0;
        for (Neuron neuron : getNeurons()) {
            if (neuron instanceof Bias) {
                continue;
            }
            neuron.setOutputValue(inputValues[i++]);
        }
        return getNextLayer().feed(inputValues, expectedOutput);
    }

    @Override
    public void adjustWeights(TrainingStrategy trainingStrategy) {
        getNextLayer().adjustWeights(trainingStrategy);
    }

    private static List<Neuron> createNeurons(int size, boolean withBias) {
        Preconditions.checkArgument(size > 0, "Input layer size must be greater than zero");
        int actualSize = withBias ? size + 1 : size;
        Neuron[] neurons = new BasicNeuron[actualSize];
        int i = 0, n = 1;
        if (withBias) {
            neurons[i++] = new Bias("InputLayer_Bias");
        }
        while (i < actualSize) {
            neurons[i++] = new BasicNeuron("InputNeuron " + n++);
        }
        return List.of(neurons);
    }
}
