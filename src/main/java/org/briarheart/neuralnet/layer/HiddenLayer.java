package org.briarheart.neuralnet.layer;

import com.google.common.base.Preconditions;
import lombok.Getter;
import org.briarheart.neuralnet.activation.ActivationFunction;
import org.briarheart.neuralnet.neuron.BasicNeuron;
import org.briarheart.neuralnet.neuron.Bias;
import org.briarheart.neuralnet.neuron.Neuron;
import org.briarheart.neuralnet.training.TrainingStrategy;

import java.util.List;

/**
 * @author Roman Chigvintsev
 */
public class HiddenLayer extends AbstractNeuralLayer {
    @Getter
    private final String name;

    public HiddenLayer(String name, int size, ActivationFunction activationFunction) {
        super(size, createNeurons(name, size), activationFunction);
        Preconditions.checkNotNull(activationFunction, "Activation function must not be null");
        this.name = name;
    }

    @Override
    public void adjustWeights(TrainingStrategy trainingStrategy) {
        super.adjustWeights(trainingStrategy);
        getNextLayer().adjustWeights(trainingStrategy);
    }

    private static List<Neuron> createNeurons(String layerName, int size) {
        Preconditions.checkArgument(size > 0, "Hidden layer size must be greater than zero");
        Neuron[] neurons = new Neuron[size + 1];
        neurons[0] = new Bias(layerName + "_Bias");
        for (int i = 0; i < size; i++) {
            neurons[i + 1] = new BasicNeuron("HiddenNeuron " + (i + 1));
        }
        return List.of(neurons);
    }
}
