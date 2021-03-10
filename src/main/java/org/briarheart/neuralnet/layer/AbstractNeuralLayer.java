package org.briarheart.neuralnet.layer;

import com.google.common.base.Preconditions;
import lombok.Getter;
import lombok.Setter;
import org.briarheart.neuralnet.NeuralLink;
import org.briarheart.neuralnet.activation.ActivationFunction;
import org.briarheart.neuralnet.neuron.Bias;
import org.briarheart.neuralnet.neuron.Neuron;
import org.briarheart.neuralnet.training.TrainingStrategy;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Roman Chigvintsev
 */
public abstract class AbstractNeuralLayer implements NeuralLayer {
    @Getter
    private final int size;
    @Getter
    private final List<Neuron> neurons;
    @Getter
    private final ActivationFunction activationFunction;

    @Getter
    private NeuralLayer nextLayer;
    @Getter
    @Setter
    private transient NeuralLayer previousLayer;

    public AbstractNeuralLayer(int size, List<Neuron> neurons, ActivationFunction activationFunction) {
        Preconditions.checkNotNull(neurons, "List of neurons must not be null");
        this.size = size;
        this.neurons = neurons;
        this.activationFunction = activationFunction;
    }

    @Override
    public void setNextLayer(NeuralLayer nextLayer) {
        this.nextLayer = nextLayer;

        if (nextLayer != null) {
            connectLayers(nextLayer);
        } else {
            neurons.forEach(neuron -> neuron.setOutputs(List.of()));
        }
    }

    @Override
    public double[] feed(double[] inputValues) {
        return feed(inputValues, null);
    }

    @Override
    public double[] feed(double[] inputValues, double[] expectedOutput) {
        for (Neuron neuron : neurons) {
            if (neuron instanceof Bias) {
                continue;
            }

            double weightedSum = calculateWeightedSum(neuron);
            neuron.setWeightedSum(weightedSum);

            double outputValue = activationFunction.apply(weightedSum);
            neuron.setOutputValue(outputValue);
        }

        return nextLayer != null
                ? nextLayer.feed(inputValues, expectedOutput)
                : neurons.stream().mapToDouble(Neuron::getOutputValue).toArray();
    }

    @Override
    public void adjustWeights(TrainingStrategy trainingStrategy) {
        Preconditions.checkNotNull(trainingStrategy, "Training strategy must not be null");
        neurons.forEach(neuron -> trainingStrategy.adjustWeights(neuron, activationFunction));
    }

    protected double calculateWeightedSum(Neuron neuron) {
        List<NeuralLink> inputs = neuron.getInputs();
        double weightedSum = 0.0;
        for (NeuralLink input : inputs) {
            weightedSum += input.getWeight() * input.getFrom().getOutputValue();
        }
        return weightedSum;
    }

    private void connectLayers(NeuralLayer nextLayer) {
        for (Neuron neuron : neurons) {
            List<Neuron> nextLayerNeurons = nextLayer.getNeurons();
            List<NeuralLink> outputs = new ArrayList<>(nextLayerNeurons.size());
            for (Neuron nextLayerNeuron : nextLayerNeurons) {
                if (nextLayerNeuron instanceof Bias) {
                    continue;
                }

                List<NeuralLink> inputs = nextLayerNeuron.getInputs();
                if (inputs.isEmpty()) {
                    inputs = new ArrayList<>(neurons.size());
                    nextLayerNeuron.setInputs(inputs);
                }
                NeuralLink neuralLink = new NeuralLink(neuron, nextLayerNeuron);
                outputs.add(neuralLink);
                inputs.add(neuralLink);
            }
            neuron.setOutputs(outputs);
        }

        nextLayer.setPreviousLayer(this);
    }
}
