package org.briarheart.neuralnet.layer;

import org.briarheart.neuralnet.activation.ActivationFunction;
import org.briarheart.neuralnet.neuron.Neuron;
import org.briarheart.neuralnet.training.TrainingStrategy;

import java.util.List;

/**
 * @author Roman Chigvintsev
 */
public interface NeuralLayer {
    List<Neuron> getNeurons();

    ActivationFunction getActivationFunction();

    NeuralLayer getPreviousLayer();

    void setPreviousLayer(NeuralLayer layer);

    NeuralLayer getNextLayer();

    void setNextLayer(NeuralLayer layer);

    double[] feed(double[] inputValues);

    double[] feed(double[] inputValues, double[] expectedOutput);

    void adjustWeights(TrainingStrategy trainingStrategy);
}
