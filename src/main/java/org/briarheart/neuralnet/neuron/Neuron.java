package org.briarheart.neuralnet.neuron;

import org.briarheart.neuralnet.NeuralLink;

import java.util.List;

/**
 * @author Roman Chigvintsev
 */
public interface Neuron {
    List<NeuralLink> getInputs();

    void setInputs(List<NeuralLink> inputs);

    List<NeuralLink> getOutputs();

    void setOutputs(List<NeuralLink> outputs);

    double getWeightedSum();

    void setWeightedSum(double weightedSum);

    double getOutputValue();

    void setOutputValue(double outputValue);

    double getError();

    void setError(double error);

    double getSensibility();

    void setSensibility(double sensibility);
}
