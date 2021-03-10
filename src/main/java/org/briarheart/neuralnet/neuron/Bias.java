package org.briarheart.neuralnet.neuron;

import org.briarheart.neuralnet.NeuralLink;

import java.util.List;

/**
 * @author Roman Chigvintsev
 */
public class Bias extends BasicNeuron {
    public Bias(String name) {
        super(name);
        super.setOutputValue(1.0);
    }

    @Override
    public void setOutputValue(double outputValue) {
        // Do nothing
    }

    @Override
    public void setInputs(List<NeuralLink> inputs) {
        throw new UnsupportedOperationException("Bias neuron cannot have inputs");
    }
}
