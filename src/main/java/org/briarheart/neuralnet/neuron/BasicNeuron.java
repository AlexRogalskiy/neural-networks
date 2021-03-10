package org.briarheart.neuralnet.neuron;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.briarheart.neuralnet.NeuralLink;

import java.util.List;

/**
 * @author Roman Chigvintsev
 */
@RequiredArgsConstructor
public class BasicNeuron implements Neuron {
    @Getter
    private final String name;

    @Getter
    @Setter
    private List<NeuralLink> inputs = List.of();
    @Getter
    @Setter
    private List<NeuralLink> outputs = List.of();
    @Getter
    @Setter
    private double weightedSum;
    @Getter
    @Setter
    private double outputValue;
    @Getter
    @Setter
    private double error;
    @Getter
    @Setter
    private double sensibility;
}
