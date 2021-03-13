package org.briarheart.neuralnet.training;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.briarheart.neuralnet.NeuralLink;
import org.briarheart.neuralnet.NeuralNetwork;
import org.briarheart.neuralnet.activation.ActivationFunction;
import org.briarheart.neuralnet.layer.NeuralLayer;
import org.briarheart.neuralnet.neuron.Bias;
import org.briarheart.neuralnet.neuron.Neuron;
import org.briarheart.neuralnet.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * As Wikipedia states:
 * <blockquote cite="https://en.wikipedia.org/wiki/Self-organizing_map">
 *     A self-organizing map (SOM) or self-organizing feature map (SOFM) is a type of artificial neural network (ANN)
 *     that is trained using unsupervised learning to produce a low-dimensional (typically two-dimensional),
 *     discretized representation of the input space of the training samples, called a map, and is therefore a method
 *     to do dimensionality reduction. Self-organizing maps differ from other artificial neural networks as they apply
 *     competitive learning as opposed to error-correction learning (such as backpropagation with gradient descent),
 *     and in the sense that they use a neighborhood function to preserve the topological properties of the input
 *     space.<br/>
 *     This makes SOMs useful for visualization by creating low-dimensional views of high-dimensional data, akin to
 *     multidimensional scaling. The artificial neural network introduced by the Finnish professor Teuvo Kohonen in
 *     the 1980s is sometimes called a Kohonen map or network.
 * </blockquote>
 *
 * @author Roman Chigvintsev
 */
@RequiredArgsConstructor
public class Kohonen implements UnsupervisedTrainingStrategy {
    private static final Logger log = LoggerFactory.getLogger(Kohonen.class);

    @Getter
    private final double learningRate;

    @Override
    public void train(NeuralNetwork neuralNetwork, double[][] trainingSet) {
        NeuralLayer inputLayer = neuralNetwork.getInputLayer();
        resetWeights(inputLayer);
        for (int epoch = 0; epoch < neuralNetwork.getMaxEpochs(); epoch++) {
            for (double[] sample : trainingSet) {
                feedLayer(inputLayer, sample);
                double[] distances = calculateEuclideanDistances(neuralNetwork);
                int winnerNeuronIndex = Arrays.indexOfMin(distances);
                log.debug("Epoch #{}: [training_sample={}, winner_neuron_index={}]",
                        epoch + 1, java.util.Arrays.toString(sample), winnerNeuronIndex);
                NeuralLayer outputLayer = neuralNetwork.getOutputLayer();
                Neuron winnerNeuron = outputLayer.getNeurons().get(winnerNeuronIndex);
                adjustWeights(winnerNeuron);
            }
        }
    }

    @Override
    public void adjustWeights(Neuron neuron, ActivationFunction activationFunction) {
        neuron.getInputs().forEach(neuralLink -> {
            double newWeight = neuralLink.getWeight() + learningRate
                    * (neuralLink.getFrom().getOutputValue() - neuralLink.getWeight());
            neuralLink.setWeight(newWeight);
        });
    }

    @Override
    public double[] feed(NeuralNetwork neuralNetwork, double[] input) {
        NeuralLayer inputLayer = neuralNetwork.getInputLayer();
        NeuralLayer outputLayer = neuralNetwork.getOutputLayer();

        double[] result = new double[outputLayer.getNeurons().size()];
        java.util.Arrays.fill(result, -1.0);

        feedLayer(inputLayer, input);
        double[] distances = calculateEuclideanDistances(neuralNetwork);
        int winnerNeuronIndex = Arrays.indexOfMin(distances);
        result[winnerNeuronIndex] = 1.0;

        return result;
    }

    private void resetWeights(NeuralLayer layer) {
        layer.getNeurons().forEach(neuron -> neuron.getOutputs().forEach(output -> output.setWeight(0.0)));
    }

    private void feedLayer(NeuralLayer layer, double[] input) {
        int i = 0;
        for (Neuron neuron : layer.getNeurons()) {
            if (!(neuron instanceof Bias)) {
                neuron.setOutputValue(input[i++]);
            }
        }
    }

    private double[] calculateEuclideanDistances(NeuralNetwork neuralNetwork) {
        NeuralLayer outputLayer = neuralNetwork.getOutputLayer();
        double[] result = new double[outputLayer.getNeurons().size()];
        for (int i = 0; i < outputLayer.getNeurons().size(); i++) {
            Neuron neuron = outputLayer.getNeurons().get(i);
            double distance = 0.0;
            for (NeuralLink input : neuron.getInputs()) {
                distance += Math.pow(input.getFrom().getOutputValue() - input.getWeight(), 2.0);
            }
            result[i] = distance;
        }
        return result;
    }
}
