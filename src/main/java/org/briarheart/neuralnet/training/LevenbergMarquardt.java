package org.briarheart.neuralnet.training;

import com.google.common.base.Preconditions;
import org.briarheart.neuralnet.NeuralLink;
import org.briarheart.neuralnet.NeuralNetwork;
import org.briarheart.neuralnet.layer.NeuralLayer;
import org.briarheart.neuralnet.neuron.Neuron;
import org.briarheart.neuralnet.util.math.IdentityMatrix;
import org.briarheart.neuralnet.util.math.Matrix;

/**
 * As Wikipedia states:
 * <blockquote>
 *     In mathematics and computing, the Levenberg–Marquardt algorithm (LMA or just LM), also known as the damped
 *     least-squares (DLS) method, is used to solve non-linear least squares problems. These minimization problems
 *     arise especially in least squares curve fitting.<br/>
 *     ...<br/>
 *     The algorithm was first published in 1944 by Kenneth Levenberg, while working at the Frankford Army Arsenal.
 *     It was rediscovered in 1963 by Donald Marquardt, who worked as a statistician at DuPont, and independently by
 *     Girard, Wynne and Morrison.
 * </blockquote>
 *
 * @author Roman Chigvintsev
 */
public class LevenbergMarquardt extends Backpropagation {
    private final double damping;

    private Matrix jacobian;
    private Matrix error;

    public LevenbergMarquardt(double learningRate) {
        this(learningRate, 0.1);
    }

    public LevenbergMarquardt(double learningRate, double damping) {
        super(learningRate);
        this.damping = damping;
    }

    @Override
    public void train(NeuralNetwork neuralNetwork, double[][] trainingSet, double[][] expectedOutput) {
        Preconditions.checkNotNull(neuralNetwork, "Neural network must not be null");
        Preconditions.checkNotNull(trainingSet, "Training set must not be null");

        int numberOfColumns = 0;
        NeuralLayer currentLayer = neuralNetwork.getInputLayer().getNextLayer();
        while (currentLayer != null) {
            for (Neuron neuron : currentLayer.getNeurons()) {
                numberOfColumns += neuron.getInputs().size();
            }
            currentLayer = currentLayer.getNextLayer();
        }

        jacobian = new Matrix(trainingSet.length, numberOfColumns);
        error = new Matrix(trainingSet.length, 1);

        super.train(neuralNetwork, trainingSet, expectedOutput);
    }

    @Override
    protected double train(
            NeuralNetwork neuralNetwork,
            double[][] trainingSet,
            double[][] expectedOutput,
            int sampleIndex
    ) {
        double meanError = super.train(neuralNetwork, trainingSet, expectedOutput, sampleIndex);
        updateJacobianMatrix(neuralNetwork, sampleIndex, meanError);
        return meanError;
    }

    @Override
    protected void onEpochEnd(NeuralNetwork neuralNetwork, int epoch) {
        applyWeightDeltas(neuralNetwork);
    }

    private void updateJacobianMatrix(NeuralNetwork neuralNetwork, int sampleIndex, double meanError) {
        NeuralLayer inputLayer = neuralNetwork.getInputLayer();

        int col = 0;
        NeuralLayer currentLayer = inputLayer.getNextLayer();
        while (currentLayer != null) {
            for (Neuron neuron : currentLayer.getNeurons()) {
                for (NeuralLink input : neuron.getInputs()) {
                    double value = neuron.getSensibility() * input.getFrom().getOutputValue() / meanError;
                    jacobian.set(sampleIndex, col++, value);
                }
            }
            currentLayer = currentLayer.getNextLayer();
        }

        error.set(sampleIndex, 0, meanError);
    }

    private void applyWeightDeltas(NeuralNetwork neuralNetwork) {
        Matrix m1 = jacobian.transpose()
                .multiply(jacobian)
                .add(new IdentityMatrix(jacobian.getSize().getY()).multiplyScalar(damping));
        Matrix m2 = jacobian.transpose().multiply(error);
        Matrix delta = m1.inverse().multiply(m2);

        int i = 0;
        NeuralLayer currentLayer = neuralNetwork.getInputLayer().getNextLayer();
        while (currentLayer != null) {
            for (Neuron neuron : currentLayer.getNeurons()) {
                for (NeuralLink input : neuron.getInputs()) {
                    input.setWeight(input.getWeight() + delta.get(i++, 0));
                }
            }
            currentLayer = currentLayer.getNextLayer();
        }
    }
}
