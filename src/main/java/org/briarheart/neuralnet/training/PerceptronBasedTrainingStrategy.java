package org.briarheart.neuralnet.training;

import com.google.common.base.Preconditions;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.briarheart.neuralnet.NeuralNetwork;
import org.briarheart.neuralnet.layer.NeuralLayer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * @author Roman Chigvintsev
 */
@RequiredArgsConstructor
@Slf4j
public abstract class PerceptronBasedTrainingStrategy implements SupervisedTrainingStrategy {
    @Getter
    private final double learningRate;

    @Getter
    @Setter
    private double error;

    @Override
    public void train(NeuralNetwork neuralNetwork, double[][] trainingSet, double[][] expectedOutput) {
        Preconditions.checkNotNull(neuralNetwork, "Neural network must not be null");
        Preconditions.checkNotNull(trainingSet, "Training set must not be null");
        Preconditions.checkNotNull(expectedOutput, "Expected output must not be null");

        Map<Integer, Double> msePerEpoch = new HashMap<>();
        int epoch = 0;
        double error = 0.0;
        while (epoch < neuralNetwork.getMaxEpochs()) {
            for (int i = 0; i < trainingSet.length; i++) {
                error = train(neuralNetwork, trainingSet[i], expectedOutput[0][i], epoch);
            }
            msePerEpoch.put(epoch, Math.pow(error, 2.0));
            epoch++;
        }

        neuralNetwork.setMsePerEpoch(msePerEpoch);
        neuralNetwork.setTrainingError(error);
    }

    protected double train(NeuralNetwork neuralNetwork, double[] trainingSample, double expectedOutput, int epoch) {
        NeuralLayer inputLayer = neuralNetwork.getInputLayer();
        double[] estimatedOutput = inputLayer.feed(trainingSample, new double[] {expectedOutput});
        double error = expectedOutput - estimatedOutput[0];

        if (log.isDebugEnabled()) {
            log.debug("Epoch #{}: [training_sample={}, estimated_output={}, expected_output={}, error={}]",
                    epoch + 1, Arrays.toString(trainingSample), estimatedOutput[0], expectedOutput, error);
        }

        if (Math.abs(error) > neuralNetwork.getTargetError()) {
            inputLayer.adjustWeights(this);
        }

        return error;
    }
}
