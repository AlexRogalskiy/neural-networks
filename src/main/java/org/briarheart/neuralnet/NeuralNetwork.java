package org.briarheart.neuralnet;

import com.google.common.base.Preconditions;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import lombok.Getter;
import lombok.Setter;
import org.briarheart.neuralnet.activation.ActivationFunction;
import org.briarheart.neuralnet.layer.HiddenLayer;
import org.briarheart.neuralnet.layer.InputLayer;
import org.briarheart.neuralnet.layer.NeuralLayer;
import org.briarheart.neuralnet.layer.OutputLayer;
import org.briarheart.neuralnet.training.*;

import java.util.Map;

/**
 * @author Roman Chigvintsev
 */
public class NeuralNetwork {
    @Getter
    private final NeuralLayer inputLayer;
    @Getter
    private final transient NeuralLayer outputLayer;

    @Getter
    private final transient int maxEpochs;
    @Getter
    private final transient double targetError;
    @Getter
    private final transient double learningRate;
    @Getter
    private final transient TrainingStrategy trainingStrategy;

    @Getter
    @Setter
    private Map<Integer, Double> msePerEpoch = Map.of();

    @Getter
    @Setter
    private transient double trainingError;
    @Getter
    @Setter
    private transient double meanError;

    private NeuralNetwork(Builder builder) {
        Preconditions.checkArgument(builder.numberOfInputs > 0, "Number of inputs must be greater than zero");
        Preconditions.checkArgument(builder.numberOfOutputs > 0, "Number of outputs must be greater than zero");

        this.inputLayer = new InputLayer(builder.numberOfInputs);
        NeuralLayer previousLayer = this.inputLayer;

        if (builder.numberOfLayers > 1) {
            for (int i = 0; i < builder.numberOfLayers - 1; i++) {
                HiddenLayer hiddenLayer = new HiddenLayer("HiddenLayer " + (i + 1), builder.hiddenLayerSize,
                        builder.defaultActivationFunction);
                previousLayer.setNextLayer(hiddenLayer);
                previousLayer = hiddenLayer;
            }
        }

        ActivationFunction outputLayerActivationFunction = builder.outputLayerActivationFunction != null
                ? builder.outputLayerActivationFunction
                : builder.defaultActivationFunction;
        this.outputLayer = new OutputLayer(builder.numberOfOutputs, outputLayerActivationFunction);
        previousLayer.setNextLayer(this.outputLayer);

        this.maxEpochs = builder.maxEpochs;
        this.targetError = builder.targetError;
        this.learningRate = builder.learningRate;
        this.trainingStrategy = builder.trainingStrategy;
    }

    public static NeuralNetwork.PerceptronBuilder perceptronBuilder() {
        return new PerceptronBuilder();
    }

    public static NeuralNetwork.AdalineBuilder adalineBuilder() {
        return new AdalineBuilder();
    }

    public static NeuralNetwork.BackpropagationBuilder backpropagationBuilder() {
        return new BackpropagationBuilder();
    }

    public static NeuralNetwork.LevenbergMarquardtBuilder levenbergMarquardtBuilder() {
        return new LevenbergMarquardtBuilder();
    }

    public static NeuralNetwork.KohonenBuilder kohonenBuilder() {
        return new KohonenBuilder();
    }

    public void train(double[][] trainingSet, double[] expectedOutput) {
        train(trainingSet, new double[][]{expectedOutput});
    }

    public void train(double[][] trainingSet, double[][] expectedOutput) {
        ((SupervisedTrainingStrategy) trainingStrategy).train(this, trainingSet, expectedOutput);
    }

    public void train(double[][] trainingSet) {
        ((UnsupervisedTrainingStrategy) trainingStrategy).train(this, trainingSet);
    }

    public double[] feed(double[] input) {
        return trainingStrategy.feed(this, input);
    }

    @Override
    public String toString() {
        return toJson(true);
    }

    public String toJson(boolean prettyPrint) {
        GsonBuilder gsonBuilder = new GsonBuilder();
        if (prettyPrint) {
            gsonBuilder.setPrettyPrinting();
        }
        Gson gson = gsonBuilder.create();
        return gson.toJson(this);
    }

    public static abstract class Builder {
        protected int numberOfInputs = 2;
        protected int numberOfOutputs = 1;
        protected int numberOfLayers = 1;
        protected int hiddenLayerSize = 2;

        protected int maxEpochs = 10;
        protected double learningRate = 1.0;
        protected double targetError = 0.002;

        protected ActivationFunction defaultActivationFunction;
        protected ActivationFunction outputLayerActivationFunction;

        private TrainingStrategy trainingStrategy;

        private Builder(ActivationFunction defaultActivationFunction) {
            this.defaultActivationFunction = defaultActivationFunction;
        }

        public NeuralNetwork build() {
            this.trainingStrategy = getTrainingStrategy(learningRate);
            return new NeuralNetwork(this);
        }

        public Builder numberOfInputs(int numberOfInputs) {
            this.numberOfInputs = numberOfInputs;
            return this;
        }

        public Builder maxEpochs(int maxEpochs) {
            this.maxEpochs = maxEpochs;
            return this;
        }

        public Builder targetError(double targetError) {
            this.targetError = targetError;
            return this;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder defaultActivationFunction(ActivationFunction activationFunction) {
            this.defaultActivationFunction = activationFunction;
            return this;
        }

        protected abstract TrainingStrategy getTrainingStrategy(double learningRate);
    }

    public static abstract class MultilayerNetworkBuilder extends Builder {
        private MultilayerNetworkBuilder() {
            this(null);
        }

        private MultilayerNetworkBuilder(ActivationFunction defaultActivationFunction) {
            this(defaultActivationFunction, defaultActivationFunction);
        }

        private MultilayerNetworkBuilder(ActivationFunction defaultActivationFunction,
                                         ActivationFunction outputLayerActivationFunction) {
            super(defaultActivationFunction);
            this.outputLayerActivationFunction = outputLayerActivationFunction;
        }

        @Override
        public MultilayerNetworkBuilder numberOfInputs(int numberOfInputs) {
            return (MultilayerNetworkBuilder) super.numberOfInputs(numberOfInputs);
        }

        public MultilayerNetworkBuilder numberOfOutputs(int numberOfOutputs) {
            this.numberOfOutputs = numberOfOutputs;
            return this;
        }

        public MultilayerNetworkBuilder numberOfLayers(int numberOfLayers) {
            this.numberOfLayers = numberOfLayers;
            return this;
        }

        public MultilayerNetworkBuilder hiddenLayerSize(int size) {
            this.hiddenLayerSize = size;
            return this;
        }

        @Override
        public MultilayerNetworkBuilder learningRate(double learningRate) {
            return (MultilayerNetworkBuilder) super.learningRate(learningRate);
        }

        @Override
        public MultilayerNetworkBuilder maxEpochs(int maxEpochs) {
            return (MultilayerNetworkBuilder) super.maxEpochs(maxEpochs);
        }

        @Override
        public MultilayerNetworkBuilder targetError(double targetError) {
            return (MultilayerNetworkBuilder) super.targetError(targetError);
        }

        @Override
        public MultilayerNetworkBuilder defaultActivationFunction(ActivationFunction activationFunction) {
            return (MultilayerNetworkBuilder) super.defaultActivationFunction(activationFunction);
        }

        public MultilayerNetworkBuilder outputLayerActivationFunction(ActivationFunction activationFunction) {
            this.outputLayerActivationFunction = activationFunction;
            return this;
        }
    }

    public static class PerceptronBuilder extends Builder {
        private PerceptronBuilder() {
            super(ActivationFunction.HARD_LIMITING_THRESHOLD);
        }

        @Override
        protected TrainingStrategy getTrainingStrategy(double learningRate) {
            return new Perceptron(learningRate);
        }
    }

    public static class AdalineBuilder extends Builder {
        private AdalineBuilder() {
            super(ActivationFunction.LINEAR);
        }

        @Override
        protected TrainingStrategy getTrainingStrategy(double learningRate) {
            return new Adaline(learningRate);
        }
    }

    public static class BackpropagationBuilder extends MultilayerNetworkBuilder {
        private BackpropagationBuilder() {
            super(ActivationFunction.SIGMOID, ActivationFunction.LINEAR);
        }

        @Override
        protected TrainingStrategy getTrainingStrategy(double learningRate) {
            return new Backpropagation(learningRate);
        }
    }

    public static class LevenbergMarquardtBuilder extends MultilayerNetworkBuilder {
        private LevenbergMarquardtBuilder() {
            super(ActivationFunction.SIGMOID, ActivationFunction.LINEAR);
        }

        @Override
        protected TrainingStrategy getTrainingStrategy(double learningRate) {
            return new LevenbergMarquardt(learningRate);
        }
    }

    public static class KohonenBuilder extends MultilayerNetworkBuilder {
        private KohonenBuilder() {
        }

        @Override
        protected TrainingStrategy getTrainingStrategy(double learningRate) {
            return new Kohonen(learningRate);
        }
    }
}
