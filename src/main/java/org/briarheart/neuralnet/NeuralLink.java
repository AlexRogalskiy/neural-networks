package org.briarheart.neuralnet;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.briarheart.neuralnet.neuron.Neuron;

/**
 * @author Roman Chigvintsev
 */
@RequiredArgsConstructor
public class NeuralLink {
    @Getter
    private final transient Neuron from;
    @Getter
    private final transient Neuron to;

    @Getter
    @Setter
    private double weight = Math.random();

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
}
