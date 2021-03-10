package org.briarheart.neuralnet.util;

import com.google.common.base.Preconditions;
import org.briarheart.neuralnet.util.resource.Resource;
import org.briarheart.neuralnet.util.resource.ResourceNotFoundException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Roman Chigvintsev
 */
public class CsvDataLoader implements DataLoader {
    @Override
    public double[][] load(Resource resource) throws IOException {
        Preconditions.checkNotNull(resource, "Resource must not be null");
        URL resourceUrl = resource.getUrl();
        if (resourceUrl == null) {
            throw new ResourceNotFoundException(resource);
        }

        try (InputStream resourceStream = resourceUrl.openStream();
             BufferedReader resourceReader = new BufferedReader(new InputStreamReader(resourceStream))) {
            List<double[]> rows = new ArrayList<>();
            String line = resourceReader.readLine();
            while (line != null) {
                String[] components = line.split(",");
                double[] values = new double[components.length];
                for (int i = 0; i < components.length; i++) {
                    values[i] = Double.parseDouble(components[i]);
                }
                rows.add(values);
                line = resourceReader.readLine();
            }
            return rows.toArray(new double[0][]);
        }
    }
}
