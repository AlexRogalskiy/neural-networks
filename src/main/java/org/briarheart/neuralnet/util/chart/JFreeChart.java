package org.briarheart.neuralnet.util.chart;

import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import org.jfree.chart.ChartFrame;

/**
 * @author Roman Chigvintsev
 */
@RequiredArgsConstructor
public class JFreeChart implements Chart {
    @NonNull
    private final ChartFrame frame;

    @Override
    public void show() {
        frame.pack();
        frame.setVisible(true);
    }
}
