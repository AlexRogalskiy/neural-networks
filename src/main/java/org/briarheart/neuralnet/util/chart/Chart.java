package org.briarheart.neuralnet.util.chart;

import lombok.Getter;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.labels.ItemLabelAnchor;
import org.jfree.chart.labels.ItemLabelPosition;
import org.jfree.chart.labels.StandardCategoryToolTipGenerator;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.category.LayeredBarRenderer;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.ui.TextAnchor;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.awt.*;
import java.util.HashMap;
import java.util.Map;

/**
 * @author Roman Chigvintsev
 */
public interface Chart {
    void show();

    static XYLineChartBuilder xyLineChartBuilder() {
        return new XYLineChartBuilder();
    }

    static LayeredBarChartBuilder layeredBarChartBuilder() {
        return new LayeredBarChartBuilder();
    }

    @Getter
    abstract class AbstractChartBuilder {
        private String title;
        private String xAxisLabel;
        private String yAxisLabel;

        protected final Map<String, AbstractDataSeries> dataSeriesMap = new HashMap<>();

        protected AbstractChartBuilder() {
        }

        public abstract Chart build();

        public AbstractChartBuilder title(String title) {
            this.title = title;
            return this;
        }

        public AbstractChartBuilder xAxisLabel(String xAxisLabel) {
            this.xAxisLabel = xAxisLabel;
            return this;
        }

        public AbstractChartBuilder yAxisLabel(String yAxisLabel) {
            this.yAxisLabel = yAxisLabel;
            return this;
        }
    }

    class XYLineChartBuilder extends AbstractChartBuilder {
        private double[] axisRange;

        private XYLineChartBuilder() {
        }

        @Override
        public XYLineChartBuilder title(String title) {
            return (XYLineChartBuilder) super.title(title);
        }

        @Override
        public XYLineChartBuilder xAxisLabel(String xAxisLabel) {
            return (XYLineChartBuilder) super.xAxisLabel(xAxisLabel);
        }

        @Override
        public XYLineChartBuilder yAxisLabel(String yAxisLabel) {
            return (XYLineChartBuilder) super.yAxisLabel(yAxisLabel);
        }

        public XYLineChartBuilder axisRange(double lower, double upper) {
            this.axisRange = new double[]{lower, upper};
            return this;
        }

        public XYLineDataSeries.Builder dataSeries() {
            return new XYLineDataSeries.Builder(this);
        }

        @Override
        public Chart build() {
            XYSeriesCollection seriesCollection = new XYSeriesCollection();
            org.jfree.chart.JFreeChart chart = ChartFactory.createXYLineChart(getTitle(), getXAxisLabel(),
                    getYAxisLabel(), seriesCollection, PlotOrientation.VERTICAL, true, true, false);
            XYPlot plot = chart.getXYPlot();

            int seriesIndex = 0;
            for (String key : dataSeriesMap.keySet()) {
                XYSeries series = new XYSeries(key);

                XYLineDataSeries dataSeries = (XYLineDataSeries) dataSeriesMap.get(key);
                for (int i = 0; i < dataSeries.values.length; i++) {
                    series.add(i + 1, dataSeries.values[i]);
                }
                seriesCollection.addSeries(series);

                if (dataSeries.color != null) {
                    plot.getRenderer().setSeriesPaint(seriesIndex, dataSeries.color);
                }

                if (dataSeries.strokeWidth >= 0.0 || dataSeries.dashedStroke) {
                    XYItemRenderer renderer = plot.getRenderer();
                    float strokeWidth = dataSeries.strokeWidth >= 0.0f ? dataSeries.strokeWidth : 1.0f;
                    float[] dash = dataSeries.dashedStroke ? new float[]{1.0f, 6.0f} : null;
                    renderer.setSeriesStroke(seriesIndex, new BasicStroke(strokeWidth, BasicStroke.CAP_ROUND,
                            BasicStroke.JOIN_ROUND, 1.0f, dash, 0.0f));
                }

                seriesIndex++;
            }

            if (axisRange != null) {
                plot.getRangeAxis().setRange(axisRange[0], axisRange[1]);
            }

            ChartFrame frame = new ChartFrame(getTitle(), chart);
            return new JFreeChart(frame);
        }
    }

    class LayeredBarChartBuilder extends AbstractChartBuilder {
        @Override
        public LayeredBarChartBuilder title(String title) {
            return (LayeredBarChartBuilder) super.title(title);
        }

        @Override
        public LayeredBarChartBuilder xAxisLabel(String xAxisLabel) {
            return (LayeredBarChartBuilder) super.xAxisLabel(xAxisLabel);
        }

        @Override
        public LayeredBarChartBuilder yAxisLabel(String yAxisLabel) {
            return (LayeredBarChartBuilder) super.yAxisLabel(yAxisLabel);
        }

        public LayeredBarChartDataSeries.Builder dataSeries() {
            return new LayeredBarChartDataSeries.Builder(this);
        }

        @Override
        public Chart build() {
            CategoryAxis categoryAxis = new CategoryAxis(getXAxisLabel());
            ValueAxis valueAxis = new NumberAxis(getYAxisLabel());

            LayeredBarRenderer renderer = new LayeredBarRenderer();
            ItemLabelPosition position1 = new ItemLabelPosition(ItemLabelAnchor.OUTSIDE12, TextAnchor.BOTTOM_CENTER);
            renderer.setDefaultPositiveItemLabelPosition(position1);
            ItemLabelPosition position2 = new ItemLabelPosition(ItemLabelAnchor.OUTSIDE6, TextAnchor.TOP_CENTER);
            renderer.setDefaultNegativeItemLabelPosition(position2);
            renderer.setDefaultToolTipGenerator(new StandardCategoryToolTipGenerator());

            DefaultCategoryDataset dataset = new DefaultCategoryDataset();
            int i = 0;
            for (String key : dataSeriesMap.keySet()) {
                LayeredBarChartDataSeries dataSeries = (LayeredBarChartDataSeries) dataSeriesMap.get(key);
                for (int j = 0; j < dataSeries.values.length; j++) {
                    dataset.addValue(dataSeries.values[j], dataSeries.key, (Integer) j);
                }
                if (dataSeries.barWidth > 0.0) {
                    renderer.setSeriesBarWidth(i, dataSeries.barWidth);
                }
                i++;
            }

            CategoryPlot plot = new CategoryPlot(dataset, categoryAxis, valueAxis, renderer);
            plot.setOrientation(PlotOrientation.VERTICAL);

            org.jfree.chart.JFreeChart chart = new org.jfree.chart.JFreeChart(getTitle(),
                    org.jfree.chart.JFreeChart.DEFAULT_TITLE_FONT, plot, true);
            ChartFactory.getChartTheme().apply(chart);

            ChartFrame frame = new ChartFrame(getTitle(), chart);
            return new JFreeChart(frame);
        }
    }

    abstract class AbstractDataSeries {
        protected final String key;
        protected final double[] values;

        private AbstractDataSeries(Builder<?> builder) {
            this.key = builder.key;
            this.values = builder.values;
        }

        public static abstract class Builder<B extends AbstractChartBuilder> {
            protected final B chartBuilder;

            protected String key;
            protected double[] values;

            private Builder(B chartBuilder) {
                this.chartBuilder = chartBuilder;
            }

            public abstract B done();

            public Builder<B> key(String key) {
                this.key = key;
                return this;
            }

            public Builder<B> values(double[] values) {
                this.values = values;
                return this;
            }
        }
    }

    class XYLineDataSeries extends AbstractDataSeries {
        private final Color color;
        private final float strokeWidth;
        private final boolean dashedStroke;

        private XYLineDataSeries(Builder builder) {
            super(builder);
            this.color = builder.color;
            this.strokeWidth = builder.strokeWidth;
            this.dashedStroke = builder.dashedStroke;
        }

        public static class Builder extends AbstractDataSeries.Builder<XYLineChartBuilder> {
            private Color color;
            private float strokeWidth = -1.0f;
            private boolean dashedStroke;

            private Builder(XYLineChartBuilder chartBuilder) {
                super(chartBuilder);
            }

            @Override
            public Builder key(String key) {
                return (Builder) super.key(key);
            }

            @Override
            public Builder values(double[] values) {
                return (Builder) super.values(values);
            }

            public Builder color(Color color) {
                this.color = color;
                return this;
            }

            public Builder strokeWidth(float strokeWidth) {
                this.strokeWidth = strokeWidth;
                return this;
            }

            public Builder dashedStroke(boolean dashedStroke) {
                this.dashedStroke = dashedStroke;
                return this;
            }

            @Override
            public XYLineChartBuilder done() {
                chartBuilder.dataSeriesMap.put(key, new XYLineDataSeries(this));
                return chartBuilder;
            }
        }
    }

    class LayeredBarChartDataSeries extends AbstractDataSeries {
        private final double barWidth;

        private LayeredBarChartDataSeries(Builder builder) {
            super(builder);
            this.barWidth = builder.barWidth;
        }

        public static class Builder extends AbstractDataSeries.Builder<LayeredBarChartBuilder> {
            private double barWidth;

            private Builder(LayeredBarChartBuilder chartBuilder) {
                super(chartBuilder);
            }

            @Override
            public Builder key(String key) {
                return (LayeredBarChartDataSeries.Builder) super.key(key);
            }

            @Override
            public Builder values(double[] values) {
                return (LayeredBarChartDataSeries.Builder) super.values(values);
            }

            public Builder barWidth(double barWidth) {
                this.barWidth = barWidth;
                return this;
            }

            @Override
            public LayeredBarChartBuilder done() {
                chartBuilder.dataSeriesMap.put(key, new LayeredBarChartDataSeries(this));
                return chartBuilder;
            }
        }
    }
}
