/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.briarheart.neuralnet.util.math;

/**
 * @author Roman Chigvintsev
 */
public class IdentityMatrix extends Matrix {
    public IdentityMatrix(int order) {
        super(order, order);
        for (int i = 0; i < order; i++) {
            for (int j = 0; j < order; j++) {
                super.set(i, j, i == j ? 1 : 0);
            }
        }
    }

    @Override
    public void set(int i, int j, double value) {
        throw new UnsupportedOperationException();
    }
}
