package com.codingapi.deeplearning.demo03;

import com.codingapi.deeplearning.demo03.db.RegressionFunctionExampleData;
import com.codingapi.deeplearning.demo03.learn.GradientDescentAlgorithmFunction;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
@Slf4j
class DeepLearningJavaDemo03ApplicationTests {

    @Autowired
    private RegressionFunctionExampleData regressionFunctionExampleData;


    @Test
    void train(){
        INDArray data = regressionFunctionExampleData.loadData();
        GradientDescentAlgorithmFunction gradientDescentAlgorithmFunction = new GradientDescentAlgorithmFunction(0.01,20000,data);
        gradientDescentAlgorithmFunction.train();

        INDArray test = Nd4j.create(1,5);
        test.putRow(1,Nd4j.create(new double[]{1,21,62,55,1}));

        System.out.println(gradientDescentAlgorithmFunction.hypothesisFunction(test));

    }
}
