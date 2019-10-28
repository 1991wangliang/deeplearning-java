package com.codingapi.deeplearning.demo02;

import com.codingapi.deeplearning.demo02.db.RegressionFunctionExampleData;
import com.codingapi.deeplearning.demo02.learn.GradientDescentAlgorithmFunction;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
@Slf4j
class DeepLearningJavaDemo02ApplicationTests {

    @Autowired
    private RegressionFunctionExampleData regressionFunctionExampleData;


    @Test
    void randomData() {
        regressionFunctionExampleData.randData(100,15,300);
    }


    @Test
    void train(){
        INDArray exampleDataArrays =  regressionFunctionExampleData.loadData();
        GradientDescentAlgorithmFunction gradientDescentAlgorithmFunction
                = new GradientDescentAlgorithmFunction(0.01,10000,exampleDataArrays);
        gradientDescentAlgorithmFunction.train();
    }
}
