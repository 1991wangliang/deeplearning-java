package com.codingapi.deeplearning.demo03.db;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Component;

import java.io.IOException;

/**
 * 数据操作工具类
 * @author lorne
 * @date 2019-10-22
 * @description y = ax+b
 */
@Component
public class RegressionFunctionExampleData {

    private String filePath = "init/demo03.csv";


    public INDArray loadData(){
        try {
            INDArray data =  Nd4j.readNumpy(filePath,",");
            INDArray ones = Nd4j.ones(data.rows(),1);
            return Nd4j.concat(1,ones,data);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

}
