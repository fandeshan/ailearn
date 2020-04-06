package com.fds.neural;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LinearRegression {
    public double lt=0.01;
    public int iterations = 10;
    public double w = 0.0;
    public double b = 0.0;
    public LinearRegression(){}
    public LinearRegression(double lt,int iterations){
        this.iterations = iterations;
        this.lt = lt;
    }
    public double coustFun(LinearRegression self,double x[],double y[],double weight,double bias){
        int n = x.length;
        double totalError = 0.0;
        for (int i = 0;i < n; i++){
            totalError += Math.pow((y[i] - (weight*x[i]+bias)),2);
        }
        return totalError/2;
    }

    public Map<String,Double> updateWeight(LinearRegression self,double x[],double y[]
            ,double weight,double bias,double learningRate){
        double dw = 0;
        double db = 0;
        int n = x.length;
        for (int i = 0;i < n; i++){
            dw += -2*x[i]*(y[i]-(weight*x[i]+bias));
            db += -2*(y[i]-(weight*x[i]+bias));
        }
        weight -=(dw/n)*learningRate;
        bias -= (db/n)*learningRate;
        Map map = new HashMap();
        map.put("weight",weight);
        map.put("bias",bias);
        return map;
    }

    public Map<String,Object> fit(LinearRegression self,double x[],double y[]){
        List<Double> costHistory = new ArrayList();
        for (int i = 0;i < self.iterations; i++){
            Map<String,Double> map = self.updateWeight(self,x,y,self.w,self.b,self.lt);
            self.w = map.get("weight");
            self.b = map.get("bias");
            double cost = self.coustFun(self,x,y,self.w,self.b);
            costHistory.add(cost);
            if(i%10==0){
                System.out.printf("iter=%d,weight=%f,bias = %f,cost=%f",i,self.w,self.b,cost);
                System.out.println();
            }
        }
        Map<String,Object> res = new HashMap<>();
        res.put("weight",self.w);
        res.put("bias",self.b);
        res.put("bias",costHistory);
        return res;
    }

    public double predict(LinearRegression self,double x){
        x = (x+100)/200;
        return self.w*x + b;
    }

    public static void main(String[] args) {
        double x[] = {1,2,3,10,20,50,100,-2,-10,-100,-5,-20};
        double y[] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0};
        LinearRegression model = new LinearRegression(0.01,500);
        double X[] = new double[x.length];
        for (int i = 0;i <X.length;i++){
            X[i] = (x[i]+100)/200;
        }
        model.fit(model,X,y);
        double test_x[] = {90,80,81,82,75,40,32,15,5,1,-1,-15,-20,-22,-33,-45,-60,-90};
        for(int i = 0 ;i<test_x.length;i++ ){
            System.out.printf("input :%f ==> predict:%f", test_x[i],model.predict(model,test_x[i]));
            System.out.println();
        }
    }
}
