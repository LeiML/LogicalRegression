//
// Created by LeiLei on 2020/6/25.
//

#ifndef Logical_Hpp
#define Logical_Hpp
#include <Matrix.hpp>
//二分类的逻辑回归
class Logical{
public:
    //默认构造函数
    Logical();
    //单独设定学习率
    explicit Logical(const float&rate);
    //单独设定循环迭代次数
    explicit Logical(const int &cycle);
    //全部都自定义的构造函数
    Logical(const float&rate, const int &cycle);
    //进行训练
    void train(Matrix &sample, Matrix &label);
    //预测数据的分类
    void predict(Matrix & sample);
private:
    float rate;                     //学习率，默认参数的时候设定为0.01
    int cycles;                     //循环迭代次数，默认参数设定为100次
    vector<vector<double>>w{}; //权重构建的数组，用于之后转变成对应的矩阵
    //重新构建数据集，给数据集添加一列为1
    virtual Matrix remake(Matrix &sample);
    //使用梯度下降算法进行权值矩阵的获取
    void gradient(Matrix &sample, Matrix &label);
};

#endif //Logical_Hpp
