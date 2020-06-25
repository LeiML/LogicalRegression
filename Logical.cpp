//
// Created by LeiLei on 2020/6/25.
//

#include "Logical.hpp"

Logical::Logical() : rate(0.01), cycles(100){}

Logical::Logical(const float &rate) : rate(rate), cycles(100){}

Logical::Logical(const int &cycle) : rate(0.01), cycles(cycle){}

Logical::Logical(const float &rate, const int &cycle) : rate(rate), cycles(cycle){}

Matrix Logical::remake(Matrix &sample) {
    Matrix result = Matrix(sample.row, sample.col+1);
    for(int i=0;i<result.row;i++){
        for (int j=0;j<result.col;j++)
            result.at(i, j) = (j==0)?1:sample.at(i, j-1);
    }
    return result;
}

void Logical::gradient(Matrix &sap, Matrix &label) {
    //重构样本矩阵
    Matrix sample = this->remake(sap);
    //初始化权值矩阵，权值矩阵等于样本的列数，列向量
    Matrix weight = Matrix::ones(sample.col, 1);
    for(int i=0;i<this->cycles;i++){
        //预测值的获取
        Matrix temp = sample.dot(weight);
        //进行sigmoid函数，进行输出值的激活
        Matrix lab = (temp.exp() + 1).exp(-1);
        //获取损失，loss=预测值-真实值
        Matrix error = lab - label;
        //更新权重
        weight = weight + sample.transpose().dot(error) * this->rate;
    }
    this->w = weight.vec();
}

void Logical::train(Matrix &sample, Matrix &label) {
    try{
        if (sample.row != label.row) throw MyException("【Error】the sample and label's Dimension is not equal");
    } catch (MyException & e) {
        cerr << e.what() << endl;
        return;
    }
    this->gradient(sample, label);
}

void Logical::predict(Matrix &sample) {
    Matrix samp = this->remake(sample);
    Matrix weight = Matrix(this->w);
    Matrix temp = samp.dot(weight);
    Matrix lab = (temp.exp() + 1).exp(-1);
    cout << lab << endl;
}
