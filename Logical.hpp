//
// Created by LeiLei on 2020/6/25.
//

#ifndef Logical_Hpp
#define Logical_Hpp
#include <Matrix.hpp>
//��������߼��ع�
class Logical{
public:
    //Ĭ�Ϲ��캯��
    Logical();
    //�����趨ѧϰ��
    explicit Logical(const float&rate);
    //�����趨ѭ����������
    explicit Logical(const int &cycle);
    //ȫ�����Զ���Ĺ��캯��
    Logical(const float&rate, const int &cycle);
    //����ѵ��
    void train(Matrix &sample, Matrix &label);
    //Ԥ�����ݵķ���
    void predict(Matrix & sample);
private:
    float rate;                     //ѧϰ�ʣ�Ĭ�ϲ�����ʱ���趨Ϊ0.01
    int cycles;                     //ѭ������������Ĭ�ϲ����趨Ϊ100��
    vector<vector<double>>w{}; //Ȩ�ع��������飬����֮��ת��ɶ�Ӧ�ľ���
    //���¹������ݼ��������ݼ����һ��Ϊ1
    virtual Matrix remake(Matrix &sample);
    //ʹ���ݶ��½��㷨����Ȩֵ����Ļ�ȡ
    void gradient(Matrix &sample, Matrix &label);
};

#endif //Logical_Hpp
