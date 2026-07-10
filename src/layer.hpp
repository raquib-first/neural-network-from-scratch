#pragma once
#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include "matrix.hpp"
#include "optimizer.hpp"
using namespace std;

class Layer{
    private:
        matrix x;
        matrix w;
        matrix b;
        matrix z;
        matrix a;
        string activation;
        matrix dw;
        matrix db;
        optimizer opt;
        double dropout_rate;
        bool training;
        matrix dropout_mask;
    public:
        Layer(size_t input_size, size_t output_size, string activation, optimizer opt=optimizer(), double dropout_rate=0.0);
        matrix relu(const matrix& a);
        matrix sigmoid(const matrix& a);
        matrix softmax(const matrix& a);
        matrix forward(const matrix& input);
        matrix getA();
        matrix getZ();
        matrix backword(const matrix& da);
        void update(double learning_rate);
        matrix reluDerivative(const matrix& z);
        void save(ofstream& file) const;
        void load(ifstream& file);
        void setTraining(bool mode);
};

Layer::Layer(size_t input_size, size_t output_size, string activation,optimizer opt, double dropout_rate){
    this->w=matrix::random(output_size,input_size);
    this->b=matrix(output_size,1,0.0);
    this->activation=activation;
    this->opt=opt;
    this->dropout_rate=dropout_rate;
    this->training=true;
}
matrix Layer::relu(const matrix& a){    
    size_t Arows=a.getRows();
    size_t Acols=a.getCols();
    matrix result(Arows,Acols,0.0);
    for(int i=0;i<Arows;i++){
        for(int j=0;j<Acols;j++){
            if(a(i,j)>0){
                result(i,j)=a(i,j);
            } //no need for else because already initialized with 0.
        }
    }
    return result;
}

matrix Layer::sigmoid(const matrix& a){    
    size_t Arows=a.getRows();
    size_t Acols=a.getCols();
    matrix result(Arows,Acols,0.0);
    for(int i=0;i<Arows;i++){
        for(int j=0;j<Acols;j++){            
            result(i,j)=1/(1+exp(-a(i,j))); 
        }
    }
    return result;
}

matrix Layer::softmax(const matrix& a){    
    size_t Arows=a.getRows();
    size_t Acols=a.getCols();
    matrix result(Arows,Acols,0.0);
    double sum=0.0;
    double maxval=a(0,0);
    
    for(int i=0;i<Arows;i++){ //finding max value.
        for(int j=0;j<Acols;j++){
            maxval=max(maxval,a(i,j));
        }
    }

    for(int i=0;i<Arows;i++){
        for(int j=0;j<Acols;j++){
            sum+=exp(a(i,j)-maxval); //subtracting to maxvalue because if sum is high exponential can be overflows to infinity.
        }
    }

    for(int i=0;i<Arows;i++){
        for(int j=0;j<Acols;j++){
            result(i,j)=exp(a(i,j)-maxval)/sum;
        }
    }
    return result;
}

matrix Layer::forward(const matrix& input){
    this->x=input;
    this->z=(this->w) * (input) + this->b;
    if(activation=="relu"){
        this->a=relu(this->z);
    }
    else if(activation=="sigmoid"){
        this->a=sigmoid(this->z);
    }
    else{
        this->a=softmax(this->z);
    }

    if(training && dropout_rate > 0.0){
        // Step 1 — generate mask
        mt19937 gen(random_device{}());
        uniform_real_distribution<double> dist(0.0, 1.0);
        dropout_mask = matrix(a.getRows(), a.getCols(), 0.0);
        for(size_t i=0;i<a.getRows();i++){
            for(size_t j=0;j<a.getCols();j++){
                dropout_mask(i,j)=dist(gen) > dropout_rate ? 1.0 : 0.0;
            }
        }
        // Step 2 — apply mask
        this->a = this->a.hadamard(dropout_mask);

        // Step 3 — scale
        this->a = this->a * (1.0 / (1.0 - dropout_rate));
    }

    return this->a;
}

matrix Layer::getA(){
    return this->a;
}

matrix Layer::getZ(){
    return this->z;
}

matrix Layer::backword(const matrix& da){
    matrix dz(z.getRows(),z.getCols(),0.0);
    matrix dx(x.getRows(),x.getCols(),0.0);
    if(this->activation=="relu"){
        dz=da.hadamard(reluDerivative(this->z));
    }
    else if(this->activation=="softmax"){
        dz=da;
    }
    this->dw=dz*(this->x.transpose());
    this->db=dz;
    dx=(this->w.transpose())*dz;
    return dx;
}

void Layer::update(double learning_rate){
    opt.update(w,b,dw,db);
}

matrix Layer::reluDerivative(const matrix& z){
    matrix result(z.getRows(),z.getCols(),0.0);
    for(size_t i=0;i<z.getRows();i++)
        for(size_t j=0;j<z.getCols();j++)
            result(i,j)=z(i,j) > 0 ? 1.0 : 0.0;
    return result;
}

void Layer::save(ofstream& file) const {
    w.save(file);
    b.save(file);
}

void Layer::load(ifstream& file) {
    w.load(file);
    b.load(file);
}

void Layer::setTraining(bool mode){
    this->training = mode;
}