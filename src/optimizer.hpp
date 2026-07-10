#pragma once
#include<iostream>
#include<cmath>
#include<string>
#include"matrix.hpp"

using namespace std;

class optimizer{
    private:
        string type;
        matrix m_w,m_b; //moment 1 for weight and bias.
        matrix v_w,v_b; //moment 2 for weight and bias.
        int t; //timestep.
        //hyperparamet
        double b1,b2,eps,lr;
    public:
        optimizer(string type="adam", double lr=0.001, double b1=0.9, double b2=0.999, double eps=1e-8);
        void update(matrix& w, matrix& b, const matrix& dw, const matrix& db);
};

optimizer::optimizer(string type,double lr, double b1, double b2, double eps){
    this->type=type;
    this->b1=b1;
    this->b2=b2;
    this->lr=lr;
    this->eps=eps;
    this->t=0;
}

void optimizer::update(matrix& w, matrix& b, const matrix& dw, const matrix& db){
    if(type=="SGD"){
        w=w - dw * lr;
        b=b - db * lr;
    }
    else{
        this->t++;
        if(this->m_w.getRows()==0){
            this->m_w=matrix(w.getRows(), w.getCols(), 0.0);
            this->v_w=matrix(w.getRows(), w.getCols(), 0.0);
            this->m_b=matrix(b.getRows(), b.getCols(), 0.0);
            this->v_b=matrix(b.getRows(), b.getCols(), 0.0);
        }

        this->m_w=m_w*b1 + dw*(1-b1);
        this->v_w=v_w*b2 + dw.hadamard(dw)*(1-b2);
        this->m_b=m_b*b1 + db*(1-b1);
        this->v_b=v_b*b2 + db.hadamard(db)*(1-b2);

        matrix m_w_hat=m_w * (1/(1-pow(b1,t)));
        matrix v_w_hat=v_w * (1/(1-pow(b2,t)));
        matrix m_b_hat = m_b * (1/(1-pow(b1,t)));
        matrix v_b_hat = v_b * (1/(1-pow(b2,t)));

        w=w - (m_w_hat / (v_w_hat.applyFunc(sqrt) + eps)) * lr;
        b=b - (m_b_hat / (v_b_hat.applyFunc(sqrt) + eps)) * lr;
    }
}
