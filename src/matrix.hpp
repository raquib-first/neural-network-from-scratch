#pragma once
#include<iostream>
#include<vector>
#include<string>
#include<random>
#include<fstream>
using namespace std;

class matrix{
    private:
        vector<double> data;
        size_t rows;
        size_t cols;
    public:
        matrix();
        matrix(size_t rows, size_t cols, double init=0.0);
        double& operator()(size_t i, size_t j);               // for non-const objects
        const double& operator()(size_t i, size_t j) const;   // for const objects
        matrix operator+(const matrix& other) const;
        matrix operator+(double scalar) const;
        matrix operator-(const matrix& other) const;
        matrix operator*(const matrix& other) const;
        matrix operator*(double scalar) const;
        matrix operator/(const matrix& other) const;
        matrix transpose() const;
        matrix hadamard(const matrix& other) const;
        string shape() const;
        void print() const;
        static matrix random(size_t rows, size_t cols);  // He init
        size_t getRows() const;
        size_t getCols() const;
        matrix applyFunc(double(*func)(double)) const;
        void save(ofstream& file) const;
        void load(ifstream& file);
};

matrix::matrix() : rows(0), cols(0) {}

matrix::matrix(size_t rows, size_t cols, double init){
    this->rows=rows;
    this->cols=cols;
    data=vector<double>(rows*cols,init);
}

double& matrix::operator()(size_t i,size_t j){
    return data[i*cols+j];
}

const double& matrix::operator()(size_t i,size_t j) const{
    return data[i*cols+j];
}

matrix matrix::operator+(const matrix& other) const{
    if(other.rows!=this->rows || other.cols!=this->cols){
        throw runtime_error("shape mismatch");
    }
    matrix result(rows,cols,0.0);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            result(i,j)=(*this)(i,j)+other(i,j);
        }
    }
    return result;
}

matrix matrix::operator+(double scalar) const{
    matrix result(rows,cols,0.0);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            result(i,j)=(*this)(i,j)+scalar;
        }
    }
    return result;
}

matrix matrix::operator-(const matrix& other) const{
    if(other.rows!=this->rows || other.cols!=this->cols){
        throw runtime_error("shape mismatch");
    }
    matrix result(rows,cols,0.0);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            result(i,j)=(*this)(i,j)-other(i,j);
        }
    }
    return result;
}

matrix matrix::operator*(const matrix& other) const{
    if(other.rows!=this->cols){
        throw runtime_error("shape mismatch");
    }
    matrix result(rows,other.cols,0.0);
    for(int i=0;i<rows;i++){
        for(int j=0;j<other.cols;j++){
            double ret=0;
            for(int k=0;k<cols;k++){
                ret+=(*this)(i,k)*other(k,j);
            }
            result(i,j)=ret;
        }
    }    
    return result;
}
matrix matrix::operator*(double scalar) const{
    matrix result(rows,cols,0.0);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            result(i,j)=(*this)(i,j)*scalar;
        }
    }
    return result;
}

matrix matrix::operator/(const matrix& other) const{
    if(other.rows!=this->rows || other.cols!=this->cols){
        throw runtime_error("shape mismatch");
    }
    matrix result(rows,cols,0.0);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            result(i,j)=(*this)(i,j)/other(i,j);
        }
    }
    return result;
}

matrix matrix::transpose() const{
    matrix result(cols,rows,0.0);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            result(j,i)=(*this)(i,j);
        }
    }
    return result;
}

matrix matrix::hadamard(const matrix& other) const{
    if(other.rows!=this->rows || other.cols!=this->cols){
        throw runtime_error("shape mismatch");
    }
    matrix result(rows,cols,0.0);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            result(i,j)=(*this)(i,j)*other(i,j);
        }
    }
    return result;
}

string matrix::shape() const{
    string result="";
    result=to_string(rows)+"*"+to_string(cols);
    return result;
}

void matrix::print() const{
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            cout << (*this)(i,j) << " ";
        }
        cout << "\n";
    }
    cout<<"\n";
    cout<<"\n";
}

matrix matrix::random(size_t rows, size_t cols){
    matrix result(rows,cols,0.0);
    mt19937 gen(42);   // 42 is the seed, fixed for reproducibility
    normal_distribution<double> dist(0.0, 1.0);
    double scale = sqrt(2.0 / cols);  // He initialization scale
    for(size_t i = 0; i < rows; i++)
        for(size_t j = 0; j < cols; j++)
            result(i,j) = dist(gen)*scale;
    return result;
}

size_t matrix::getRows() const { return rows; }
size_t matrix::getCols() const { return cols; }

matrix matrix::applyFunc(double(*func)(double)) const{
    matrix result(rows, cols, 0.0);
    for(size_t i=0;i<rows;i++)
        for(size_t j=0;j<cols;j++)
            result(i,j) = func((*this)(i,j));
    return result;
}

void matrix::save(ofstream& file) const{

    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(data.data()), rows*cols*sizeof(double));

}

void matrix::load(ifstream& file){

    file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    data.resize(rows * cols);
    file.read(reinterpret_cast<char*>(data.data()), rows*cols*sizeof(double));
    
}