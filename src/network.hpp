#include<iostream>
#include<cmath>
#include<fstream>
#include"layer.hpp"
#include"matrix.hpp"
using namespace std;

class network{
    private:
        vector<Layer> layers;
    public:
        void addlayer(Layer layer);
        matrix forward(const matrix& input);
        void backword(const matrix& output, const matrix& label);
        void update(double learning_rate);
        double computeLoss(const matrix& output, const matrix& label);
        void save(string filename) const;
        void load(string filename);
        void setTraining(bool mode);
};

void network::addlayer(Layer layer){
    this->layers.push_back(layer);
}

matrix network::forward(const matrix& input){
    matrix output=input;
    for(int i=0;i<this->layers.size();i++){
        output=layers[i].forward(output);
    }
    return output;
}

double network::computeLoss(const matrix& output, const matrix& label){
    double result=0.0;
    for(int i=0;i<output.getRows();i++){
        result+=label(i,0)*log(output(i,0));
    }
    return -result;
}

void network::backword(const matrix& output, const matrix &label){
    matrix da=output-label;
    matrix dx=da;
    for(int i=layers.size()-1;i>=0;i--){
        dx=layers[i].backword(dx);
    }
}

void network::update(double learning_rate){
    for(int i=0;i<layers.size();i++){
        layers[i].update(learning_rate);
    }
}

void network::save(string filename) const{
    ofstream file(filename,ios::binary);
    for(int i=0;i<layers.size();i++){
        layers[i].save(file);
    }
}
void network::load(string filename){
    ifstream file(filename,ios::binary);
    for(int i=0;i<layers.size();i++){
        layers[i].load(file);
    }
}

void network::setTraining(bool mode){
    for(int i=0;i<layers.size();i++)
        layers[i].setTraining(mode);
}