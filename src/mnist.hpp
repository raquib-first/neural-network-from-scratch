#pragma once
#include<iostream>
#include<fstream>
#include"matrix.hpp"
using namespace std;

int reverseInt(int i){
    unsigned char c1,c2,c3,c4;
    c1=i&255; c2=(i>>8)&255; c3=(i>>16)&255; c4=(i>>24)&255;
    return ((int)c1<<24)+((int)c2<<16)+((int)c3<<8)+c4;
}

vector<matrix> loadImages(string filename){
    ifstream file(filename,ios::binary);
    if(!file) throw runtime_error("cannot open "+ filename);

    int magic,numImages,rows,cols;
    file.read(reinterpret_cast<char*>(&magic), sizeof(int));
    file.read(reinterpret_cast<char*>(&numImages), sizeof(int));
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    magic=reverseInt(magic);
    numImages=reverseInt(numImages);
    rows=reverseInt(rows);
    cols=reverseInt(cols);
    
    vector<matrix> images;
    for(int i=0;i<numImages;i++){
        matrix img(rows*cols,1,0.0);
        for(int j=0;j<rows*cols;j++){
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            img(j, 0)=pixel/255.0;
        }
        images.push_back(img);
    }
    return images;
}

vector<int> loadLabels(string filename){
    ifstream file(filename,ios::binary);
    if(!file) throw runtime_error("cannot open "+ filename);

    int magic,numLabels;
    file.read(reinterpret_cast<char*>(&magic), sizeof(int));
    file.read(reinterpret_cast<char*>(&numLabels), sizeof(int));
    magic=reverseInt(magic);
    numLabels=reverseInt(numLabels);
    
    vector<int> labels;
    for(int i=0;i<numLabels;i++){
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels.push_back((int)label);
    }
    return labels;
}
