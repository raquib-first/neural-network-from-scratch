// // // // #include<iostream>
// // // // #include "matrix.hpp"
// // // // using namespace std;
// // // // int main(){
// // // //     matrix A(2,2,0.0);
// // // //     matrix B(2,2,0.0);
// // // //     matrix C(2,3,0.0);

// // // //     A(0,0)=1; A(0,1)=2;
// // // //     A(1,0)=3; A(1,1)=4;

// // // //     B(0,0)=5; B(0,1)=6;
// // // //     B(1,0)=7; B(1,1)=8;

// // // //     C(0,0)=1; C(0,1)=2; C(0,2)=3;
// // // //     C(1,0)=4; C(1,1)=5; C(1,2)=6;

// // // //     matrix sum=A+B;
// // // //     sum.print();

// // // //     matrix sum1=A*B;
// // // //     sum1.print();

// // // //     matrix sum3=C.transpose();
// // // //     sum3.print();

// // // //     matrix sum4=A.hamadard(B);
// // // //     sum4.print();
    
// // // //     cout<<C.shape();
// // // // }


// // // #include "layer.hpp"

// // // int main(){
// // //     // // create a layer: 4 inputs, 3 outputs, relu
// // //     // Layer L1(4, 3, "relu");

// // //     // // create a (4x1) input
// // //     // matrix input(4, 1, 0.0);
// // //     // input(0,0) = 1.0;
// // //     // input(1,0) = 2.0;
// // //     // input(2,0) = 3.0;
// // //     // input(3,0) = 4.0;

// // //     // // forward pass
// // //     // matrix output = L1.forward(input);

// // //     // // print result
// // //     // cout << "Output shape: " << output.shape() << "\n";
// // //     // output.print();

// // //     // 1. Create two layers
// // //     Layer L1(4, 3, "relu");
// // //     Layer L2(3, 2, "softmax");

// // //     // 2. Create input (4x1)
// // //     matrix input(4, 1, 0.0);
// // //     input(0,0)=1.0; input(1,0)=2.0;
// // //     input(2,0)=3.0; input(3,0)=4.0;

// // //     // 3. Forward pass
// // //     matrix a1 = L1.forward(input);
// // //     matrix a2 = L2.forward(a1);

// // //     // 4. Fake gradient (pretend loss derivative)
// // //     // real label y = [1, 0] (one-hot)
// // //     matrix y(2, 1, 0.0);
// // //     y(0,0) = 1.0;

// // //     // 5. Output gradient = ŷ - y
// // //     matrix da2 = a2 - y;

// // //     // 6. Backward pass
// // //     matrix da1 = L2.backword(da2);
// // //     matrix da0 = L1.backword(da1);

// // //     // 7. Update weights
// // //     L2.update(0.01);
// // //     L1.update(0.01);

// // //     cout << "Forward output:\n"; a2.print();
// // //     cout << "Gradient into L1:\n"; da0.print();
// // // }

// // #include "network.hpp"

// // int main(){
// //     // build network
// //     network net;
// //     net.addlayer(Layer(4, 8, "relu", optimizer("adam", 0.001)));
// //     net.addlayer(Layer(8, 3, "softmax", optimizer("SGD", 0.001)));

// //     // net.addlayer(Layer(4, 8, "relu"));
// //     // net.addlayer(Layer(8, 3, "softmax"));

// //     // input (4×1)
// //     matrix input(4, 1, 0.0);
// //     input(0,0)=1.0; input(1,0)=2.0;
// //     input(2,0)=3.0; input(3,0)=4.0;

// //     // label (3×1) one-hot — correct class is index 1
// //     matrix label(3, 1, 0.0);
// //     label(1,0) = 1.0;

// //     // run 100 iterations and print loss
// //     for(int i=0; i<100; i++){
// //         matrix output = net.forward(input);
// //         double loss   = net.computeLoss(output, label);
// //         net.backword(output, label);
// //         net.update(0.01);
// //         if(i % 10 == 0)
// //             cout << "iteration " << i << " loss: " << loss << "\n";
// //     }
// // }

// #include "network.hpp"

// int main(){
//     network net;
//     net.addlayer(Layer(4, 8, "relu", optimizer("adam", 0.01), 0.2));  // 20% dropout
//     net.addlayer(Layer(8, 3, "softmax"));

//     matrix input(4, 1, 0.0);
//     input(0,0)=1.0; input(1,0)=2.0;
//     input(2,0)=3.0; input(3,0)=4.0;

//     matrix label(3, 1, 0.0);
//     label(1,0) = 1.0;

//     // train
//     net.setTraining(true);
//     for(int i=0; i<100; i++){
//         matrix output = net.forward(input);
//         double loss   = net.computeLoss(output, label);
//         net.backword(output, label);
//         net.update(0.01);
//         if(i%10==0) cout << "iter " << i << " loss: " << loss << "\n";
//     }

//     // save
//     net.save("model.bin");
//     cout << "Model saved.\n";

//     // load into new network and run inference
//     network net2;
//     net2.addlayer(Layer(4, 8, "relu", optimizer("adam", 0.01), 0.2));
//     net2.addlayer(Layer(8, 3, "softmax"));
//     net2.load("model.bin");
//     net2.setTraining(false);

//     matrix output2 = net2.forward(input);
//     cout << "Inference output:\n";
//     output2.print();
// }
#include<algorithm>
#include<chrono>
#include"network.hpp"
#include"mnist.hpp"

matrix oneHot(int label, int classes){
    matrix m(classes, 1, 0.0);
    m(label, 0) = 1.0;
    return m;
}

int main(){
    auto start = chrono::high_resolution_clock::now();

    vector<int> indices(60000);
    iota(indices.begin(), indices.end(), 0);  // fills 0,1,2,...,59999

    vector<matrix> train_images=loadImages("train-images.idx3-ubyte");
    vector<int>    train_labels=loadLabels("train-labels.idx1-ubyte");
    vector<matrix> test_images =loadImages("t10k-images.idx3-ubyte");
    vector<int>    test_labels =loadLabels("t10k-labels.idx1-ubyte");

    network net;
    net.addlayer(Layer(784,128,"relu"));
    net.addlayer(Layer(128,64,"relu"));
    net.addlayer(Layer(64,10,"softmax"));

    for(int epoch=0;epoch<3;epoch++){
        shuffle(indices.begin(), indices.end(), mt19937(random_device{}()));

        double loss=0;
        for(int j=0;j<train_images.size();j++){
            matrix input=train_images[indices[j]];
            matrix label=oneHot(train_labels[indices[j]], 10);
            matrix output=net.forward(input);
            loss +=net.computeLoss(output, label);
            net.backword(output, label);
            net.update(0.01);
        }
        cout <<"epoch "<< epoch <<" avg loss: "<<loss/train_images.size()<<"\n";
    }

        // test accuracy
    net.setTraining(false);
    int correct = 0;
    for(int i=0; i<test_images.size(); i++){
        matrix output = net.forward(test_images[i]);
        int predicted = 0;
        for(int j=1; j<10; j++)
            if(output(j,0) > output(predicted,0)) predicted = j;
        if(predicted == test_labels[i]) correct++;
    }
    cout << "accuracy: " << (correct/10000.0)*100 << "%\n";

    // save
    net.save("model.bin");
    auto end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::seconds>(end-start).count() << "s\n";
}