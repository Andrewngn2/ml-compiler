#include "tensor.h"
#include <iostream>
#include <stdexcept> //for throwing errors
//this is the logic for the tensor

//tensor is scope and this is the constructor which uses a member intializer list
tensor::tensor(std::vector<float> data, std::vector<int> shape) 
    : data(data), shape(shape){}
    
// prints the tensor   
void tensor::print()const{
    std::cout <<"Shape: ";

    for (int dim : shape){
        std::cout << dim << " ";
    }

    std::cout <<"Data: ";

    for (float value : data){
        std::cout << value <<" ";
    }
    
}
const std::vector<float>& tensor::getData() const {
    return data;
}
const std::vector<int>& tensor::getShape() const{
    return shape;
}


tensor matMul2d(tensor a, tensor b){

    int a_num_columns = a.getShape()[1];
    int b_num_rows = b.getShape()[0];
    int a_num_rows= a.getShape()[0];
    int b_num_columns = b.getShape()[1];
    std::vector<float> a_data = a.getData();
    std::vector<float> b_data = b.getData();
    std::vector<float> data;
    if (a_num_columns == b_num_rows){
        

        for (int row_a =0;row_a< a.getShape()[0]; ++row_a){

            for (int col_b = 0; col_b<b_num_columns; ++col_b){
                float sum = 0;
                for(int i = 0; i < a_num_columns; ++i){
                   float pair =  a_data[row_a *a_num_columns + i] * b_data[i * b_num_columns + col_b];
                    sum += pair;
                }
                
                   
                data.push_back(sum);
            }
            
        }

        
        
        

        return tensor(data, {a_num_rows, b_num_columns});
    }

    throw std::invalid_argument("Matrix dimensions do not match for matmul");
}