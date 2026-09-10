#include "tensor.h"
#include <iostream>
#include <stdexcept> //for throwing errors
//this is the logic for the tensor

//tensor is scope and this is the constructor which uses a member intializer list
tensor::tensor(std::vector<float> data, std::vector<int> shape) 
    : data(data), shape(shape){}
tensor::tensor(){}
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

tensor tensor::ReLU() const{
    std::vector<float> new_data;

    for(float element : data)
    {
        if(element <0){
            new_data.push_back(0);
        }else{
            new_data.push_back(element);
        }
    }
    return tensor(new_data,shape);
}

tensor tensor::add(const tensor& other) const{
    int a_num_rows = shape[0];
    int b_num_rows = other.shape[0];
    int a_num_columns =shape[1];
    int b_num_columns = other.shape[1];

    if (a_num_rows == b_num_rows && a_num_columns == b_num_columns)
        {
            std::vector<float> new_data;
            for(int row=0; row<a_num_rows;++row)
                {
                    for(int col= 0; col<a_num_columns; ++col){
                        float new_element = data[row * a_num_columns + col] + other.getData()[row * b_num_columns + col];
                       new_data.push_back(new_element);
                    }

                }
            
        return tensor(new_data, {a_num_rows, a_num_columns});
        }
    throw std::invalid_argument("Tensor shapes do not match for addition");
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