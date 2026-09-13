#include "tensor.h"
#include <iostream>
#include <tuple>
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
    std::vector<float> data(a_num_rows * b_num_columns, 0.0f);
    if (a_num_columns == b_num_rows){
        
            //contiguous memory access
        for (int row = 0; row < a_num_rows; ++row) {

            for (int i = 0; i < a_num_columns; ++i) {

                float a_value =
                    a_data[row * a_num_columns + i];

                for (int col = 0; col < b_num_columns; ++col) {

                    data[row * b_num_columns + col] +=
                        a_value *
                        b_data[i * b_num_columns + col];
                }
            }
        }

        
        
        

        return tensor(data, {a_num_rows, b_num_columns});
    }

    throw std::invalid_argument("Matrix dimensions do not match for matmul");
}

tensor fusedMatMulAdd2d(tensor a, tensor b, tensor bias) {

    int a_num_rows = a.getShape()[0];
    int a_num_columns = a.getShape()[1];

    int b_num_rows = b.getShape()[0];
    int b_num_columns = b.getShape()[1];

    if (a_num_columns != b_num_rows) {
        throw std::invalid_argument(
            "Matrix dimensions do not match for fused matmul + add"
        );
    }

    if (bias.getShape()[0] != a_num_rows ||
        bias.getShape()[1] != b_num_columns) {
        throw std::invalid_argument(
            "Bias shape does not match matmul output"
        );
    }

    std::vector<float> a_data = a.getData();
    std::vector<float> b_data = b.getData();
    std::vector<float> bias_data = bias.getData();

    std::vector<float> data(a_num_rows * b_num_columns, 0.0f);

    for (int row = 0; row < a_num_rows; ++row) {

        for (int i = 0; i < a_num_columns; ++i) {

            float a_value =
                a_data[row * a_num_columns + i];

            for (int col = 0; col < b_num_columns; ++col) {

                data[row * b_num_columns + col] +=
                    a_value *
                    b_data[i * b_num_columns + col];
            }
        }
    }

    // add bias
    for (int row = 0; row < a_num_rows; ++row) {
        for (int col = 0; col < b_num_columns; ++col) {
            data[row * b_num_columns + col] +=
                bias_data[row * b_num_columns + col];
        }
    }

    return tensor(data, {a_num_rows, b_num_columns});
}

tensor naiveMatMul(tensor a, tensor b) {

    int a_num_rows = a.getShape()[0];
    int a_num_columns = a.getShape()[1];

    int b_num_rows = b.getShape()[0];
    int b_num_columns = b.getShape()[1];

    if (a_num_columns != b_num_rows) {
        throw std::invalid_argument(
            "Matrix dimensions do not match for matmul"
        );
    }

    std::vector<float> a_data = a.getData();
    std::vector<float> b_data = b.getData();

    std::vector<float> data;

    for (int row = 0; row < a_num_rows; ++row) {

        for (int col = 0; col < b_num_columns; ++col) {

            float sum = 0;

            for (int i = 0; i < a_num_columns; ++i) {

                sum +=
                    a_data[row * a_num_columns + i] *
                    b_data[i * b_num_columns + col];
            }

            data.push_back(sum);
        }
    }

    return tensor(data, {a_num_rows, b_num_columns});
}



#include <chrono>
#include <tuple>
#include <limits>

// Ensure matMul2d accepts inputs by const reference: matMul2d(const tensor&, const tensor&)
std::tuple<long long, long long, long long> benchmarkMatMul(const tensor& input, const tensor& weight, int runs, int naive) {
    if (runs <= 0) return {0, 0, 0};

    long long total_time = 0;
    long long max_time = 0;
    long long min_time = std::numeric_limits<long long>::max(); // Start at infinity
    if(naive == 0){
    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        tensor result = matMul2d(input, weight);

        auto end = std::chrono::high_resolution_clock::now();
        
        long long current_time = std::chrono::duration_cast<
            std::chrono::microseconds
        >(end - start).count();

        total_time += current_time;

        if (current_time > max_time) {
            max_time = current_time;
        }
        if (current_time < min_time) {
            min_time = current_time;
        }
    }
    }else{
        for(int i =0; i<runs;++i){

        
        auto start = std::chrono::high_resolution_clock::now();

        tensor result = naiveMatMul(input, weight);

        auto end = std::chrono::high_resolution_clock::now();
        
        long long current_time = std::chrono::duration_cast<
            std::chrono::microseconds
        >(end - start).count();

        total_time += current_time;

        if (current_time > max_time) {
            max_time = current_time;
        }
        if (current_time < min_time) {
            min_time = current_time;
        }
    }


    }

    long long average_time = total_time / runs;

    return {average_time, max_time, min_time};
}

