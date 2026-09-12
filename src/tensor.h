//this is header file its job is to explain 
//what a tensor is and what it can do
 
#include <vector>
#pragma once
//library that allows for dynamic resizable array


class tensor {
    private:
    std::vector<float> data;
    std::vector<int> shape;

    public:
        tensor();
        tensor(std::vector<float> data, std::vector<int> shape);
            //print tensor without modifying variables in tensor
        void print() const;
        const std::vector<float>& getData() const;
        const std::vector<int>& getShape() const;
        tensor ReLU() const;
        
        tensor add(const tensor& other) const;
        // tensorA is protected from changing (because of the trailing const).
        //tensorB is protected from changing (because of the const& parameter).
        //A completely new tensor (tensorC) is generated and returned.
    };
    tensor naiveMatMul(tensor a, tensor b);
    tensor matMul2d(tensor a, tensor b);
    tensor fusedMatMulAdd2d(tensor a, tensor b, tensor bias);
    std::tuple<long long,long long, long long>benchmarkMatMul ( const tensor& input, const tensor& weight, int runs, int naive);
    
