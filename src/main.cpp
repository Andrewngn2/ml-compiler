
#include <iostream>
#include "tensor.h"

int main() {
    //illustrate y = ReLu(xW+b)
    tensor input_tensor({1,2,3,4},{2,2});

    tensor weight_tensor({6,7,6,7},{2,2});

    tensor bias_tensor({1,2,1,2},{2,2});

    tensor result = matMul2d(input_tensor, weight_tensor);
    result.print();

    result = result.add(bias_tensor);
    result.print();
    result = result.relu();

    result.print();



    

    return 0;
}
