
#include <iostream>
#include "tensor.h"

int main() {
    tensor x({1,2,3,4,5,6},{2,3});
    tensor b({1,2,3,4},{2,2});
    tensor a({10,20,30,40},{1,4});


    x.print();
    b.print();
    a.print();
    return 0;
}
