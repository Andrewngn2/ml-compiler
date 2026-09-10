//this is header file its job is to explain 
//what a tensor is and what it can do
 
#include <vector>
//library that allows for dynamic resizable array


class tensor {
    private:
    std::vector<float> data;
    std::vector<int> shape;

    public:
        tensor(std::vector<float> data, std::vector<int> shape);
            //print tensor without modifying variables in tensor
        void print() const;
        const std::vector<float>& getData() const;
        const std::vector<int>& getShape() const;
    };
