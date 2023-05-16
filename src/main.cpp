#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <iostream>
#include <vector>
#include <any>
#include <memory>

#include "LinearLayer.h"
#include "ReLU.h"
#include "Sequential.h"
#include "OptimizerTypeErasure.h"
#include "CriterionTypeErasure.h"

namespace {
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    template<typename T>
    int foo(T x) {
        int y = x;
        y++;
        return y;
    }
    
    template<typename T>
    void print(std::vector<T> x) {
        std::cout << "size: " << x.size() << std::endl;
        for (auto &y : x) {
            std::cout << *y << "\n----\n";
        }
        std::cout << std::endl;
    }
    
    template<typename... T>
    void print(T... x) {
        std::vector<int> result;
        auto i = {result.push_back(x)...};
        for (int i : result) {
            std::cout << i << ' ';
        }
        std::cout << std::endl;
    }
}

int main() {
    Layers::LinearLayer linearLayer(2, 3, true, 45);
    Eigen::Rand::P8_mt19937_64 urng = {42};
    Base::Matrix x = Eigen::Rand::normal<Base::Matrix>(1, 2, urng);
    std::cout << "x = \n" << x << std::endl;
    AF::ReLU relu;
//    print(linearLayer.GetParameters());


//    Layers::Sequential nn(Layers::LinearLayer(2, 3, true, 45), AF::ReLU());
    Layers::Sequential nn(Layers::LinearLayer(2, 3, true, 45), AF::ReLU());
    std::cout << "--------" << std::endl;
    std::cout << nn(x) << std::endl;
    
    
    
    
    
//    std::cout << relu(linearLayer(x)) << std::endl;

    
}
